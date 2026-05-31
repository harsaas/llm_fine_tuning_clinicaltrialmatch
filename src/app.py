import os
import streamlit as st
import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# 🔑 Load variables from the local .env file
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
DEFAULT_HF_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
HF_MODEL_ID = os.getenv("HF_MODEL_ID") or DEFAULT_HF_MODEL_ID

# If the router rejects HF_MODEL_ID as unsupported, try these in order.
# Keep this list short and broadly compatible; users can override via .env.
FALLBACK_MODEL_IDS = [
    os.getenv("HF_FALLBACK_MODEL_ID"),
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-2-2b-it",
    "Qwen/Qwen2.5-7B-Instruct",
]
FALLBACK_MODEL_IDS = [m for m in FALLBACK_MODEL_IDS if m and m != HF_MODEL_ID]

st.set_page_config(page_title="TrialMatch-LLM Demo", page_icon="🧬", layout="wide")

st.title("🧬 TrialMatch-LLM: Precision Clinical Trial Matching")
st.markdown("---")

if not os.getenv("HF_MODEL_ID"):
    st.warning(
        f"HF_MODEL_ID is not set in .env — using default: {DEFAULT_HF_MODEL_ID}. "
        "Add HF_MODEL_ID to .env to override."
    )

# 🔗 Hugging Face endpoints
# Primary: https://api-inference.huggingface.co/models/{model_id}
# Fallback: https://router.huggingface.co/v1/chat/completions (OpenAI-compatible)
# (Some networks fail DNS for api-inference; router often still works.)
HF_ENDPOINTS = [
    {
        "kind": "classic",
        "url": f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}",
    },
    {
        "kind": "router_chat",
        "url": "https://router.huggingface.co/v1/chat/completions",
    },
]

QDRANT_PATH = "trial_store.db"
QDRANT_COLLECTION = "clinical_trials"


@st.cache_resource
def get_qdrant_client() -> QdrantClient:
    # Cache to avoid re-opening the local store on every Streamlit rerun.
    return QdrantClient(path=QDRANT_PATH)


@st.cache_resource
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def qdrant_top_hit(client: QdrantClient, query_vector: list[float], limit: int = 1):
    """Compatibility wrapper: qdrant-client has drifted between `search` and `query_points` APIs."""
    if hasattr(client, "search"):
        return client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_vector,
            limit=limit,
        )

    # Newer API
    res = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )
    return getattr(res, "points", res)

def query_huggingface_verdict(*, prompt: str, messages: list[dict], max_tokens: int, temperature: float):
    """Calls HF classic inference API, then falls back to HF router chat-completions.

    Returns: {"text": str, "endpoint": str, "model_used": str} or {"error": str, "details": ..., "endpoint": str}
    """
    if not HF_TOKEN:
        st.error("Missing HF_TOKEN! Please check your local .env file setup.")
        return {"error": "Missing HF_TOKEN"}
        
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    def _post_and_json(url: str, payload: dict):
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        try:
            return resp, resp.json()
        except ValueError:
            return resp, None

    last_exc: Exception | None = None
    response = None
    data = None
    used = None
    model_used = None

    for endpoint in HF_ENDPOINTS:
        try:
            if endpoint["kind"] == "classic":
                payload = {
                    "inputs": prompt,
                    "parameters": {"max_new_tokens": max_tokens, "temperature": temperature},
                }
            else:
                # Try configured model first, then a small fallback list if router says it's unsupported.
                candidate_models = [HF_MODEL_ID] + FALLBACK_MODEL_IDS
                for candidate in candidate_models:
                    payload = {
                        "model": candidate,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "stream": False,
                    }
                    response, data = _post_and_json(endpoint["url"], payload)
                    used = endpoint
                    model_used = candidate

                    # If model not supported, keep trying fallbacks.
                    if response.status_code == 400 and isinstance(data, dict):
                        err = data.get("error")
                        if isinstance(err, dict) and err.get("code") == "model_not_supported":
                            continue
                    break
                break

            response, data = _post_and_json(endpoint["url"], payload)
            used = endpoint
            model_used = HF_MODEL_ID
            break
        except requests.RequestException as e:
            last_exc = e
            continue

    if response is None:
        return {"error": f"Request failed (all endpoints): {last_exc}"}

    # If JSON parsing failed, include text preview.
    if data is None:
        text_preview = (response.text or "").strip()[:500]
        return {
            "error": f"Non-JSON response from Hugging Face (status {response.status_code}).",
            "raw": text_preview,
            "endpoint": used["url"] if used else None,
        }

    # Friendly guidance for common HF errors.
    if isinstance(data, dict) and "error" in data:
        err_obj = data.get("error")
        err_msg = err_obj if isinstance(err_obj, str) else err_obj.get("message") if isinstance(err_obj, dict) else None
        if isinstance(err_msg, str) and (
            "does not have sufficient permissions" in err_msg or "Inference Providers" in err_msg
        ):
            return {
                "error": (
                    "Your HF token does not have permission to call Hugging Face Inference Providers (router endpoint). "
                    "Fix options: (1) Create a new fine-grained HF token with 'Make calls to Inference Providers' permission "
                    "and replace HF_TOKEN in .env; or (2) fix DNS/network so api-inference.huggingface.co resolves, so the app can use the classic endpoint."
                ),
                "endpoint": used["url"] if used else None,
                "details": err_msg,
            }
        if isinstance(err_obj, dict) and err_obj.get("code") == "model_not_supported":
            return {
                "error": (
                    "Hugging Face router could not run the requested model with any of your enabled providers. "
                    "Fix: either enable a provider that supports this model (HF Settings → Inference Providers), or set HF_MODEL_ID in .env to a supported chat model."
                ),
                "endpoint": used["url"] if used else None,
                "details": err_obj,
                "model_used": model_used,
            }

    # Normalize output across endpoint kinds.
    if response.status_code >= 400:
        return {
            "error": f"HTTP {response.status_code}",
            "endpoint": used["url"] if used else None,
            "model_used": model_used,
            "details": data,
        }

    if used and used["kind"] == "classic":
        # Classic inference API typically returns a list like [{"generated_text": "..."}]
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return {
                "text": data[0].get("generated_text", ""),
                "endpoint": used["url"],
                "model_used": model_used,
            }
        return {
            "error": "Unexpected classic inference response.",
            "endpoint": used["url"],
            "details": data,
        }

    # Router chat-completions response (OpenAI-compatible)
    if isinstance(data, dict):
        try:
            return {
                "text": data["choices"][0]["message"]["content"],
                "endpoint": used["url"] if used else None,
                "model_used": model_used,
            }
        except Exception:
            return {
                "error": "Unexpected router chat response.",
                "endpoint": used["url"] if used else None,
                "model_used": model_used,
                "details": data,
            }

    return {
        "error": "Unexpected response payload type.",
        "endpoint": used["url"] if used else None,
        "model_used": model_used,
        "details": data,
    }

col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Step 1: Input Patient Electronic Health Record (EHR)")
    sample_note = (
        "Patient is a 54yo female diagnosed with stage III NSCLC. "
        "Biopsy confirmed positive for EGFR exon 19 deletion. "
        "Previously treated with cisplatin."
    )
    patient_ehr = st.text_area("Paste Doctor Notes:", value=sample_note, height=250)
    run_btn = st.button("Run Trial Match Engine", type="primary")

with col2:
    st.subheader("🤖 Step 2: Hybrid RAG + Fine-Tuned LLM Verdict")
    
    if run_btn:
        with st.spinner("Searching local Qdrant vectors & executing cloud reasoning..."):
            try:
                # 1. Fetch relevant criteria out of your local Qdrant DB
                qdrant_client = get_qdrant_client()
                embedder = get_embedder()
                
                query_vector = embedder.encode(patient_ehr).tolist()
                hits = qdrant_top_hit(qdrant_client, query_vector, limit=1)
                
                if hits:
                    chunk = hits[0].payload.get("criteria_chunk", "No criteria text available.")
                    title = hits[0].payload.get("title", "Unknown Study")
                    
                    st.info(f"**Top Retrieved Protocol:** {title}")
                    
                    # 2. Assemble the Hybrid RAG prompt sequence
                    system_msg = "You are an expert clinical trial matching AI. Analyze the patient EHR against the eligibility criteria."
                    user_msg = (
                        f"Patient EHR:\n{patient_ehr}\n\n"
                        f"Clinical Trial Protocol ({title}):\n{chunk}\n\n"
                        "Provide a clear verdict: MATCH / NO MATCH / UNCERTAIN, followed by a short rationale."
                    )

                    # Keep a single-string prompt for classic inference API, but use messages for router chat.
                    prompt = f"System: {system_msg}\n\nUser: {user_msg}\n\nAssistant:"
                    messages = [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ]
                    
                    # 3. Query the model endpoint (classic first, router chat fallback)
                    llm_response = query_huggingface_verdict(
                        prompt=prompt,
                        messages=messages,
                        max_tokens=500,
                        temperature=0.2,
                    )

                    # 4. Display result
                    if llm_response and llm_response.get("text"):
                        st.success("### Match Evaluation Complete")
                        if llm_response.get("model_used") and llm_response.get("model_used") != HF_MODEL_ID:
                            st.caption(
                                f"Used fallback model: {llm_response['model_used']} (HF_MODEL_ID={HF_MODEL_ID})"
                            )
                        st.write(llm_response["text"].strip())
                    elif llm_response and "error" in llm_response:
                        st.error(f"Hugging Face API Error: {llm_response['error']}")
                        if llm_response.get("endpoint"):
                            st.caption(f"Endpoint: {llm_response['endpoint']}")
                        if llm_response.get("model_used"):
                            st.caption(f"Model attempted: {llm_response['model_used']}")
                        if llm_response.get("details") is not None:
                            with st.expander("HF error details"):
                                st.json(llm_response.get("details"))
                    else:
                        st.error(f"Unexpected API response payload structure: {llm_response}")
                else:
                    st.warning("No matching clinical trials found in your local Qdrant DB.")
                    
            except Exception as e:
                st.error(f"An unexpected processing error occurred: {str(e)}")
