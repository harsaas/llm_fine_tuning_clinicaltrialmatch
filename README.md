# TrialMatch-LLM (Fine-Tuning + RAG Demo)

This repo contains a simple end-to-end **clinical trial matching** demo:

1. **Ingest** clinical trial eligibility criteria into a local **Qdrant** vector store (`trial_store.db`).
2. Run a **Streamlit** app where you paste a patient EHR note.
3. The app retrieves the most relevant trial criteria chunk (RAG) and asks a cloud LLM (via **Hugging Face Inference Router**) to produce a verdict: **MATCH / NO MATCH / UNCERTAIN**.

## What’s In Here

- [src/ingest_trials.py](src/ingest_trials.py): Downloads trial records, chunks eligibility criteria, embeds them, and writes a local Qdrant store.
- [src/app.py](src/app.py): Streamlit UI (EHR input → Qdrant retrieval → HF verdict).
- [src/format_data.py](src/format_data.py): Generates synthetic chat-style `train.jsonl`/`val.jsonl` under `data/` (optional).
- [src/train.py](src/train.py) + [evaluation/evaluate_model.py](evaluation/evaluate_model.py): Optional LoRA fine-tuning/eval scripts (GPU/Colab-oriented; may not run on Windows).

## Prerequisites

- Python **3.10+**
- Internet access to download:
  - the clinical trials dataset from Hugging Face Datasets
  - the embedding model (`sentence-transformers/all-MiniLM-L6-v2`)
  - a chat model via Hugging Face Inference Router

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
./.venv/Scripts/Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Environment (.env)

Create a `.env` file in the repo root (same folder as this README):

```env
# Required for Hugging Face calls
HF_TOKEN=hf_xxx

# Optional: model selection (app falls back automatically if unsupported)
HF_MODEL_ID=meta-llama/Meta-Llama-3-8B-Instruct
HF_FALLBACK_MODEL_ID=mistralai/Mistral-7B-Instruct-v0.3
```

Notes:

- `HF_TOKEN` should be a fine-grained token with permission to **“Make calls to Inference Providers”** (needed for the router endpoint).
- If you omit `HF_MODEL_ID`, the app defaults to `Qwen/Qwen2.5-1.5B-Instruct` (but it may be rejected by your enabled providers).

## Step 1 — Build the Local Vector DB (Qdrant)

Run ingestion (creates/overwrites `trial_store.db` and the `clinical_trials` collection):

```powershell
python src/ingest_trials.py
```

By default, ingestion reads up to ~1000 trial records from `louisbrulenaudet/clinical-trials` and embeds eligibility text into Qdrant.

## Step 2 — Run the Streamlit Demo

```powershell
streamlit run src/app.py
```

In the UI:

- Paste a patient note into **“Paste Doctor Notes”**
- Click **“Run Trial Match Engine”**
- The app shows the **top retrieved protocol** and the LLM verdict

## Troubleshooting

### Qdrant “already accessed by another instance”

If you see an error like:

> Storage folder trial_store.db is already accessed by another instance of Qdrant client

Fix:

- Stop other running `streamlit`/Python processes that might have `trial_store.db` open.
- Restart the Streamlit app.
- If you need concurrent access, run a Qdrant server instead of local-file mode.

### Hugging Face router errors

- **Permission error** (Inference Providers): create a new token with the proper permission and update `HF_TOKEN`.
- **`model_not_supported`**: set `HF_MODEL_ID` to a model supported by your enabled providers, or enable a provider that supports your chosen model.

### DNS issues reaching `api-inference.huggingface.co`

Some networks fail DNS for the classic HF inference endpoint. The app uses `https://router.huggingface.co/v1/chat/completions` as a fallback.

## Optional: Fine-Tuning & Evaluation

This repo includes scripts for generating synthetic chat data and doing LoRA fine-tuning/evaluation. These are typically run on a GPU environment (e.g., Colab) and may require extra dependencies (e.g., `unsloth`) and CUDA.

- Generate synthetic data: run the `prepare_pipeline_datasets(...)` function in [src/format_data.py](src/format_data.py) (writes `data/train.jsonl` and `data/val.jsonl`).
- Fine-tune: [src/train.py](src/train.py)
- Evaluate: [evaluation/evaluate_model.py](evaluation/evaluate_model.py)

## Security

- Do not commit your `.env` (tokens) to git.
