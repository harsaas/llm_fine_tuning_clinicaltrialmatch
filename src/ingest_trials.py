from datasets import load_dataset
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: langchain-text-splitters. Install it with `pip install langchain-text-splitters`."
    ) from exc


def run_ingestion(
    dataset_name: str = "louisbrulenaudet/clinical-trials",
    split: str = "train",
    max_trials: int = 1000,
    collection_name: str = "clinical_trials",
    qdrant_path: str = "trial_store.db",
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    upsert_batch_size: int = 256,
) -> None:
    print("Loading dataset from Hugging Face...")
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    trials = list(dataset.take(max_trials))
    print(f"Successfully loaded {len(trials)} trial records.")

    print("Initializing embedding model...")
    embedding_model = SentenceTransformer(embedding_model_name)
    vector_dim = embedding_model.get_embedding_dimension()

    print("Initializing local Qdrant store...")
    client = QdrantClient(path=qdrant_path)
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    points: list[PointStruct] = []
    point_id = 1

    print("Processing and embedding eligibility criteria...")
    for trial in tqdm(trials, desc="Embedding trials"):
        nct_id = trial.get("id") or trial.get("nct_id") or "Unknown_ID"
        title = trial.get("title") or "No Title Available"

        eligibility = (
            trial.get("eligibility")
            or trial.get("eligibility_criteria")
            or trial.get("criteria")
            or ""
        )
        if not eligibility:
            continue

        eligibility_text = str(eligibility)
        chunks = splitter.split_text(eligibility_text)
        for chunk in chunks:
            vector = embedding_model.encode(chunk).tolist()
            payload = {"nct_id": nct_id, "title": title, "criteria_chunk": chunk}
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))
            point_id += 1

            if len(points) >= upsert_batch_size:
                client.upsert(collection_name=collection_name, points=points)
                points.clear()

    if points:
        client.upsert(collection_name=collection_name, points=points)

    print(f"✅ Ingestion complete! Vector DB at: {qdrant_path}")


if __name__ == "__main__":
    run_ingestion()



