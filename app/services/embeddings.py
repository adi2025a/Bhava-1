from typing import List
from sentence_transformers import SentenceTransformer
from app.config import settings

# Global singleton instance for the sentence transformer model
_model: SentenceTransformer | None = None


def load_embedding_model() -> SentenceTransformer:
    """
    Loads and returns the SentenceTransformer embedding model.
    Warms up the model so loading happens once at module/app startup rather than per request.
    """
    global _model
    if _model is None:
        model_name = settings.EMBEDDING_MODEL_NAME
        print(f"[EMBEDDINGS] Loading SentenceTransformer model: {model_name}...")
        _model = SentenceTransformer(model_name)
        print("[EMBEDDINGS] SentenceTransformer model loaded successfully.")
    return _model


def embed_text(text: str) -> List[float]:
    """
    Generates an embedding vector for a single string.
    """
    model = load_embedding_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Batch generates embedding vectors for a list of strings (used in ingestion).
    """
    if not texts:
        return []
    model = load_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True, batch_size=32, show_progress_bar=False)
    return embeddings.tolist()
