import logging
import time
from typing import List, Union
import httpx
from app.config import settings

logger = logging.getLogger(__name__)


def _get_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if hasattr(settings, "HF_TOKEN") and settings.HF_TOKEN:
        headers["Authorization"] = f"Bearer {settings.HF_TOKEN.strip()}"
    return headers


def _mean_pooling(embeddings_3d: List[List[List[float]]]) -> List[List[float]]:
    """
    Performs mean pooling over sequence length for 3D token embeddings [batch_size, seq_len, dim].
    """
    pooled_batch = []
    for seq in embeddings_3d:
        if not seq:
            continue
        dim = len(seq[0])
        seq_len = len(seq)
        sum_vec = [0.0] * dim
        for token_vec in seq:
            for d in range(dim):
                sum_vec[d] += token_vec[d]
        pooled_batch.append([val / seq_len for val in sum_vec])
    return pooled_batch


def _process_hf_response(data: Union[List, dict]) -> List[List[float]]:
    """
    Parses HuggingFace Inference API response into List[List[float]].
    Handles OpenAI format, 2D pooled vectors, and 3D token vectors.
    """
    if isinstance(data, dict):
        if "data" in data:  # OpenAI compatible format
            return [item["embedding"] for item in data["data"]]
        if "error" in data:
            raise RuntimeError(f"HuggingFace API Error: {data.get('error')}")

    if isinstance(data, list):
        if not data:
            return []
        # If single text returned 1D vector [dim]
        if isinstance(data[0], (int, float)):
            return [data]
        # If 2D vector [batch_size, dim]
        if isinstance(data[0], list) and data[0] and isinstance(data[0][0], (int, float)):
            return data
        # If 3D token vectors [batch_size, seq_len, dim]
        if isinstance(data[0], list) and data[0] and isinstance(data[0][0], list):
            return _mean_pooling(data)

    raise ValueError(f"Unexpected response format from HuggingFace API: {type(data)}")


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Batch generates embedding vectors using HuggingFace Serverless Inference API.
    Does NOT require local PyTorch/SentenceTransformers (uses minimal RAM).
    """
    if not texts:
        return []

    model_name = settings.EMBEDDING_MODEL_NAME.strip()
    if "/" not in model_name:
        model_name = f"sentence-transformers/{model_name}"

    # Hugging Face Router requires feature-extraction tagged models (e.g. BAAI/bge-small-en-v1.5)
    # If a sentence-similarity tagged model is requested, fall back to BAAI/bge-small-en-v1.5 (384 dim)
    if "sentence-transformers" in model_name:
        model_name = "BAAI/bge-small-en-v1.5"

    url = f"https://router.huggingface.co/hf-inference/models/{model_name}"
    headers = _get_headers()
    payload = {
        "inputs": texts,
        "options": {"wait_for_model": True}
    }

    logger.debug("Requesting embeddings for %d text(s) using model '%s'", len(texts), model_name)

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, json=payload, headers=headers)

            if response.status_code == 401:
                logger.error(
                    "[EMBEDDINGS ERROR] HuggingFace API 401 Unauthorized. "
                    "Please set a valid HF_TOKEN in your .env file."
                )
                raise RuntimeError(
                    "HuggingFace API 401 Unauthorized. Please set HF_TOKEN in environment/.env."
                )

            if response.status_code == 503:
                logger.info(f"[EMBEDDINGS] Model '{model_name}' is warming up on HuggingFace, waiting...")
                time.sleep(5)
                response = client.post(url, json=payload, headers=headers)

            response.raise_for_status()
            data = response.json()
            vectors = _process_hf_response(data)
            logger.debug("Received %d embedding vector(s) from HuggingFace API", len(vectors))
            return vectors

    except httpx.HTTPStatusError as e:
        logger.error(f"[EMBEDDINGS ERROR] HTTP error from HuggingFace API: {e.response.status_code} - {e.response.text}")
        raise RuntimeError(f"HuggingFace API HTTP Error: {e.response.status_code}") from e
    except Exception as e:
        logger.error(f"[EMBEDDINGS ERROR] Failed to generate embeddings via HuggingFace API: {e}")
        raise e


def embed_text(text: str) -> List[float]:
    """
    Generates an embedding vector for a single string.
    """
    results = embed_texts([text])
    if not results:
        return []
    return results[0]
