import logging
from typing import Any, Dict, List
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse, ResponseHandlingException

from app.config import settings
from app.services.embeddings import embed_text

logger = logging.getLogger("uvicorn.error")

_qdrant_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    """
    Returns an initialized QdrantClient connecting to Qdrant Cloud.
    Always uses settings.QDRANT_URL and settings.QDRANT_API_KEY.
    """
    global _qdrant_client
    if _qdrant_client is None:
        if not settings.QDRANT_URL or not settings.QDRANT_API_KEY:
            raise ValueError(
                "QDRANT_URL and QDRANT_API_KEY must be configured in environment / .env file."
            )
        _qdrant_client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=10.0,
        )
    return _qdrant_client


def search_context(query: str, collection_name: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Embeds the user query, searches the specified Qdrant Cloud collection,
    and returns a list of context dictionaries with text, source, chapter, verse, score.
    Handles non-existent collection warnings and cloud connection errors gracefully.
    """
    try:
        client = get_qdrant_client()
        query_vector = embed_text(query)

        # Execute search in Qdrant collection
        # Compatibility handling for qdrant-client versions
        if hasattr(client, "query_points"):
            response = client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )
            hits = response.points
        else:
            hits = client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
            )

        results = []
        for hit in hits:
            payload = hit.payload or {}
            results.append({
                "text": payload.get("text", ""),
                "sanskrit": payload.get("sanskrit", ""),
                "transliteration": payload.get("transliteration", ""),
                "translations": payload.get("translations", {}),
                "primary_translator": payload.get("primary_translator", ""),
                "source": payload.get("source", collection_name),
                "chapter": payload.get("chapter"),
                "verse": payload.get("verse"),
                "score": float(hit.score),
            })
        return results

    except UnexpectedResponse as e:
        if e.status_code == 404:
            logger.warning(
                f"[QDRANT WARNING] Collection '{collection_name}' does not exist on Qdrant Cloud. "
                "Returning empty context list."
            )
            return []
        logger.error(f"[QDRANT CLOUD ERROR] API Error searching '{collection_name}': {e.status_code} {e.content}")
        return []

    except ResponseHandlingException as e:
        logger.error(f"[QDRANT CLOUD ERROR] Qdrant exception while querying '{collection_name}': {e}")
        return []

    except Exception as e:
        logger.error(f"[QDRANT CLOUD ERROR] Unexpected error searching '{collection_name}': {e}")
        return []


def search_all_collections(
    query: str,
    collection_names: List[str],
    top_k_per_collection: int = 3
) -> List[Dict[str, Any]]:
    """
    Searches multiple Qdrant Cloud collections, merges results, and sorts by similarity score descending.
    """
    merged_results = []
    for coll in collection_names:
        hits = search_context(query, collection_name=coll, top_k=top_k_per_collection)
        merged_results.extend(hits)

    # Sort merged results by vector similarity score (descending)
    merged_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return merged_results
