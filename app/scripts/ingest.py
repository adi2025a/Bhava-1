import argparse
import logging
import re
import sys
import time
import uuid
from typing import Any, Dict, List

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from app.config import settings
from app.services.embeddings import embed_texts

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def parse_geeta_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Parses geeta.txt where each verse block is:
      Chapter X, Verse Y
      Sanskrit: ...
      Meaning: ...
    Each verse becomes one chunk.
    """
    verses = []
    current: Dict[str, Any] = {}

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            header = re.match(r"Chapter\s+(\d+),\s+Verse\s+(\d+)", line)
            if header:
                if current.get("meaning"):
                    verses.append(current)
                current = {
                    "chapter": int(header.group(1)),
                    "verse": int(header.group(2)),
                    "sanskrit": "",
                    "meaning": "",
                }
                continue

            if line.startswith("Sanskrit:"):
                current["sanskrit"] = line[len("Sanskrit:"):].strip()
                continue

            if line.startswith("Meaning:"):
                current["meaning"] = line[len("Meaning:"):].strip()
                continue

    if current.get("meaning"):
        verses.append(current)

    return verses


def flush_batch(
    client: QdrantClient,
    collection_name: str,
    source_name: str,
    batch_buffer: List[Dict[str, Any]],
) -> int:
    if not batch_buffer:
        return 0

    texts = [b["meaning"] for b in batch_buffer]
    embeddings = embed_texts(texts)

    points: List[PointStruct] = []
    for b, emb in zip(batch_buffer, embeddings):
        ch = b["chapter"]
        v = b["verse"]
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"BG{ch}.{v}"))
        points.append(
            PointStruct(
                id=point_id,
                vector=emb,
                payload={
                    "text": b["meaning"],
                    "sanskrit": b["sanskrit"],
                    "chapter": ch,
                    "verse": v,
                    "source": source_name,
                },
            )
        )

    client.upsert(collection_name=collection_name, points=points)
    return len(points)


def run_ingest(
    file_path: str,
    collection_name: str,
    source_name: str,
    batch_size: int,
) -> None:
    start_time = time.time()

    print(f"Parsing verses from '{file_path}'...")
    verses = parse_geeta_file(file_path)
    if not verses:
        logger.error("No verses parsed. Check the file format.")
        sys.exit(1)
    print(f"Parsed {len(verses)} verses.")

    qdrant = QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        timeout=120.0,
    )

    try:
        existing = [c.name for c in qdrant.get_collections().collections]
        if collection_name not in existing:
            print(f"Creating collection '{collection_name}' with 384-dim COSINE vectors...")
            qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
    except Exception as e:
        logger.error(f"Failed to check/create Qdrant collection: {e}")
        sys.exit(1)

    total_ingested = 0
    batch_buffer: List[Dict[str, Any]] = []

    for verse in verses:
        batch_buffer.append(verse)
        if len(batch_buffer) >= batch_size:
            count = flush_batch(qdrant, collection_name, source_name, batch_buffer)
            total_ingested += count
            print(f"Upserted batch of {count} verses (total: {total_ingested})")
            batch_buffer.clear()

    if batch_buffer:
        count = flush_batch(qdrant, collection_name, source_name, batch_buffer)
        total_ingested += count
        print(f"Upserted final batch of {count} verses (total: {total_ingested})")

    elapsed = time.time() - start_time
    print("\n" + "=" * 50)
    print("INGESTION SUMMARY")
    print("=" * 50)
    print(f"Total verses ingested: {total_ingested}")
    print(f"Total time: {elapsed:.2f} seconds")
    print("=" * 50 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest Bhagavad Gita verses from geeta.txt into Qdrant."
    )
    parser.add_argument(
        "--file",
        default="app/scripts/geeta.txt",
        help="Path to geeta.txt (default: app/scripts/geeta.txt)",
    )
    parser.add_argument(
        "--collection",
        required=True,
        help="Qdrant collection name (e.g. 'gita_collection')",
    )
    parser.add_argument(
        "--source",
        default="Bhagavad Gita",
        help="Source name tag stored in each point's payload (default: 'Bhagavad Gita')",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of verses per embedding batch (default: 50)",
    )

    args = parser.parse_args()

    run_ingest(
        file_path=args.file,
        collection_name=args.collection,
        source_name=args.source,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
