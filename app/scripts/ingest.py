import argparse
import asyncio
import logging
import sys
import time
import uuid
from typing import Any, Dict, List, Tuple

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from app.config import settings
from app.services.embeddings import embed_texts

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def extract_verse_translations(
    data: Dict[str, Any], primary_translator_req: str
) -> Tuple[str, str, str, Dict[str, Dict[str, str]]]:
    """
    Builds a translations dict for all commentators with non-empty 'et' fields.
    Returns: (primary_text, transliteration, primary_key_used, translations_dict)
    """
    translations: Dict[str, Dict[str, str]] = {}
    for k, v in data.items():
        if (
            isinstance(v, dict)
            and "et" in v
            and isinstance(v["et"], str)
            and v["et"].strip()
        ):
            translations[k] = {
                "author": str(v.get("author", k)),
                "translation": v["et"].strip(),
            }

    transliteration = str(data.get("transliteration", "")).strip()

    primary_text = ""
    primary_key_used = ""

    if primary_translator_req in translations:
        primary_key_used = primary_translator_req
        primary_text = translations[primary_translator_req]["translation"]
    else:
        fallback_order = [
            k for k in ["siva", "tej", "purohit", "prabhu"] if k != primary_translator_req
        ]
        for key in fallback_order:
            if key in translations:
                primary_key_used = key
                primary_text = translations[key]["translation"]
                break

        if not primary_text and translations:
            first_key = next(iter(translations))
            primary_key_used = first_key
            primary_text = translations[first_key]["translation"]

    return primary_text, transliteration, primary_key_used, translations


def build_embed_text(ch: int, v: int, transliteration: str, primary_text: str) -> str:
    """
    Combines chapter/verse reference + transliteration + English meaning
    into a single rich string for embedding. More signal = better retrieval.
    """
    parts = [f"Chapter {ch}, Verse {v}"]
    if transliteration:
        parts.append(transliteration)
    parts.append(primary_text)
    return " | ".join(parts)


async def flush_batch(
    client: QdrantClient,
    collection_name: str,
    source_name: str,
    batch_buffer: List[Dict[str, Any]],
) -> int:
    if not batch_buffer:
        return 0

    embed_inputs = [b["embed_text"] for b in batch_buffer]
    embeddings = embed_texts(embed_inputs)

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
                    "text": b["primary_text"],
                    "embed_text": b["embed_text"],
                    "sanskrit": b["sanskrit"],
                    "transliteration": b["transliteration"],
                    "chapter": ch,
                    "verse": v,
                    "source": source_name,
                    "primary_translator": b["primary_key_used"],
                    "translations": b["translations"],
                },
            )
        )

    client.upsert(collection_name=collection_name, points=points)
    return len(points)


async def run_ingest(
    collection_name: str,
    source_name: str,
    primary_translator: str,
    start_chapter: int,
    end_chapter: int,
    delay: float,
    batch_size: int,
):
    start_time = time.time()

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    async_http = httpx.AsyncClient(timeout=30.0, follow_redirects=True, headers=headers)

    qdrant = QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        timeout=120.0,
    )

    # Ensure collection exists
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
        await async_http.aclose()
        sys.exit(1)

    # Fetch chapter metadata
    print("Fetching chapter metadata from vedicscriptures.github.io...")
    try:
        resp = await async_http.get("https://vedicscriptures.github.io/chapters/")
        resp.raise_for_status()
        chapters_data = resp.json()
    except Exception as e:
        logger.error(f"Failed to fetch chapters metadata: {e}")
        await async_http.aclose()
        sys.exit(1)

    chapter_verse_counts: Dict[int, int] = {}
    for item in chapters_data:
        ch_num = int(item.get("chapter_number", 0))
        v_count = int(item.get("verses_count", 0))
        if ch_num > 0:
            chapter_verse_counts[ch_num] = v_count

    target_chapters = [
        ch for ch in range(start_chapter, end_chapter + 1) if ch in chapter_verse_counts
    ]
    if not target_chapters:
        logger.error(f"No chapters found in range {start_chapter}-{end_chapter}.")
        await async_http.aclose()
        sys.exit(1)

    total_expected = sum(chapter_verse_counts[ch] for ch in target_chapters)
    print(
        f"Ingesting {len(target_chapters)} chapters "
        f"(Ch {start_chapter}-{end_chapter}), {total_expected} verses planned."
    )

    total_ingested = 0
    skipped: List[str] = []
    batch_buffer: List[Dict[str, Any]] = []

    try:
        for ch in target_chapters:
            v_count = chapter_verse_counts[ch]
            ch_count = 0

            for v in range(1, v_count + 1):
                verse_id = f"BG{ch}.{v}"
                url = f"https://vedicscriptures.github.io/slok/{ch}/{v}/"

                try:
                    v_resp = await async_http.get(url)
                    v_resp.raise_for_status()
                    v_data = v_resp.json()
                except Exception as e:
                    logger.warning(f"Failed to fetch {verse_id}: {e}")
                    skipped.append(verse_id)
                    await asyncio.sleep(delay)
                    continue

                await asyncio.sleep(delay)

                sanskrit = str(v_data.get("slok", "")).strip()
                primary_text, transliteration, key_used, translations = \
                    extract_verse_translations(v_data, primary_translator)

                if not primary_text:
                    logger.warning(f"{verse_id}: no English translation found, skipping.")
                    skipped.append(verse_id)
                    continue

                embed_text = build_embed_text(ch, v, transliteration, primary_text)

                batch_buffer.append({
                    "chapter": ch,
                    "verse": v,
                    "primary_text": primary_text,
                    "embed_text": embed_text,
                    "primary_key_used": key_used,
                    "sanskrit": sanskrit,
                    "transliteration": transliteration,
                    "translations": translations,
                })
                ch_count += 1

                if len(batch_buffer) >= batch_size:
                    count = await flush_batch(qdrant, collection_name, source_name, batch_buffer)
                    total_ingested += count
                    print(f"Upserted batch of {count} verses (total: {total_ingested})")
                    batch_buffer.clear()

            print(f"Chapter {ch}: {ch_count}/{v_count} verses fetched and queued")

        if batch_buffer:
            count = await flush_batch(qdrant, collection_name, source_name, batch_buffer)
            total_ingested += count
            print(f"Upserted final batch of {count} verses (total: {total_ingested})")

    finally:
        await async_http.aclose()

    elapsed = time.time() - start_time
    print("\n" + "=" * 50)
    print("INGESTION SUMMARY")
    print("=" * 50)
    print(f"Total verses ingested : {total_ingested}")
    print(f"Total verses skipped  : {len(skipped)}")
    if skipped:
        print(f"Skipped               : {', '.join(skipped)}")
    print(f"Time taken            : {elapsed:.2f}s")
    print("=" * 50 + "\n")


async def main():
    parser = argparse.ArgumentParser(
        description="Ingest Bhagavad Gita verses from vedicscriptures API into Qdrant."
    )
    parser.add_argument("--collection", required=True, help="Qdrant collection name")
    parser.add_argument("--source", default="Bhagavad Gita", help="Source name tag")
    parser.add_argument(
        "--primary-translator", default="siva",
        help="Primary commentator key (default: siva)"
    )
    parser.add_argument("--start-chapter", type=int, default=1)
    parser.add_argument("--end-chapter", type=int, default=18)
    parser.add_argument(
        "--delay", type=float, default=0.3,
        help="Delay between API calls in seconds (default: 0.3)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=20,
        help="Verses per embedding batch (default: 20)"
    )

    args = parser.parse_args()

    await run_ingest(
        collection_name=args.collection,
        source_name=args.source,
        primary_translator=args.primary_translator,
        start_chapter=args.start_chapter,
        end_chapter=args.end_chapter,
        delay=args.delay,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    asyncio.run(main())
