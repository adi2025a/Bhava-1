import argparse
import os
import sys
import uuid
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from app.config import settings
from app.services.embeddings import embed_texts


def run_ingest(file_path: str, source_name: str, collection_name: str, chunk_size: int, chunk_overlap: int, batch_size: int):
    """
    Reads a local text file, chunks it, embeds chunks in batches,
    and upserts points into a Qdrant Cloud vector collection.
    """
    if not os.path.exists(file_path):
        sys.exit(1)

    with open(file_path, "r", encoding="utf-8") as f:
        text_content = f.read()

    if not text_content.strip():
        sys.exit(1)

    # 1. Chunk text using RecursiveCharacterTextSplitter
    print(f"Splitting text into chunks (chunk_size={chunk_size}, chunk_overlap={chunk_overlap})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_text(text_content)
    total_chunks = len(chunks)
    print(f"Generated {total_chunks} text chunks.")

    # 2. Connect to Qdrant Cloud
    client = QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        timeout=30.0,
    )

    # 3. Create collection if it does not exist
    collections_response = client.get_collections()
    existing_collections = [c.name for c in collections_response.collections]

    vector_dimension = 384

    if collection_name not in existing_collections:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_dimension,
                distance=Distance.COSINE,
            ),
        )

    for i in range(0, total_chunks, batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_embeddings = embed_texts(batch_chunks)

        points: List[PointStruct] = []
        for idx, (chunk_text, vector) in enumerate(zip(batch_chunks, batch_embeddings)):
            chunk_index = i + idx
            point_id = str(uuid.uuid4())
            payload = {
                "text": chunk_text,
                "source": source_name,
                "chunk_index": chunk_index,
            }
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        client.upsert(
            collection_name=collection_name,
            points=points,
        )

        upserted_count = min(i + batch_size, total_chunks)
        print(f"Upserted {upserted_count}/{total_chunks} chunks to '{collection_name}'")

    print(f"\n[SUCCESS] Ingestion completed successfully! {total_chunks} chunks indexed in Qdrant collection '{collection_name}'.")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest text files into Qdrant Cloud for RAG context retrieval."
    )
    parser.add_argument(
        "--file",
        required=True,
        help="Path to source text file (e.g. data/gita.txt)",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Name of the text source (e.g. 'Bhagavad Gita')",
    )
    parser.add_argument(
        "--collection",
        required=True,
        help="Qdrant collection name (e.g. 'gita_collection')",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Character size per chunk (default: 500)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Character overlap between chunks (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Upsert batch size (default: 100)",
    )

    args = parser.parse_args()
    run_ingest(
        file_path=args.file,
        source_name=args.source,
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
