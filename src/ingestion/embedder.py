"""
Batch embedder: converts chunks to vectors and stores in ChromaDB.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm

from src.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
_BATCH_SIZE = 64


def embed_and_store(
    chunks: list[Chunk],
    persist_dir: str | Path = "./data/chroma",
    collection_name: str = "financial_docs",
    batch_size: int = _BATCH_SIZE,
) -> int:
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_dir))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=_EMBEDDING_MODEL,
    )
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    existing_ids: set[str] = set(collection.get(include=[])["ids"])

    new_chunks = []
    for chunk in chunks:
        chunk_id = _content_hash(chunk.content)
        if chunk_id not in existing_ids:
            new_chunks.append((chunk_id, chunk))

    if not new_chunks:
        logger.info("All chunks already embedded — skipping")
        return 0

    logger.info(f"Embedding {len(new_chunks)} new chunks (skipping {len(chunks) - len(new_chunks)} duplicates)")

    added = 0
    for i in tqdm(range(0, len(new_chunks), batch_size), desc="Embedding batches"):
        batch = new_chunks[i : i + batch_size]
        ids = [chunk_id for chunk_id, _ in batch]
        documents = [chunk.content for _, chunk in batch]
        metadatas = [
            {
                "source": chunk.source,
                "ticker": chunk.ticker,
                "filing_type": chunk.filing_type,
                "year": chunk.year,
                "section": chunk.section,
                "chunk_index": chunk.chunk_index,
            }
            for _, chunk in batch
        ]
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        added += len(batch)

    logger.info(f"Stored {added} new chunks in ChromaDB at {persist_dir}")
    return added


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
