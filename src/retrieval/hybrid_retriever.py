"""
Hybrid retriever: BM25 sparse + ChromaDB dense, fused with Reciprocal Rank Fusion.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.agents.state import Document

logger = logging.getLogger(__name__)

_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"


@dataclass
class HybridRetriever:
    chroma_client: chromadb.Client
    collection_name: str = "financial_docs"
    bm25_index: Optional[BM25Okapi] = field(default=None, repr=False)
    bm25_docs: list[Document] = field(default_factory=list, repr=False)
    _reranker: Optional[CrossEncoder] = field(default=None, repr=False)
    rrf_k: int = 60

    @classmethod
    def from_config(
        cls,
        persist_dir: str = "./data/chroma",
        collection_name: str = "financial_docs",
        load_reranker: bool = True,
    ) -> "HybridRetriever":
        client = chromadb.PersistentClient(path=persist_dir)
        instance = cls(chroma_client=client, collection_name=collection_name)

        instance._rebuild_bm25_from_chroma()

        if load_reranker:
            logger.info(f"Loading cross-encoder: {_RERANKER_MODEL}")
            instance._reranker = CrossEncoder(_RERANKER_MODEL)

        return instance

    def _get_collection(self) -> chromadb.Collection:
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=_EMBEDDING_MODEL
        )
        return self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )

    def _rebuild_bm25_from_chroma(self) -> None:
        try:
            collection = self._get_collection()
            result = collection.get(include=["documents", "metadatas"])
            if not result["documents"]:
                logger.warning("ChromaDB collection is empty. Run ingestion first.")
                return

            self.bm25_docs = [
                Document(
                    content=doc,
                    source=meta.get("source", ""),
                    ticker=meta.get("ticker", ""),
                    filing_type=meta.get("filing_type", ""),
                    year=meta.get("year", 0),
                    section=meta.get("section", ""),
                    relevance_score=0.0,
                    grade="",
                )
                for doc, meta in zip(result["documents"], result["metadatas"])
            ]

            tokenized = [doc["content"].lower().split() for doc in self.bm25_docs]
            self.bm25_index = BM25Okapi(tokenized)
            logger.info(f"BM25 index built over {len(self.bm25_docs)} documents")

        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")

    def _dense_retrieve(self, query: str, top_k: int) -> list[tuple[Document, int]]:
        collection = self._get_collection()
        results = collection.query(
            query_texts=[query],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        docs_with_ranks = []
        for rank, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            similarity = 1 - dist
            document = Document(
                content=doc,
                source=meta.get("source", ""),
                ticker=meta.get("ticker", ""),
                filing_type=meta.get("filing_type", ""),
                year=meta.get("year", 0),
                section=meta.get("section", ""),
                relevance_score=float(similarity),
                grade="",
            )
            docs_with_ranks.append((document, rank))

        return docs_with_ranks

    def _bm25_retrieve(self, query: str, top_k: int) -> list[tuple[Document, int]]:
        if self.bm25_index is None:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25_index.get_scores(tokenized_query)

        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]

        docs_with_ranks = []
        for rank, idx in enumerate(top_indices):
            doc = {**self.bm25_docs[idx], "relevance_score": float(scores[idx])}
            docs_with_ranks.append((doc, rank))

        return docs_with_ranks

    @staticmethod
    def _reciprocal_rank_fusion(
        *ranked_lists: list[tuple[Document, int]],
        k: int = 60,
    ) -> list[Document]:
        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for ranked_list in ranked_lists:
            for doc, rank in ranked_list:
                key = hash(doc["content"][:200])
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank + 1)
                doc_map[key] = doc

        sorted_keys = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)
        return [
            {**doc_map[k], "relevance_score": rrf_scores[k]}
            for k in sorted_keys
        ]

    def retrieve(self, query: str, top_k: int = 10) -> list[Document]:
        dense_results = self._dense_retrieve(query, top_k)
        sparse_results = self._bm25_retrieve(query, top_k)
        merged = self._reciprocal_rank_fusion(dense_results, sparse_results, k=self.rrf_k)
        return merged[:top_k]

    def rerank(self, query: str, docs: list[Document], top_n: int = 5) -> list[Document]:
        if self._reranker is None or not docs:
            return docs[:top_n]

        pairs = [(query, doc["content"]) for doc in docs]
        scores = self._reranker.predict(pairs)

        reranked = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            {**doc, "relevance_score": float(score)}
            for doc, score in reranked[:top_n]
        ]
