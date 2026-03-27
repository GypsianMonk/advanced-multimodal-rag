"""
retrieval/hybrid.py
───────────────────
Hybrid retrieval: merges BM25 (keyword) and Vector (semantic) results
using Reciprocal Rank Fusion (RRF) for robust ranking.
"""

from __future__ import annotations

from loguru import logger

from retrieval.bm25 import BM25Retriever
from retrieval.vector import VectorRetriever


def _rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score."""
    return 1.0 / (k + rank)


class HybridRetriever:
    def __init__(
        self,
        bm25: BM25Retriever,
        vector: VectorRetriever,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
    ):
        self.bm25 = bm25
        self.vector = vector
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

    def search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 20,
        modality_filter: str | None = None,
    ) -> list[dict]:
        broad_k = top_k * 3

        bm25_results = self.bm25.search(query, top_k=broad_k)
        vector_results = self.vector.search(query_embedding, top_k=broad_k)

        if modality_filter:
            bm25_results = [r for r in bm25_results if r.get("type") == modality_filter]
            vector_results = [r for r in vector_results if r.get("type") == modality_filter]

        scores: dict[str, float] = {}
        doc_map: dict[str, dict] = {}

        for rank, doc in enumerate(bm25_results):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + self.bm25_weight * _rrf_score(rank)
            doc_map[doc_id] = doc

        for rank, doc in enumerate(vector_results):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + self.vector_weight * _rrf_score(rank)
            doc_map[doc_id] = doc

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for doc_id, fused_score in ranked:
            doc = {**doc_map[doc_id], "hybrid_score": fused_score}
            results.append(doc)

        logger.info(f"Hybrid retrieval → {len(results)} results (filter={modality_filter})")
        return results
