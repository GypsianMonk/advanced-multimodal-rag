"""
fusion/diversity_ranker.py
──────────────────────────
Maximal Marginal Relevance (MMR) re-ranking for result diversity.
Prevents the context window from being filled with near-duplicate chunks
by balancing relevance vs novelty.

MMR formula:
  score = λ * relevance(doc, query)
        - (1 - λ) * max_similarity(doc, already_selected)

λ = 1.0 → pure relevance (standard top-K)
λ = 0.0 → pure diversity (maximum spread)
λ = 0.5 → balanced (recommended default)
"""

from __future__ import annotations

import numpy as np
from loguru import logger


def cosine_sim(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / (denom + 1e-9))


class MMRRanker:
    """
    Maximal Marginal Relevance ranker.
    Requires documents to have an 'embedding' key.
    """

    def __init__(self, lambda_param: float = 0.5):
        """
        Args:
            lambda_param: Trade-off between relevance and diversity.
                          0.0 = maximum diversity, 1.0 = pure relevance.
        """
        self.lambda_param = lambda_param

    def rank(
        self,
        documents: list[dict],
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[dict]:
        """
        Apply MMR to select top_k diverse + relevant documents.

        Args:
            documents: Candidates with 'embedding' key and a relevance score.
            query_embedding: Embedded query vector.
            top_k: Number of documents to select.

        Returns:
            Selected documents in MMR-ranked order.
        """
        if not documents:
            return []

        # Validate embeddings exist
        docs_with_emb = [d for d in documents if d.get("embedding")]
        if not docs_with_emb:
            logger.warning("MMR: no embeddings found, returning top-K by score")
            return sorted(
                documents,
                key=lambda d: d.get("hybrid_score", d.get("rerank_score", 0)),
                reverse=True,
            )[:top_k]

        selected: list[dict] = []
        remaining = docs_with_emb[:]

        while remaining and len(selected) < top_k:
            best_doc = None
            best_mmr = float("-inf")

            for doc in remaining:
                # Relevance: cosine similarity to query
                relevance = cosine_sim(doc["embedding"], query_embedding)

                # Redundancy: max similarity to already selected docs
                if selected:
                    redundancy = max(
                        cosine_sim(doc["embedding"], s["embedding"])
                        for s in selected
                    )
                else:
                    redundancy = 0.0

                mmr_score = (
                    self.lambda_param * relevance
                    - (1 - self.lambda_param) * redundancy
                )

                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_doc = doc

            if best_doc:
                best_doc = {**best_doc, "mmr_score": round(best_mmr, 4)}
                selected.append(best_doc)
                remaining.remove(next(d for d in remaining if d["id"] == best_doc["id"]))

        logger.info(
            f"MMR: selected {len(selected)}/{len(documents)} docs "
            f"(λ={self.lambda_param})"
        )
        return selected
