"""
retrieval/vector.py
───────────────────
Dense vector search using FAISS.
Supports cosine similarity via inner product on normalized vectors.
"""

from __future__ import annotations

import numpy as np
import faiss
from loguru import logger


class VectorRetriever:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # Inner product = cosine on normalized vecs
        self.documents: list[dict] = []

    def add(self, documents: list[dict]) -> None:
        """Add documents (must include 'embedding' key) to the FAISS index."""
        vecs = np.array([doc["embedding"] for doc in documents], dtype="float32")
        faiss.normalize_L2(vecs)
        self.index.add(vecs)
        self.documents.extend(documents)
        logger.info(f"FAISS index now holds {self.index.ntotal} vectors")

    def search(self, query_embedding: list[float], top_k: int = 20) -> list[dict]:
        """Return top_k documents by cosine similarity."""
        q = np.array([query_embedding], dtype="float32")
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append({**self.documents[idx], "vector_score": float(score)})
        return results

    def save(self, index_path: str, docs_path: str) -> None:
        faiss.write_index(self.index, index_path)
        import pickle
        with open(docs_path, "wb") as f:
            pickle.dump(self.documents, f)
        logger.info(f"FAISS index saved to {index_path}")

    def load(self, index_path: str, docs_path: str) -> None:
        self.index = faiss.read_index(index_path)
        import pickle
        with open(docs_path, "rb") as f:
            self.documents = pickle.load(f)
        logger.info(f"FAISS index loaded: {self.index.ntotal} vectors, {len(self.documents)} docs")
