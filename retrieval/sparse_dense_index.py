"""
retrieval/sparse_dense_index.py
────────────────────────────────
Unified index that keeps BM25 (sparse) and FAISS (dense) in sync.
Single add/search interface — no need to manage two indexes separately.
Supports incremental additions without full rebuild.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from loguru import logger

from retrieval.bm25 import BM25Retriever
from retrieval.vector import VectorRetriever
from retrieval.hybrid import HybridRetriever


class SparseDenseIndex:
    """
    Unified sparse + dense index with incremental update support.

    Usage:
        index = SparseDenseIndex(dim=384)
        index.add(docs)                        # add any time
        results = index.search(q, q_emb, k=10) # single call
        index.save("data/")                    # persist both indexes
        index.load("data/")                    # restore
    """

    def __init__(
        self,
        dim: int = 384,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
    ):
        self.dim = dim
        self._documents: list[dict] = []
        self._bm25 = BM25Retriever()
        self._vector = VectorRetriever(dim=dim)
        self._hybrid = HybridRetriever(
            self._bm25, self._vector, bm25_weight, vector_weight
        )
        self._dirty = False  # BM25 needs rebuild if True

    def add(self, documents: list[dict]) -> None:
        """Add documents incrementally. BM25 is rebuilt; FAISS appends."""
        if not documents:
            return
        self._documents.extend(documents)
        self._vector.add(documents)   # FAISS: O(n_new)
        self._dirty = True
        logger.info(f"Index now holds {len(self._documents)} documents")

    def _ensure_bm25_built(self) -> None:
        if self._dirty:
            self._bm25.build(self._documents)
            self._dirty = False

    def search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 20,
        modality_filter: str | None = None,
    ) -> list[dict]:
        self._ensure_bm25_built()
        return self._hybrid.search(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
            modality_filter=modality_filter,
        )

    def count(self) -> int:
        return len(self._documents)

    def save(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        self._ensure_bm25_built()

        import faiss
        faiss.write_index(self._vector.index, str(path / "faiss.index"))

        with open(path / "documents.pkl", "wb") as f:
            pickle.dump(self._documents, f)

        logger.success(f"Index saved to {directory} ({len(self._documents)} docs)")

    def load(self, directory: str) -> None:
        path = Path(directory)

        import faiss
        self._vector.index = faiss.read_index(str(path / "faiss.index"))

        with open(path / "documents.pkl", "rb") as f:
            self._documents = pickle.load(f)

        self._vector.documents = self._documents
        self._dirty = True   # will rebuild BM25 on next search
        logger.success(f"Index loaded from {directory} ({len(self._documents)} docs)")

    def stats(self) -> dict:
        return {
            "total_documents": len(self._documents),
            "faiss_vectors": self._vector.index.ntotal,
            "modalities": {
                m: sum(1 for d in self._documents if d.get("type") == m)
                for m in ("text", "image", "table")
            },
            "bm25_built": not self._dirty,
        }
