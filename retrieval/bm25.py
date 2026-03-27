"""
retrieval/bm25.py
─────────────────
BM25 keyword search using rank_bm25.
Handles tokenization, stopword removal, and top-K retrieval.
"""

from __future__ import annotations

import re
import pickle
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi
from loguru import logger

STOPWORDS = {
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "for",
    "of", "and", "or", "but", "not", "with", "this", "that", "was",
    "are", "be", "as", "by", "from", "has", "have", "had",
}


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


class BM25Retriever:
    def __init__(self):
        self.index: BM25Okapi | None = None
        self.documents: list[dict] = []

    def build(self, documents: list[dict]) -> None:
        """Build BM25 index from a list of document dicts (must have 'content' key)."""
        self.documents = documents
        corpus = [_tokenize(doc["content"]) for doc in documents]
        self.index = BM25Okapi(corpus)
        logger.success(f"BM25 index built with {len(documents)} documents")

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """Return top_k documents with BM25 scores."""
        if self.index is None:
            raise RuntimeError("BM25 index not built. Call .build() first.")
        tokens = _tokenize(query)
        scores = self.index.get_scores(tokens)
        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        results = []
        for idx, score in ranked:
            if score > 0:
                results.append({**self.documents[idx], "bm25_score": float(score)})
        return results

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"index": self.index, "documents": self.documents}, f)
        logger.info(f"BM25 index saved to {path}")

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.index = data["index"]
        self.documents = data["documents"]
        logger.info(f"BM25 index loaded from {path} ({len(self.documents)} docs)")
