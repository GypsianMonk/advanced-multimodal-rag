"""
embeddings/text_embedder.py
───────────────────────────
Sentence-Transformers based text embedding with caching.
"""

from __future__ import annotations

import numpy as np
from functools import lru_cache
from loguru import logger


class TextEmbedder:
    """Wraps a SentenceTransformer model for text embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading text embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        logger.success(f"Text embedder ready — dim={self.dim}")

    def embed(self, text: str) -> list[float]:
        """Embed a single string."""
        vec = self.model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """Embed a list of strings efficiently."""
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )
        return vecs.tolist()

    def similarity(self, a: list[float], b: list[float]) -> float:
        """Cosine similarity between two embeddings."""
        va, vb = np.array(a), np.array(b)
        return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))
