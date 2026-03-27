"""
ranking/reranker.py
───────────────────
Multi-stage reranking:
  - Cross-encoder for text documents
  - CLIP similarity for image documents
"""

from __future__ import annotations

from loguru import logger


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading cross-encoder: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: list[dict], top_k: int = 10) -> list[dict]:
        text_docs = [d for d in documents if d.get("type") != "image"]
        image_docs = [d for d in documents if d.get("type") == "image"]

        if text_docs:
            pairs = [(query, doc["content"]) for doc in text_docs]
            scores = self.model.predict(pairs)
            for doc, score in zip(text_docs, scores):
                doc["rerank_score"] = float(score)
            text_docs.sort(key=lambda d: d["rerank_score"], reverse=True)

        combined = text_docs + image_docs
        return combined[:top_k]


class CLIPReranker:
    def __init__(self, image_embedder):
        self.image_embedder = image_embedder

    def rerank_images(self, query: str, image_docs: list[dict], top_k: int = 5) -> list[dict]:
        for doc in image_docs:
            score = self.image_embedder.image_text_similarity(doc["content"], query)
            doc["clip_score"] = float(score)
        image_docs.sort(key=lambda d: d["clip_score"], reverse=True)
        return image_docs[:top_k]


class MultiModalReranker:
    """Unified reranker that handles both text and image documents."""

    def __init__(self, text_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", image_embedder=None):
        self.text_reranker = CrossEncoderReranker(text_model)
        self.clip_reranker = CLIPReranker(image_embedder) if image_embedder else None

    def rerank(self, query: str, documents: list[dict], top_k: int = 10) -> list[dict]:
        text_docs = [d for d in documents if d.get("type") != "image"]
        image_docs = [d for d in documents if d.get("type") == "image"]

        reranked_text = self.text_reranker.rerank(query, text_docs, top_k=top_k)

        if self.clip_reranker and image_docs:
            reranked_images = self.clip_reranker.rerank_images(query, image_docs, top_k=max(2, top_k // 5))
        else:
            reranked_images = image_docs[:2]

        final = reranked_text + reranked_images
        logger.info(f"Reranked → {len(final)} docs ({len(reranked_text)} text, {len(reranked_images)} images)")
        return final[:top_k]
