"""
fusion/context_fusion.py
────────────────────────
Aggregates and scores multi-modal retrieval results into a
unified context window ready for LLM generation.
"""

from __future__ import annotations

from datetime import datetime
from loguru import logger


MODALITY_WEIGHTS = {"text": 1.0, "table": 0.9, "image": 0.7}


class ContextScorer:
    """Scores each document chunk based on relevance, recency, and source trust."""

    def __init__(self, trusted_sources: list[str] | None = None):
        self.trusted_sources = set(trusted_sources or [])

    def score(self, doc: dict, query_embedding: list[float]) -> float:
        relevance = doc.get("hybrid_score", doc.get("rerank_score", 0.5))
        modality = MODALITY_WEIGHTS.get(doc.get("type", "text"), 0.8)
        source_boost = 1.1 if doc.get("source", "") in self.trusted_sources else 1.0
        return relevance * modality * source_boost


class ContextFusion:
    """
    Merges ranked documents from multiple modalities into a final context block.
    Enforces token budget and diversity constraints.
    """

    def __init__(
        self,
        max_tokens: int = 3000,
        max_images: int = 3,
        scorer: ContextScorer | None = None,
    ):
        self.max_tokens = max_tokens
        self.max_images = max_images
        self.scorer = scorer or ContextScorer()

    def _estimate_tokens(self, text: str) -> int:
        return len(text) // 4  # rough estimate

    def fuse(self, documents: list[dict], query_embedding: list[float]) -> dict:
        # Score and sort
        for doc in documents:
            doc["context_score"] = self.scorer.score(doc, query_embedding)
        documents.sort(key=lambda d: d["context_score"], reverse=True)

        selected_text = []
        selected_images = []
        total_tokens = 0
        image_count = 0

        for doc in documents:
            dtype = doc.get("type", "text")

            if dtype == "image":
                if image_count < self.max_images:
                    selected_images.append(doc)
                    image_count += 1
            else:
                tokens = self._estimate_tokens(doc.get("content", ""))
                if total_tokens + tokens <= self.max_tokens:
                    selected_text.append(doc)
                    total_tokens += tokens

        logger.info(
            f"Context fused: {len(selected_text)} text chunks "
            f"({total_tokens} tokens), {len(selected_images)} images"
        )

        return {
            "text_chunks": selected_text,
            "images": selected_images,
            "total_tokens": total_tokens,
            "doc_count": len(selected_text) + len(selected_images),
        }

    def build_prompt_context(self, fused: dict) -> str:
        """Render fused context into a prompt-ready string."""
        parts = []
        for i, chunk in enumerate(fused["text_chunks"], 1):
            source = chunk.get("source", "unknown")
            parts.append(f"[Source {i}: {source}]\n{chunk['content']}")

        if fused["images"]:
            parts.append(f"\n[{len(fused['images'])} image(s) included for visual context]")

        return "\n\n---\n\n".join(parts)
