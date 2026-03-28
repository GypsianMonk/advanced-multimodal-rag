"""
retrieval/router.py
───────────────────
Query Router: directs queries to the optimal retrieval strategy
based on detected intent and modality from the query understanding layer.

Routes:
  - visual   → image-only CLIP search
  - analytical → hybrid (broad context needed)
  - comparative → table + hybrid with metadata filter
  - factual  → hybrid (default)
"""

from __future__ import annotations

from dataclasses import dataclass
from loguru import logger

from llm.query_understanding import QueryIntent


@dataclass
class RouteDecision:
    strategy: str           # hybrid | vector_only | bm25_only | image_only
    modality_filter: str | None
    top_k_multiplier: float  # boost broad retrieval for complex queries
    reason: str


class QueryRouter:
    def route(self, intent: QueryIntent) -> RouteDecision:
        if intent.intent == "visual" or intent.modality == "image":
            return RouteDecision(
                strategy="image_only",
                modality_filter="image",
                top_k_multiplier=1.0,
                reason="Visual query → CLIP image search",
            )

        if intent.intent == "analytical":
            return RouteDecision(
                strategy="hybrid",
                modality_filter=None,       # retrieve all modalities
                top_k_multiplier=2.0,       # need broader context
                reason="Analytical query → broad hybrid retrieval",
            )

        if intent.intent == "comparative" or intent.modality == "table":
            return RouteDecision(
                strategy="hybrid",
                modality_filter="table",
                top_k_multiplier=1.5,
                reason="Comparative query → table + hybrid",
            )

        # Default: factual
        return RouteDecision(
            strategy="hybrid",
            modality_filter="text",
            top_k_multiplier=1.0,
            reason="Factual query → standard hybrid retrieval",
        )

    def log_route(self, query: str, decision: RouteDecision) -> None:
        logger.info(
            f"Routing '{query[:60]}' → "
            f"strategy={decision.strategy}, "
            f"filter={decision.modality_filter}, "
            f"k_mult={decision.top_k_multiplier} | {decision.reason}"
        )
