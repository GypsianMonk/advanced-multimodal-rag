"""
llm/query_understanding.py
──────────────────────────
Query Understanding Layer:
  - Intent classification (factual / visual / analytical / comparative)
  - Modality detection (text / image / table)
  - Query rewriting and expansion
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from loguru import logger


@dataclass
class QueryIntent:
    intent: str          # factual | visual | analytical | comparative
    modality: str        # text | image | table | mixed
    expanded_queries: list[str]
    rewritten_query: str


VISUAL_KEYWORDS = {"graph", "chart", "plot", "diagram", "image", "figure", "show", "visualize", "trend"}
TABLE_KEYWORDS = {"table", "comparison", "compare", "list", "breakdown", "statistics", "data"}
ANALYTICAL_KEYWORDS = {"why", "how", "analyze", "explain", "reason", "cause", "impact"}


def classify_intent(query: str) -> str:
    q = query.lower()
    if any(kw in q for kw in VISUAL_KEYWORDS):
        return "visual"
    if any(kw in q for kw in ANALYTICAL_KEYWORDS):
        return "analytical"
    if any(kw in q for kw in TABLE_KEYWORDS):
        return "comparative"
    return "factual"


def detect_modality(query: str, intent: str) -> str:
    if intent == "visual":
        return "image"
    q = query.lower()
    if any(kw in q for kw in TABLE_KEYWORDS):
        return "table"
    return "text"


class QueryUnderstanding:
    """Uses an LLM to expand and rewrite queries for better retrieval."""

    def __init__(self, generator=None):
        self.generator = generator

    def expand_synonyms(self, query: str) -> list[str]:
        """Rule-based synonym expansion (fast, no LLM needed)."""
        expansions = [query]
        replacements = {
            "fraud": ["anomaly", "suspicious transaction", "risk"],
            "detect": ["identify", "find", "flag"],
            "revenue": ["sales", "income", "earnings"],
            "customer": ["client", "user", "buyer"],
        }
        for word, synonyms in replacements.items():
            if word in query.lower():
                for syn in synonyms:
                    expansions.append(query.lower().replace(word, syn))
        return list(set(expansions))[:4]

    def analyze(self, query: str) -> QueryIntent:
        intent = classify_intent(query)
        modality = detect_modality(query, intent)
        expanded = self.expand_synonyms(query)

        # LLM-based rewrite if available
        rewritten = query
        if self.generator:
            try:
                rewritten = self.generator.generate(
                    query=query,
                    context="Rewrite this search query to be more specific and retrieval-friendly. Return only the rewritten query.",
                    max_tokens=64,
                )
            except Exception:
                pass

        logger.info(f"Query intent={intent}, modality={modality}, expansions={len(expanded)}")
        return QueryIntent(
            intent=intent,
            modality=modality,
            expanded_queries=expanded,
            rewritten_query=rewritten,
        )
