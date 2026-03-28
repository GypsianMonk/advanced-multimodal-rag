"""
retrieval/metadata_filter.py
─────────────────────────────
Pre-retrieval and post-retrieval metadata filtering.
Supports date ranges, source whitelists, document types, and custom predicates.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable


class MetadataFilter:
    """
    Chainable filter for document metadata.

    Usage:
        f = MetadataFilter().by_type("text").by_source(["report.pdf"]).min_score(0.4)
        filtered = f.apply(documents)
    """

    def __init__(self):
        self._predicates: list[Callable[[dict], bool]] = []

    def by_type(self, modality: str) -> "MetadataFilter":
        self._predicates.append(lambda d: d.get("type") == modality)
        return self

    def by_source(self, sources: list[str]) -> "MetadataFilter":
        source_set = set(sources)
        self._predicates.append(lambda d: d.get("source", "") in source_set)
        return self

    def exclude_source(self, sources: list[str]) -> "MetadataFilter":
        source_set = set(sources)
        self._predicates.append(lambda d: d.get("source", "") not in source_set)
        return self

    def min_score(self, threshold: float, score_key: str = "hybrid_score") -> "MetadataFilter":
        self._predicates.append(lambda d: d.get(score_key, 0.0) >= threshold)
        return self

    def by_date_range(self, start: datetime, end: datetime) -> "MetadataFilter":
        def _check(d: dict) -> bool:
            raw = d.get("date")
            if not raw:
                return True  # pass-through if no date metadata
            try:
                dt = datetime.fromisoformat(raw)
                return start <= dt <= end
            except ValueError:
                return True
        self._predicates.append(_check)
        return self

    def custom(self, fn: Callable[[dict], bool]) -> "MetadataFilter":
        self._predicates.append(fn)
        return self

    def apply(self, documents: list[dict]) -> list[dict]:
        result = documents
        for pred in self._predicates:
            result = [d for d in result if pred(d)]
        return result

    def reset(self) -> "MetadataFilter":
        self._predicates.clear()
        return self
