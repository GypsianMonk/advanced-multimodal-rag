"""
utils/cache.py
──────────────
Query result cache to avoid redundant LLM calls for repeated queries.
Uses an LRU in-memory cache with optional Redis backend.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import Any

from loguru import logger


def _query_hash(query: str, top_k: int, modality: str | None) -> str:
    key = f"{query.strip().lower()}|{top_k}|{modality}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


class InMemoryCache:
    def __init__(self, max_size: int = 256):
        self._cache: dict[str, Any] = {}
        self._order: list[str] = []
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        if key in self._cache:
            self.hits += 1
            logger.debug(f"Cache HIT  [{key}]")
            return self._cache[key]
        self.misses += 1
        return None

    def set(self, key: str, value: Any) -> None:
        if key in self._cache:
            return
        if len(self._order) >= self.max_size:
            oldest = self._order.pop(0)
            del self._cache[oldest]
        self._cache[key] = value
        self._order.append(key)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
            "size": len(self._cache),
        }

    def clear(self) -> None:
        self._cache.clear()
        self._order.clear()
        self.hits = 0
        self.misses = 0


class RedisCache:
    def __init__(self, host: str = "localhost", port: int = 6379, ttl: int = 3600):
        import redis
        self.client = redis.Redis(host=host, port=port, decode_responses=True)
        self.ttl = ttl

    def get(self, key: str) -> Any | None:
        raw = self.client.get(f"rag:cache:{key}")
        if raw:
            logger.debug(f"Redis cache HIT [{key}]")
            return json.loads(raw)
        return None

    def set(self, key: str, value: Any) -> None:
        self.client.setex(f"rag:cache:{key}", self.ttl, json.dumps(value, default=str))

    def clear(self) -> None:
        keys = self.client.keys("rag:cache:*")
        if keys:
            self.client.delete(*keys)


class QueryCache:
    """High-level cache wrapper used by the RAG pipeline."""

    def __init__(self, backend: str = "memory", **kwargs):
        self.backend = InMemoryCache(**kwargs) if backend == "memory" else RedisCache(**kwargs)

    def lookup(self, query: str, top_k: int = 10, modality: str | None = None) -> Any | None:
        key = _query_hash(query, top_k, modality)
        return self.backend.get(key)

    def store(self, query: str, result: Any, top_k: int = 10, modality: str | None = None) -> None:
        key = _query_hash(query, top_k, modality)
        self.backend.set(key, result)
