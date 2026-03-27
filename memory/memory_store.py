"""
memory/memory_store.py
──────────────────────
Stores query history, retrieved context, and user feedback.
Supports in-memory (dev) and Redis (production) backends.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from loguru import logger


class InMemoryStore:
    def __init__(self):
        self._store: dict[str, dict] = {}

    def set(self, key: str, value: dict) -> None:
        self._store[key] = value

    def get(self, key: str) -> dict | None:
        return self._store.get(key)

    def keys(self) -> list[str]:
        return list(self._store.keys())


class RedisStore:
    def __init__(self, host: str = "localhost", port: int = 6379, ttl: int = 86400):
        import redis
        self.client = redis.Redis(host=host, port=port, decode_responses=True)
        self.ttl = ttl

    def set(self, key: str, value: dict) -> None:
        self.client.setex(key, self.ttl, json.dumps(value))

    def get(self, key: str) -> dict | None:
        raw = self.client.get(key)
        return json.loads(raw) if raw else None

    def keys(self) -> list[str]:
        return self.client.keys("*")


class MemoryManager:
    """
    High-level memory manager that stores:
    - Query logs
    - Retrieved context snapshots
    - User feedback signals
    """

    def __init__(self, backend: str = "memory", redis_config: dict | None = None):
        if backend == "redis" and redis_config:
            self.store = RedisStore(**redis_config)
        else:
            self.store = InMemoryStore()
        self.backend = backend

    def log_query(self, query: str, answer: str, context_docs: list[dict], metadata: dict | None = None) -> str:
        """Log a completed query-answer pair. Returns session ID."""
        session_id = str(uuid.uuid4())
        record = {
            "id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "answer": answer,
            "num_docs": len(context_docs),
            "sources": list({d.get("source", "") for d in context_docs}),
            "metadata": metadata or {},
        }
        self.store.set(f"session:{session_id}", record)
        logger.debug(f"Query logged → session:{session_id}")
        return session_id

    def record_feedback(self, session_id: str, rating: int, comment: str = "") -> None:
        """Record user feedback (rating 1-5) for a session."""
        record = self.store.get(f"session:{session_id}") or {}
        record["feedback"] = {"rating": rating, "comment": comment}
        self.store.set(f"session:{session_id}", record)
        logger.info(f"Feedback recorded for session {session_id}: rating={rating}")

    def get_recent_queries(self, limit: int = 20) -> list[dict]:
        all_keys = [k for k in self.store.keys() if k.startswith("session:")]
        records = [self.store.get(k) for k in all_keys if self.store.get(k)]
        records.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
        return records[:limit]
