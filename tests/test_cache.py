"""tests/test_cache.py — Tests for query result caching."""

import pytest
from utils.cache import InMemoryCache, QueryCache


def test_in_memory_cache_set_get():
    cache = InMemoryCache(max_size=5)
    cache.set("key1", {"answer": "hello"})
    assert cache.get("key1") == {"answer": "hello"}


def test_in_memory_cache_miss():
    cache = InMemoryCache()
    assert cache.get("nonexistent") is None


def test_in_memory_cache_lru_eviction():
    cache = InMemoryCache(max_size=3)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    cache.set("d", 4)  # should evict "a"
    assert cache.get("a") is None
    assert cache.get("d") == 4


def test_cache_stats():
    cache = InMemoryCache()
    cache.set("x", 42)
    cache.get("x")   # hit
    cache.get("y")   # miss
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.5


def test_query_cache_roundtrip():
    qc = QueryCache(backend="memory")
    qc.store("What is fraud?", {"answer": "Fraud is deception."}, top_k=5)
    result = qc.lookup("What is fraud?", top_k=5)
    assert result is not None
    assert result["answer"] == "Fraud is deception."


def test_query_cache_different_params():
    qc = QueryCache(backend="memory")
    qc.store("What is fraud?", {"answer": "A"}, top_k=5)
    # Different top_k → different cache key
    result = qc.lookup("What is fraud?", top_k=10)
    assert result is None
