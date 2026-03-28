"""
utils/metrics.py
────────────────
Runtime metrics collection: latency tracking, throughput counters,
and a Prometheus-compatible metrics exporter.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class LatencyStats:
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0
    _recent: deque = field(default_factory=lambda: deque(maxlen=1000))

    def record(self, ms: float) -> None:
        self.count += 1
        self.total_ms += ms
        self.min_ms = min(self.min_ms, ms)
        self.max_ms = max(self.max_ms, ms)
        self._recent.append(ms)

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count else 0.0

    def percentile(self, p: float) -> float:
        if not self._recent:
            return 0.0
        sorted_vals = sorted(self._recent)
        idx = int(len(sorted_vals) * p / 100)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]


class MetricsCollector:
    """Thread-safe metrics collector for the RAG pipeline."""

    def __init__(self):
        self._lock = Lock()
        self._latencies: dict[str, LatencyStats] = defaultdict(LatencyStats)
        self._counters: dict[str, int] = defaultdict(int)

    @contextmanager
    def measure(self, operation: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            with self._lock:
                self._latencies[operation].record(elapsed_ms)

    def increment(self, counter: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[counter] += amount

    def summary(self) -> dict:
        with self._lock:
            result = {"counters": dict(self._counters), "latencies": {}}
            for op, stats in self._latencies.items():
                result["latencies"][op] = {
                    "count": stats.count,
                    "avg_ms": round(stats.avg_ms, 2),
                    "min_ms": round(stats.min_ms, 2),
                    "max_ms": round(stats.max_ms, 2),
                    "p50_ms": round(stats.percentile(50), 2),
                    "p95_ms": round(stats.percentile(95), 2),
                    "p99_ms": round(stats.percentile(99), 2),
                }
        return result

    def reset(self) -> None:
        with self._lock:
            self._latencies.clear()
            self._counters.clear()


# Global singleton for convenience
metrics = MetricsCollector()
