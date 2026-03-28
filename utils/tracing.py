"""
utils/tracing.py
────────────────
Lightweight observability for the RAG pipeline.
Traces each stage (retrieval, reranking, generation, validation)
with latency, token counts, and quality scores.

Outputs to:
  - Loguru structured logs
  - Optional OpenTelemetry (if otel SDK installed)
  - In-memory span store for the /traces debug endpoint
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
from loguru import logger


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    start_time: float = field(default_factory=time.perf_counter)
    end_time: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "running"

    @property
    def duration_ms(self) -> float:
        end = self.end_time or time.perf_counter()
        return (end - self.start_time) * 1000

    def finish(self, status: str = "ok", **attrs) -> None:
        self.end_time = time.perf_counter()
        self.status = status
        self.attributes.update(attrs)

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "name": self.name,
            "duration_ms": round(self.duration_ms, 2),
            "status": self.status,
            **self.attributes,
        }


class Tracer:
    """
    Collects spans for a single query trace.
    Usage:
        tracer = Tracer()
        with tracer.span("retrieval") as span:
            results = retrieve(...)
            span.finish(docs=len(results))
    """

    def __init__(self):
        self.trace_id = str(uuid.uuid4())[:12]
        self.spans: list[Span] = []
        self._start = time.perf_counter()

    @contextmanager
    def span(self, name: str, **initial_attrs):
        s = Span(name=name, trace_id=self.trace_id, attributes=initial_attrs)
        self.spans.append(s)
        try:
            yield s
            if s.status == "running":
                s.finish(status="ok")
        except Exception as e:
            s.finish(status="error", error=str(e))
            raise

    @property
    def total_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000

    def summary(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "total_ms": round(self.total_ms, 2),
            "spans": [s.to_dict() for s in self.spans],
            "stage_breakdown": {
                s.name: round(s.duration_ms, 2) for s in self.spans
            },
        }

    def log(self) -> None:
        summary = self.summary()
        logger.info(
            f"Trace {self.trace_id} | {summary['total_ms']}ms total | "
            + " | ".join(f"{k}={v}ms" for k, v in summary["stage_breakdown"].items())
        )


# Global trace store (last N traces for /traces debug endpoint)
_trace_store: list[dict] = []
MAX_STORED_TRACES = 100


def store_trace(tracer: Tracer) -> None:
    global _trace_store
    _trace_store.append(tracer.summary())
    _trace_store = _trace_store[-MAX_STORED_TRACES:]


def get_recent_traces(limit: int = 20) -> list[dict]:
    return list(reversed(_trace_store))[:limit]
