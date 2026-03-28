"""tests/test_tracing.py — Tests for the observability tracing module."""

import pytest
import time
from utils.tracing import Tracer, store_trace, get_recent_traces


def test_span_records_duration():
    tracer = Tracer()
    with tracer.span("test_stage") as span:
        time.sleep(0.01)
    assert span.duration_ms >= 10
    assert span.status == "ok"


def test_span_catches_exception():
    tracer = Tracer()
    with pytest.raises(ValueError):
        with tracer.span("failing_stage") as span:
            raise ValueError("deliberate error")
    assert span.status == "error"
    assert "deliberate error" in span.attributes.get("error", "")


def test_tracer_summary_structure():
    tracer = Tracer()
    with tracer.span("retrieval"):
        pass
    with tracer.span("reranking"):
        pass
    summary = tracer.summary()
    assert "trace_id" in summary
    assert "total_ms" in summary
    assert len(summary["spans"]) == 2
    assert "retrieval" in summary["stage_breakdown"]
    assert "reranking" in summary["stage_breakdown"]


def test_store_and_retrieve_traces():
    tracer = Tracer()
    with tracer.span("test"):
        pass
    store_trace(tracer)
    recent = get_recent_traces(limit=5)
    assert len(recent) >= 1
    assert recent[0]["trace_id"] == tracer.trace_id


def test_total_ms_positive():
    tracer = Tracer()
    time.sleep(0.005)
    assert tracer.total_ms > 0
