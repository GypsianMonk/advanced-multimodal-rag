"""tests/test_routing.py — Tests for query routing logic."""

import pytest
from retrieval.router import QueryRouter
from llm.query_understanding import QueryIntent


def make_intent(intent: str, modality: str) -> QueryIntent:
    return QueryIntent(
        intent=intent,
        modality=modality,
        expanded_queries=["test"],
        rewritten_query="test query",
    )


router = QueryRouter()


def test_visual_routes_to_image():
    decision = router.route(make_intent("visual", "image"))
    assert decision.strategy == "image_only"
    assert decision.modality_filter == "image"


def test_analytical_routes_to_broad_hybrid():
    decision = router.route(make_intent("analytical", "text"))
    assert decision.strategy == "hybrid"
    assert decision.modality_filter is None
    assert decision.top_k_multiplier >= 1.5


def test_comparative_routes_to_table():
    decision = router.route(make_intent("comparative", "table"))
    assert decision.strategy == "hybrid"
    assert decision.modality_filter == "table"


def test_factual_routes_to_standard_hybrid():
    decision = router.route(make_intent("factual", "text"))
    assert decision.strategy == "hybrid"
    assert decision.modality_filter == "text"
    assert decision.top_k_multiplier == 1.0


def test_route_has_reason():
    decision = router.route(make_intent("factual", "text"))
    assert len(decision.reason) > 0
