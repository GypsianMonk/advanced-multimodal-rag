"""tests/test_fusion.py — Tests for context fusion and scoring."""

import pytest
from fusion.context_fusion import ContextScorer, ContextFusion

DOCS = [
    {"id": "1", "content": "Fraud detection report Q3 2024. Risk increased by 12%.", "type": "text", "source": "report.pdf", "hybrid_score": 0.9},
    {"id": "2", "content": "Machine learning models outperformed rule-based systems.", "type": "text", "source": "research.pdf", "hybrid_score": 0.7},
    {"id": "3", "content": "/data/chart.png", "type": "image", "source": "visuals.pdf", "hybrid_score": 0.6},
    {"id": "4", "content": "Low quality chunk with barely any useful content here.", "type": "text", "source": "junk.pdf", "hybrid_score": 0.1},
]

QUERY_EMB = [0.1] * 384


def test_context_scorer_trusted_source():
    scorer = ContextScorer(trusted_sources=["report.pdf"])
    score_trusted = scorer.score(DOCS[0], QUERY_EMB)
    score_untrusted = scorer.score(DOCS[1], QUERY_EMB)
    assert score_trusted > score_untrusted


def test_fusion_respects_token_budget():
    fusion = ContextFusion(max_tokens=50)  # Very tight budget
    fused = fusion.fuse(DOCS, QUERY_EMB)
    assert fused["total_tokens"] <= 50


def test_fusion_limits_images():
    fusion = ContextFusion(max_images=1)
    fused = fusion.fuse(DOCS, QUERY_EMB)
    assert len(fused["images"]) <= 1


def test_fusion_orders_by_score():
    fusion = ContextFusion(max_tokens=9999)
    fused = fusion.fuse(DOCS, QUERY_EMB)
    if len(fused["text_chunks"]) > 1:
        scores = [c["context_score"] for c in fused["text_chunks"]]
        assert scores == sorted(scores, reverse=True)


def test_build_prompt_context():
    fusion = ContextFusion()
    fused = fusion.fuse(DOCS, QUERY_EMB)
    prompt = fusion.build_prompt_context(fused)
    assert "Source" in prompt
    assert len(prompt) > 10
