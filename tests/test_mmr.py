"""tests/test_mmr.py — Tests for MMR diversity ranking."""

import pytest
import numpy as np
from fusion.diversity_ranker import MMRRanker, cosine_sim


def make_doc(doc_id: str, emb: list[float], score: float = 0.9) -> dict:
    return {
        "id": doc_id,
        "content": f"Document {doc_id}",
        "source": f"{doc_id}.pdf",
        "type": "text",
        "embedding": emb,
        "hybrid_score": score,
    }


# Two nearly-identical docs + one very different
DOC_A = make_doc("a", [1.0, 0.0, 0.0, 0.0])
DOC_B = make_doc("b", [0.99, 0.01, 0.0, 0.0])   # near-duplicate of A
DOC_C = make_doc("c", [0.0, 0.0, 1.0, 0.0])      # different
QUERY_EMB = [1.0, 0.0, 0.0, 0.0]


def test_cosine_sim_identical():
    assert abs(cosine_sim([1, 0], [1, 0]) - 1.0) < 1e-5


def test_cosine_sim_orthogonal():
    assert abs(cosine_sim([1, 0], [0, 1])) < 1e-5


def test_mmr_selects_diverse_docs():
    ranker = MMRRanker(lambda_param=0.5)
    results = ranker.rank([DOC_A, DOC_B, DOC_C], QUERY_EMB, top_k=2)
    assert len(results) == 2
    ids = {r["id"] for r in results}
    # Should prefer A (most relevant) and C (diverse) over near-duplicate B
    assert "a" in ids
    assert "c" in ids


def test_mmr_pure_relevance():
    ranker = MMRRanker(lambda_param=1.0)  # ignores diversity
    results = ranker.rank([DOC_A, DOC_B, DOC_C], QUERY_EMB, top_k=2)
    assert len(results) == 2
    # A and B both highly relevant to query
    ids = {r["id"] for r in results}
    assert "a" in ids


def test_mmr_returns_mmr_score():
    ranker = MMRRanker(lambda_param=0.5)
    results = ranker.rank([DOC_A, DOC_B, DOC_C], QUERY_EMB, top_k=2)
    for r in results:
        assert "mmr_score" in r


def test_mmr_empty_input():
    ranker = MMRRanker()
    assert ranker.rank([], QUERY_EMB, top_k=5) == []


def test_mmr_top_k_capped():
    ranker = MMRRanker()
    docs = [make_doc(str(i), [float(i % 2), float((i + 1) % 2), 0.0, 0.0]) for i in range(10)]
    results = ranker.rank(docs, QUERY_EMB, top_k=3)
    assert len(results) == 3


def test_mmr_no_embeddings_fallback():
    ranker = MMRRanker()
    docs_no_emb = [
        {"id": "x", "content": "test", "source": "a.pdf", "type": "text", "hybrid_score": 0.9},
        {"id": "y", "content": "test2", "source": "b.pdf", "type": "text", "hybrid_score": 0.5},
    ]
    results = ranker.rank(docs_no_emb, QUERY_EMB, top_k=2)
    assert len(results) == 2
    assert results[0]["id"] == "x"  # sorted by hybrid_score
