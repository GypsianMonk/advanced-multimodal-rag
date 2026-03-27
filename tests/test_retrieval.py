"""tests/test_retrieval.py — Unit tests for BM25, vector, and hybrid retrieval."""

import pytest
from retrieval.bm25 import BM25Retriever
from retrieval.vector import VectorRetriever
from retrieval.hybrid import HybridRetriever

DOCS = [
    {"id": "1", "content": "Fraud detection using machine learning models", "type": "text", "source": "doc1.pdf", "embedding": [0.1] * 384},
    {"id": "2", "content": "Anomaly detection in financial transactions", "type": "text", "source": "doc2.pdf", "embedding": [0.2] * 384},
    {"id": "3", "content": "Revenue trends and quarterly analysis", "type": "text", "source": "doc3.pdf", "embedding": [0.3] * 384},
]


def test_bm25_build_and_search():
    bm25 = BM25Retriever()
    bm25.build(DOCS)
    results = bm25.search("fraud detection", top_k=2)
    assert len(results) > 0
    assert "bm25_score" in results[0]
    assert results[0]["bm25_score"] > 0


def test_bm25_no_results_for_unrelated():
    bm25 = BM25Retriever()
    bm25.build(DOCS)
    results = bm25.search("zzzzunknownterm", top_k=5)
    assert all(r["bm25_score"] == 0 for r in results) or len(results) == 0


def test_vector_retriever():
    vr = VectorRetriever(dim=384)
    vr.add(DOCS)
    results = vr.search([0.15] * 384, top_k=2)
    assert len(results) == 2
    assert "vector_score" in results[0]


def test_hybrid_retriever():
    bm25 = BM25Retriever()
    bm25.build(DOCS)
    vr = VectorRetriever(dim=384)
    vr.add(DOCS)
    hybrid = HybridRetriever(bm25, vr)
    results = hybrid.search("fraud", [0.1] * 384, top_k=3)
    assert len(results) <= 3
    assert "hybrid_score" in results[0]
