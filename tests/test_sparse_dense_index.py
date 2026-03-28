"""tests/test_sparse_dense_index.py — Unified sparse+dense index tests."""

import pytest
from retrieval.sparse_dense_index import SparseDenseIndex

DIM = 8

DOCS = [
    {"id": f"doc{i}", "content": f"Document {i} about {'fraud' if i % 2 == 0 else 'revenue'} analysis",
     "type": "text", "source": f"file{i}.pdf",
     "embedding": [float(i) / 10] * DIM}
    for i in range(1, 7)
]


def test_add_and_count():
    idx = SparseDenseIndex(dim=DIM)
    idx.add(DOCS[:3])
    assert idx.count() == 3


def test_search_returns_results():
    idx = SparseDenseIndex(dim=DIM)
    idx.add(DOCS)
    results = idx.search("fraud detection", [0.3] * DIM, top_k=3)
    assert len(results) <= 3
    assert all("hybrid_score" in r for r in results)


def test_incremental_add():
    idx = SparseDenseIndex(dim=DIM)
    idx.add(DOCS[:2])
    idx.add(DOCS[2:4])
    assert idx.count() == 4
    results = idx.search("revenue", [0.2] * DIM, top_k=4)
    assert len(results) <= 4


def test_stats():
    idx = SparseDenseIndex(dim=DIM)
    idx.add(DOCS)
    stats = idx.stats()
    assert stats["total_documents"] == len(DOCS)
    assert stats["faiss_vectors"] == len(DOCS)
    assert "text" in stats["modalities"]


def test_modality_filter():
    docs = DOCS[:3] + [
        {"id": "img1", "content": "chart.png", "type": "image",
         "source": "report.pdf", "embedding": [0.5] * DIM}
    ]
    idx = SparseDenseIndex(dim=DIM)
    idx.add(docs)
    results = idx.search("fraud", [0.1] * DIM, top_k=5, modality_filter="image")
    assert all(r.get("type") == "image" for r in results)


def test_save_and_load(tmp_path):
    idx = SparseDenseIndex(dim=DIM)
    idx.add(DOCS)
    idx.save(str(tmp_path))

    idx2 = SparseDenseIndex(dim=DIM)
    idx2.load(str(tmp_path))
    assert idx2.count() == len(DOCS)
    results = idx2.search("fraud", [0.1] * DIM, top_k=3)
    assert len(results) > 0
