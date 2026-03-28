"""tests/test_metadata_filter.py — Tests for metadata filtering."""

import pytest
from retrieval.metadata_filter import MetadataFilter

DOCS = [
    {"id": "1", "type": "text",  "source": "report.pdf",   "hybrid_score": 0.9},
    {"id": "2", "type": "image", "source": "charts.pdf",   "hybrid_score": 0.7},
    {"id": "3", "type": "table", "source": "data.csv",     "hybrid_score": 0.5},
    {"id": "4", "type": "text",  "source": "junk.pdf",     "hybrid_score": 0.2},
]


def test_filter_by_type():
    f = MetadataFilter().by_type("text")
    result = f.apply(DOCS)
    assert all(d["type"] == "text" for d in result)
    assert len(result) == 2


def test_filter_by_source():
    f = MetadataFilter().by_source(["report.pdf"])
    result = f.apply(DOCS)
    assert len(result) == 1
    assert result[0]["id"] == "1"


def test_exclude_source():
    f = MetadataFilter().exclude_source(["junk.pdf"])
    result = f.apply(DOCS)
    assert all(d["source"] != "junk.pdf" for d in result)


def test_min_score():
    f = MetadataFilter().min_score(0.6)
    result = f.apply(DOCS)
    assert all(d["hybrid_score"] >= 0.6 for d in result)
    assert len(result) == 2


def test_chained_filters():
    f = MetadataFilter().by_type("text").min_score(0.5)
    result = f.apply(DOCS)
    assert len(result) == 1
    assert result[0]["id"] == "1"


def test_custom_predicate():
    f = MetadataFilter().custom(lambda d: d["id"] in {"1", "3"})
    result = f.apply(DOCS)
    assert {d["id"] for d in result} == {"1", "3"}
