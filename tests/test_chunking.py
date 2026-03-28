"""tests/test_chunking.py — Tests for all chunking strategies."""

import pytest
from ingestion.chunking_strategies import (
    SentenceAwareChunker,
    MarkdownChunker,
    SemanticChunker,
    get_chunker,
)
from utils.chunker import RecursiveChunker

LONG_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Machine learning models can detect anomalies in data. "
    "Fraud detection systems use neural networks. "
    "Revenue increased by 18% year-over-year. "
    "Customer retention reached 91.4% this quarter. "
    "The company plans to expand to three new markets. "
    "Risk models were retrained using Q3 data. "
) * 5

MARKDOWN_TEXT = """# Introduction
This section covers the basics.

## Fraud Detection
Anomaly detection is used to find fraud.
Models achieve 94% precision.

### Sub-section
More details here.

## Revenue Analysis
Revenue grew 18% YoY.
Enterprise segment up 32%.
"""


def test_sentence_aware_basic():
    chunker = SentenceAwareChunker(max_tokens=100)
    chunks = chunker.chunk(LONG_TEXT)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) > 0


def test_sentence_aware_short_text():
    chunker = SentenceAwareChunker(max_tokens=500)
    short = "Hello world. This is a test."
    chunks = chunker.chunk(short)
    assert len(chunks) == 1
    assert "Hello" in chunks[0]


def test_sentence_overlap():
    chunker = SentenceAwareChunker(max_tokens=50, overlap_sentences=1)
    chunks = chunker.chunk(LONG_TEXT)
    assert len(chunks) > 2


def test_markdown_chunker():
    chunker = MarkdownChunker()
    chunks = chunker.chunk(MARKDOWN_TEXT)
    assert len(chunks) >= 3
    assert any("Fraud" in c for c in chunks)
    assert any("Revenue" in c for c in chunks)


def test_markdown_no_headings():
    chunker = MarkdownChunker()
    plain = "Just some plain text with no headings at all."
    chunks = chunker.chunk(plain)
    assert len(chunks) == 1


def test_semantic_chunker_fallback():
    # Without embedder, falls back to SentenceAware
    chunker = SemanticChunker(embedder=None, max_tokens=100)
    chunks = chunker.chunk(LONG_TEXT)
    assert len(chunks) > 1


def test_get_chunker_factory():
    for strategy in ["sentence", "markdown", "recursive"]:
        chunker = get_chunker(strategy, max_tokens=300)
        assert chunker is not None
        chunks = chunker.chunk(LONG_TEXT)
        assert len(chunks) > 0


def test_get_chunker_invalid():
    with pytest.raises(ValueError):
        get_chunker("nonexistent_strategy")


def test_recursive_chunker():
    chunker = RecursiveChunker(chunk_size=100, overlap=20)
    chunks = chunker.chunk(LONG_TEXT)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 120  # chunk_size + some tolerance
