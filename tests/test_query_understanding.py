"""tests/test_query_understanding.py — Tests for query intent & expansion."""

import pytest
from llm.query_understanding import QueryUnderstanding, classify_intent, detect_modality


def test_classify_visual_intent():
    assert classify_intent("Show me a graph of fraud trends") == "visual"
    assert classify_intent("Plot the revenue over time") == "visual"


def test_classify_analytical_intent():
    assert classify_intent("Why did fraud increase in Q3?") == "analytical"
    assert classify_intent("How does the model detect anomalies?") == "analytical"


def test_classify_factual_intent():
    assert classify_intent("What is the fraud rate?") == "factual"


def test_detect_modality_image():
    assert detect_modality("Show me a chart", "visual") == "image"


def test_detect_modality_table():
    assert detect_modality("Compare the fraud statistics", "comparative") == "table"


def test_detect_modality_text():
    assert detect_modality("Tell me about fraud", "factual") == "text"


def test_synonym_expansion():
    qu = QueryUnderstanding()
    expanded = qu.expand_synonyms("fraud detection in transactions")
    assert len(expanded) >= 1
    assert any("fraud" in q or "anomaly" in q for q in expanded)


def test_analyze_returns_intent():
    qu = QueryUnderstanding()
    result = qu.analyze("Why did revenue drop last quarter?")
    assert result.intent in ("factual", "analytical", "comparative", "visual")
    assert result.modality in ("text", "image", "table", "mixed")
    assert isinstance(result.expanded_queries, list)
    assert isinstance(result.rewritten_query, str)
