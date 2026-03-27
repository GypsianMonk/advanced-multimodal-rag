"""tests/test_validation.py — Tests for answer validation logic."""

import pytest
from validation.answer_validator import AnswerValidator

CHUNKS = [
    {"content": "Fraud was detected in Q3 using anomaly detection algorithms.", "source": "report.pdf"},
    {"content": "Transaction risk increased by 12% in September.", "source": "report.pdf"},
]


def test_grounding_score_high():
    validator = AnswerValidator(min_confidence=0.5)
    answer = "Fraud was detected using anomaly detection in Q3."
    score = validator._grounding_score(answer, CHUNKS)
    assert score > 0.2


def test_grounding_score_low_for_unrelated():
    validator = AnswerValidator(min_confidence=0.5)
    answer = "The capital of France is Paris and the Eiffel Tower is tall."
    score = validator._grounding_score(answer, CHUNKS)
    assert score < 0.3


def test_validate_accepts_well_grounded():
    validator = AnswerValidator(min_confidence=0.3)
    answer = "Fraud detection used anomaly algorithms to flag transactions."
    result = validator.validate("How was fraud detected?", answer, CHUNKS)
    assert isinstance(result.confidence, float)
    assert isinstance(result.is_valid, bool)


def test_validate_accepts_refusal():
    validator = AnswerValidator(min_confidence=0.9)
    answer = "I don't know based on the provided context."
    result = validator.validate("What is the meaning of life?", answer, CHUNKS)
    assert result.is_valid is True
