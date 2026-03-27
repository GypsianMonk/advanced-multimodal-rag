"""
validation/answer_validator.py
───────────────────────────────
Answer Validation Layer:
  - Hallucination detection via source grounding check
  - Self-consistency scoring
  - Confidence gating with retry logic
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from loguru import logger


@dataclass
class ValidationResult:
    is_valid: bool
    confidence: float       # 0.0 – 1.0
    grounding_score: float  # how well answer is grounded in sources
    reason: str


class AnswerValidator:
    def __init__(self, min_confidence: float = 0.6, generator=None):
        self.min_confidence = min_confidence
        self.generator = generator

    def _grounding_score(self, answer: str, context_chunks: list[dict]) -> float:
        """
        Simple lexical overlap between answer and retrieved context.
        In production, replace with an NLI model or LLM-as-judge.
        """
        answer_words = set(re.findall(r"\b\w+\b", answer.lower()))
        if not context_chunks:
            return 0.0

        overlap_scores = []
        for chunk in context_chunks:
            chunk_words = set(re.findall(r"\b\w+\b", chunk.get("content", "").lower()))
            if chunk_words:
                overlap = len(answer_words & chunk_words) / len(answer_words | chunk_words)
                overlap_scores.append(overlap)

        return max(overlap_scores) if overlap_scores else 0.0

    def _llm_confidence_check(self, query: str, answer: str, context: str) -> float:
        """Ask the LLM to rate its own answer confidence (0-10)."""
        if not self.generator:
            return 0.75  # assume moderate confidence if no LLM available

        prompt = f"""Rate how well this answer is supported by the provided context.
Return ONLY a number from 0 to 10.

Context: {context[:500]}
Question: {query}
Answer: {answer}

Score (0-10):"""
        try:
            result = self.generator.generate(query=prompt, context="", max_tokens=10)
            match = re.search(r"\d+(?:\.\d+)?", result)
            if match:
                return min(float(match.group()) / 10.0, 1.0)
        except Exception:
            pass
        return 0.7

    def validate(
        self,
        query: str,
        answer: str,
        context_chunks: list[dict],
        context_str: str = "",
    ) -> ValidationResult:
        grounding = self._grounding_score(answer, context_chunks)
        llm_conf = self._llm_confidence_check(query, answer, context_str)
        confidence = 0.5 * grounding + 0.5 * llm_conf

        refusal_phrases = ["i don't know", "not in the context", "cannot answer", "no information"]
        if any(p in answer.lower() for p in refusal_phrases):
            return ValidationResult(
                is_valid=True, confidence=1.0, grounding_score=grounding,
                reason="Model correctly refused to hallucinate"
            )

        is_valid = confidence >= self.min_confidence
        reason = "Answer is well-grounded" if is_valid else f"Low confidence ({confidence:.2f}) — may hallucinate"

        logger.info(f"Validation: confidence={confidence:.2f}, grounding={grounding:.2f}, valid={is_valid}")
        return ValidationResult(
            is_valid=is_valid,
            confidence=confidence,
            grounding_score=grounding,
            reason=reason,
        )


class ValidatedRAGPipeline:
    """Wraps generator + validator with auto-retry logic."""

    def __init__(self, generator, validator: AnswerValidator, max_retries: int = 2):
        self.generator = generator
        self.validator = validator
        self.max_retries = max_retries

    def run(self, query: str, context_chunks: list[dict], context_str: str) -> dict:
        for attempt in range(self.max_retries + 1):
            answer = self.generator.generate(query=query, context=context_str)
            result = self.validator.validate(query, answer, context_chunks, context_str)

            if result.is_valid:
                return {"answer": answer, "validation": result, "attempts": attempt + 1}

            logger.warning(f"Attempt {attempt+1} failed validation: {result.reason}")
            query = f"Please answer carefully using ONLY the provided sources: {query}"

        logger.error("Max retries reached — returning last answer with warning")
        return {
            "answer": answer,
            "validation": result,
            "attempts": self.max_retries + 1,
            "warning": "Answer may not be fully grounded",
        }
