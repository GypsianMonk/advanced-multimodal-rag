"""
llm/chain_of_thought.py
───────────────────────
Chain-of-Thought (CoT) prompting wrapper.
Forces the LLM to reason step-by-step before producing a final answer,
significantly reducing hallucinations on complex queries.
"""

from __future__ import annotations

import re
from loguru import logger


COT_SYSTEM = """You are a precise, knowledge-grounded research assistant.
When answering questions, follow this exact format:

STEP 1 — UNDERSTAND: Restate the question in your own words.
STEP 2 — EVIDENCE: List the most relevant facts from the context (quote sources).
STEP 3 — GAPS: Note any information missing from the context.
STEP 4 — REASONING: Connect evidence to produce a logical answer.
FINAL ANSWER: State the answer clearly and concisely.

If the context does not contain sufficient information, say so in STEP 3 and
base your FINAL ANSWER only on what is available."""

COT_TEMPLATE = """{system}

=== CONTEXT ===
{context}
===============

Question: {query}

Work through the steps:"""


SELF_CORRECTION_PROMPT = """Review the answer below and check for:
1. Claims not supported by the context (hallucinations)
2. Logical inconsistencies
3. Missing key information from the context

If corrections are needed, provide a revised FINAL ANSWER.
If the answer is accurate, respond with "VERIFIED: " followed by the answer.

Original Answer: {answer}

Context (for verification): {context}

Your review:"""


class ChainOfThoughtGenerator:
    """Wraps an LLMGenerator to use step-by-step CoT reasoning."""

    def __init__(self, generator):
        self.generator = generator

    def generate(
        self,
        query: str,
        context: str,
        max_tokens: int = 1500,
    ) -> dict:
        """
        Generate an answer using Chain-of-Thought reasoning.
        Returns dict with 'reasoning' and 'final_answer' keys.
        """
        prompt_context = COT_TEMPLATE.format(
            system=COT_SYSTEM, context=context, query=query
        )

        raw = self.generator.generate(
            query=prompt_context,
            context="",
            max_tokens=max_tokens,
        )

        final_answer = self._extract_final_answer(raw)
        return {
            "reasoning": raw,
            "final_answer": final_answer or raw,
            "used_cot": True,
        }

    def generate_with_self_correction(
        self,
        query: str,
        context: str,
        max_tokens: int = 1500,
    ) -> dict:
        """Two-pass: generate then self-correct."""
        result = self.generate(query, context, max_tokens)

        # Self-correction pass
        correction_query = SELF_CORRECTION_PROMPT.format(
            answer=result["final_answer"],
            context=context[:1500],
        )
        corrected = self.generator.generate(
            query=correction_query,
            context="",
            max_tokens=512,
        )

        if corrected.startswith("VERIFIED:"):
            final = corrected.replace("VERIFIED:", "").strip()
            was_corrected = False
        else:
            final = self._extract_final_answer(corrected) or corrected
            was_corrected = True
            logger.info("Self-correction modified the answer")

        return {
            "reasoning": result["reasoning"],
            "final_answer": final,
            "was_corrected": was_corrected,
            "used_cot": True,
        }

    @staticmethod
    def _extract_final_answer(text: str) -> str | None:
        match = re.search(r"FINAL ANSWER[:\s]+(.*?)(?:\n\n|\Z)", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None
