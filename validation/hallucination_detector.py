"""
validation/hallucination_detector.py
─────────────────────────────────────
NLI-based hallucination detection using a Natural Language Inference model
to check whether each sentence in the answer is entailed by the context.

For each answer sentence:
  - ENTAILMENT   → supported (good)
  - NEUTRAL      → cannot verify (warn)
  - CONTRADICTION → hallucination (bad)

Falls back to lexical overlap if no NLI model is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from loguru import logger


@dataclass
class SentenceVerdict:
    sentence: str
    label: str          # ENTAILMENT | NEUTRAL | CONTRADICTION
    confidence: float
    is_hallucination: bool


@dataclass
class HallucinationReport:
    overall_score: float        # 0.0 (hallucinated) – 1.0 (fully grounded)
    sentence_verdicts: list[SentenceVerdict]
    hallucinated_sentences: list[str]
    entailed_ratio: float
    has_contradictions: bool


class NLIHallucinationDetector:
    """
    Uses a cross-encoder NLI model to verify each answer sentence
    against the retrieved context.

    Model: facebook/bart-large-mnli (zero-shot) or
           cross-encoder/nli-deberta-v3-small (faster)
    """

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small"):
        try:
            from transformers import pipeline as hf_pipeline
            logger.info(f"Loading NLI model: {model_name}")
            self._classifier = hf_pipeline(
                "zero-shot-classification",
                model=model_name,
                device=-1,   # CPU
            )
            self._available = True
            logger.success("NLI hallucination detector ready")
        except Exception as e:
            logger.warning(f"NLI model unavailable ({e}), using lexical fallback")
            self._classifier = None
            self._available = False

    def _split_sentences(self, text: str) -> list[str]:
        import re
        sents = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sents if len(s.strip()) > 10]

    def _lexical_verdict(self, sentence: str, context: str) -> SentenceVerdict:
        """Fallback: simple word overlap heuristic."""
        import re
        s_words = set(re.findall(r'\b\w+\b', sentence.lower()))
        c_words = set(re.findall(r'\b\w+\b', context.lower()))
        overlap = len(s_words & c_words) / (len(s_words) + 1e-9)

        if overlap > 0.4:
            label, conf = "ENTAILMENT", overlap
        elif overlap > 0.15:
            label, conf = "NEUTRAL", overlap
        else:
            label, conf = "CONTRADICTION", 1 - overlap

        return SentenceVerdict(
            sentence=sentence,
            label=label,
            confidence=round(conf, 3),
            is_hallucination=(label == "CONTRADICTION"),
        )

    def _nli_verdict(self, sentence: str, context: str) -> SentenceVerdict:
        result = self._classifier(
            sequences=context[:512],
            candidate_labels=[sentence],
            hypothesis_template="{}",
        )
        # Map NLI output → verdict
        score = result["scores"][0]
        if score > 0.7:
            label = "ENTAILMENT"
        elif score > 0.4:
            label = "NEUTRAL"
        else:
            label = "CONTRADICTION"

        return SentenceVerdict(
            sentence=sentence,
            label=label,
            confidence=round(score, 3),
            is_hallucination=(label == "CONTRADICTION"),
        )

    def detect(self, answer: str, context: str) -> HallucinationReport:
        sentences = self._split_sentences(answer)
        if not sentences:
            return HallucinationReport(
                overall_score=1.0,
                sentence_verdicts=[],
                hallucinated_sentences=[],
                entailed_ratio=1.0,
                has_contradictions=False,
            )

        verdicts: list[SentenceVerdict] = []
        for sent in sentences:
            if self._available and self._classifier:
                try:
                    v = self._nli_verdict(sent, context)
                except Exception:
                    v = self._lexical_verdict(sent, context)
            else:
                v = self._lexical_verdict(sent, context)
            verdicts.append(v)

        entailed = [v for v in verdicts if v.label == "ENTAILMENT"]
        contradictions = [v for v in verdicts if v.is_hallucination]

        entailed_ratio = len(entailed) / len(verdicts)
        overall_score = entailed_ratio - (0.3 * len(contradictions) / len(verdicts))
        overall_score = max(0.0, min(1.0, overall_score))

        report = HallucinationReport(
            overall_score=round(overall_score, 3),
            sentence_verdicts=verdicts,
            hallucinated_sentences=[v.sentence for v in contradictions],
            entailed_ratio=round(entailed_ratio, 3),
            has_contradictions=len(contradictions) > 0,
        )

        if report.has_contradictions:
            logger.warning(
                f"Hallucination detected: {len(contradictions)} contradictory sentences "
                f"(overall_score={overall_score:.2f})"
            )

        return report
