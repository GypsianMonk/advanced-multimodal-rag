"""
utils/evaluate.py
─────────────────
Evaluation framework for retrieval and generation quality.
Metrics: Precision@K, Recall@K, MRR, Faithfulness, Hallucination Rate
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger


@dataclass
class RetrievalMetrics:
    precision_at_k: float
    recall_at_k: float
    mrr: float


@dataclass
class GenerationMetrics:
    faithfulness: float
    hallucination_rate: float
    avg_confidence: float


@dataclass
class EvalReport:
    retrieval: RetrievalMetrics
    generation: GenerationMetrics
    latency_p50_ms: float
    latency_p95_ms: float
    total_queries: int


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / k if k else 0.0


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(relevant_ids) if relevant_ids else 0.0


def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


class Evaluator:
    def __init__(self, pipeline, eval_data_path: str):
        self.pipeline = pipeline
        self.eval_data = self._load(eval_data_path)

    def _load(self, path: str) -> list[dict]:
        with open(path) as f:
            return json.load(f)

    def run(self, k: int = 10) -> EvalReport:
        import time
        prec_scores, rec_scores, mrr_scores = [], [], []
        faith_scores, hall_rates, confidences = [], [], []
        latencies = []

        for sample in self.eval_data:
            query = sample["query"]
            relevant_ids = set(sample.get("relevant_doc_ids", []))

            t0 = time.perf_counter()
            result = self.pipeline.run(query=query, top_k=k)
            latencies.append((time.perf_counter() - t0) * 1000)

            retrieved_ids = [d.get("id", "") for d in result.get("reranked_docs", [])]
            if relevant_ids:
                prec_scores.append(precision_at_k(retrieved_ids, relevant_ids, k))
                rec_scores.append(recall_at_k(retrieved_ids, relevant_ids, k))
                mrr_scores.append(mean_reciprocal_rank(retrieved_ids, relevant_ids))

            confidences.append(result["validation"].confidence)
            hall_rates.append(0.0 if result["validation"].is_valid else 1.0)

        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]

        report = EvalReport(
            retrieval=RetrievalMetrics(
                precision_at_k=sum(prec_scores) / max(len(prec_scores), 1),
                recall_at_k=sum(rec_scores) / max(len(rec_scores), 1),
                mrr=sum(mrr_scores) / max(len(mrr_scores), 1),
            ),
            generation=GenerationMetrics(
                faithfulness=sum(confidences) / len(confidences),
                hallucination_rate=sum(hall_rates) / len(hall_rates),
                avg_confidence=sum(confidences) / len(confidences),
            ),
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            total_queries=len(self.eval_data),
        )

        logger.info(f"Eval complete: P@{k}={report.retrieval.precision_at_k:.3f}, MRR={report.retrieval.mrr:.3f}")
        return report
