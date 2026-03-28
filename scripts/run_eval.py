"""
scripts/run_eval.py
───────────────────
Runs the full evaluation suite against the RAG pipeline and
prints a formatted report with retrieval + generation metrics.

Usage:
    python scripts/run_eval.py --config config/config.yaml \
                               --eval-data data/eval/eval_set.json \
                               --top-k 10
"""

from __future__ import annotations

import argparse
import json
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from loguru import logger


def print_report(report) -> None:
    sep = "─" * 52
    print(f"\n{'='*52}")
    print(f"  📊  RAG EVALUATION REPORT")
    print(f"{'='*52}")
    print(f"\n  Queries evaluated : {report.total_queries}")
    print(f"\n  {sep}")
    print(f"  RETRIEVAL METRICS")
    print(f"  {sep}")
    print(f"  Precision@K       : {report.retrieval.precision_at_k:.4f}")
    print(f"  Recall@K          : {report.retrieval.recall_at_k:.4f}")
    print(f"  MRR               : {report.retrieval.mrr:.4f}")
    print(f"\n  {sep}")
    print(f"  GENERATION METRICS")
    print(f"  {sep}")
    print(f"  Faithfulness      : {report.generation.faithfulness:.4f}")
    print(f"  Avg Confidence    : {report.generation.avg_confidence:.4f}")
    print(f"  Hallucination Rate: {report.generation.hallucination_rate:.4f}")
    print(f"\n  {sep}")
    print(f"  LATENCY")
    print(f"  {sep}")
    print(f"  P50               : {report.latency_p50_ms:.1f} ms")
    print(f"  P95               : {report.latency_p95_ms:.1f} ms")
    print(f"{'='*52}\n")


def run_simple_eval(eval_data: list[dict], top_k: int) -> None:
    """
    Lightweight eval without a full pipeline —
    runs BM25 + vector retrieval only (no LLM needed).
    """
    from retrieval.bm25 import BM25Retriever
    from retrieval.vector import VectorRetriever
    from retrieval.hybrid import HybridRetriever
    from embeddings.text_embedder import TextEmbedder
    from utils.chunker import RecursiveChunker

    embedder = TextEmbedder("all-MiniLM-L6-v2")
    chunker = RecursiveChunker(chunk_size=256, overlap=32)

    # Build index from eval docs
    all_docs = []
    import hashlib
    for item in eval_data:
        for ctx in item.get("context_docs", []):
            for chunk in chunker.chunk(ctx):
                did = hashlib.md5(chunk.encode()).hexdigest()[:8]
                all_docs.append({
                    "id": did,
                    "content": chunk,
                    "source": "eval",
                    "type": "text",
                    "embedding": embedder.embed(chunk),
                })

    if not all_docs:
        logger.warning("No context_docs in eval set — skipping retrieval eval")
        return

    bm25 = BM25Retriever()
    bm25.build(all_docs)
    vector = VectorRetriever(dim=embedder.dim)
    vector.add(all_docs)
    hybrid = HybridRetriever(bm25, vector)

    precision_scores, recall_scores, mrr_scores = [], [], []
    latencies = []

    for item in eval_data:
        query = item["query"]
        relevant_ids = set(item.get("relevant_doc_ids", []))
        if not relevant_ids:
            continue

        t0 = time.perf_counter()
        q_emb = embedder.embed(query)
        results = hybrid.search(query, q_emb, top_k=top_k)
        latencies.append((time.perf_counter() - t0) * 1000)

        retrieved_ids = [r["id"] for r in results]

        # Precision@K
        hits = sum(1 for rid in retrieved_ids[:top_k] if rid in relevant_ids)
        precision_scores.append(hits / top_k)

        # Recall@K
        recall_scores.append(hits / len(relevant_ids))

        # MRR
        for rank, rid in enumerate(retrieved_ids, 1):
            if rid in relevant_ids:
                mrr_scores.append(1.0 / rank)
                break
        else:
            mrr_scores.append(0.0)

    if precision_scores:
        print(f"\n{'='*52}")
        print(f"  📊  RETRIEVAL EVAL (no LLM)")
        print(f"{'='*52}")
        print(f"  Queries        : {len(precision_scores)}")
        print(f"  Precision@{top_k:<3}  : {sum(precision_scores)/len(precision_scores):.4f}")
        print(f"  Recall@{top_k:<3}     : {sum(recall_scores)/len(recall_scores):.4f}")
        print(f"  MRR            : {sum(mrr_scores)/len(mrr_scores):.4f}")
        lats = sorted(latencies)
        print(f"  P50 latency    : {lats[len(lats)//2]:.1f} ms")
        print(f"  P95 latency    : {lats[int(len(lats)*0.95)]:.1f} ms")
        print(f"{'='*52}\n")


def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation suite")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--eval-data", default="data/eval/eval_set.json")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--no-llm", action="store_true",
                        help="Run retrieval-only eval (no LLM required)")
    args = parser.parse_args()

    with open(args.eval_data) as f:
        eval_data = json.load(f)

    logger.info(f"Loaded {len(eval_data)} eval samples from {args.eval_data}")

    if args.no_llm:
        run_simple_eval(eval_data, args.top_k)
    else:
        from utils.helpers import load_config
        from utils.evaluate import Evaluator
        from utils.pipeline import RAGPipeline

        config = load_config(args.config)
        pipeline = RAGPipeline(config)
        evaluator = Evaluator(pipeline, args.eval_data)
        report = evaluator.run(k=args.top_k)
        print_report(report)


if __name__ == "__main__":
    main()
