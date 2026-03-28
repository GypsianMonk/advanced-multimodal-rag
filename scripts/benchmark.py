"""
scripts/benchmark.py
─────────────────────
Benchmarks pipeline latency and throughput across a set of queries.
Outputs a report with P50/P95/P99 latencies and QPS.

Usage:
    python scripts/benchmark.py --queries 50 --concurrency 4
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from utils.helpers import load_config
from utils.pipeline import RAGPipeline

BENCHMARK_QUERIES = [
    "What is hybrid retrieval?",
    "Explain the reranking process",
    "How does BM25 work?",
    "What is CLIP used for?",
    "Describe the validation layer",
    "How is context scored?",
    "What are the evaluation metrics?",
    "Explain query expansion",
    "How does memory feedback work?",
    "What is Reciprocal Rank Fusion?",
]


def run_single(pipeline: RAGPipeline, query: str) -> float:
    t0 = time.perf_counter()
    try:
        pipeline.run(query=query, top_k=5)
    except Exception as e:
        logger.warning(f"Query failed: {e}")
    return (time.perf_counter() - t0) * 1000  # ms


def benchmark(pipeline: RAGPipeline, n_queries: int, concurrency: int) -> dict:
    queries = (BENCHMARK_QUERIES * ((n_queries // len(BENCHMARK_QUERIES)) + 1))[:n_queries]
    latencies = []

    print(f"\n🚀 Benchmarking {n_queries} queries (concurrency={concurrency})...\n")

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(run_single, pipeline, q) for q in queries]
        for i, future in enumerate(as_completed(futures), 1):
            ms = future.result()
            latencies.append(ms)
            if i % 10 == 0:
                print(f"  {i}/{n_queries} completed...")

    total_time = time.perf_counter() - t_start
    latencies.sort()

    def pct(p: float) -> float:
        idx = int(len(latencies) * p / 100)
        return latencies[min(idx, len(latencies) - 1)]

    return {
        "total_queries": n_queries,
        "concurrency": concurrency,
        "total_time_s": round(total_time, 2),
        "qps": round(n_queries / total_time, 2),
        "latency_mean_ms": round(statistics.mean(latencies), 1),
        "latency_p50_ms": round(pct(50), 1),
        "latency_p95_ms": round(pct(95), 1),
        "latency_p99_ms": round(pct(99), 1),
        "latency_max_ms": round(max(latencies), 1),
    }


def print_report(report: dict) -> None:
    print("\n" + "═" * 50)
    print("  BENCHMARK REPORT")
    print("═" * 50)
    print(f"  Total queries   : {report['total_queries']}")
    print(f"  Concurrency     : {report['concurrency']}")
    print(f"  Total time      : {report['total_time_s']}s")
    print(f"  Throughput      : {report['qps']} QPS")
    print("─" * 50)
    print(f"  Mean latency    : {report['latency_mean_ms']} ms")
    print(f"  P50 latency     : {report['latency_p50_ms']} ms")
    print(f"  P95 latency     : {report['latency_p95_ms']} ms")
    print(f"  P99 latency     : {report['latency_p99_ms']} ms")
    print(f"  Max latency     : {report['latency_max_ms']} ms")
    print("═" * 50 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Pipeline Benchmark")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--queries", type=int, default=20, help="Number of queries to run")
    parser.add_argument("--concurrency", type=int, default=2, help="Concurrent workers")
    args = parser.parse_args()

    config = load_config(args.config)
    logger.disable("")
    pipeline = RAGPipeline(config)

    report = benchmark(pipeline, args.queries, args.concurrency)
    print_report(report)


if __name__ == "__main__":
    main()
