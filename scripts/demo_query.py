"""
scripts/demo_query.py
──────────────────────
Interactive CLI demo for the Multi-Modal RAG system.
Runs without a running server — uses the pipeline directly.

Usage:
    python scripts/demo_query.py
    python scripts/demo_query.py --query "What are the fraud trends?"
    python scripts/demo_query.py --ingest ./data/raw --query "Summarise the key findings"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from utils.helpers import load_config
from utils.pipeline import RAGPipeline


BANNER = """
╔══════════════════════════════════════════════════════╗
║        Advanced Multi-Modal RAG — Demo CLI           ║
║  Hybrid retrieval · Reranking · Answer Validation    ║
╚══════════════════════════════════════════════════════╝
"""

SAMPLE_QUERIES = [
    "What are the main fraud detection techniques?",
    "Show me a comparison of retrieval methods",
    "What is the role of the re-ranking engine?",
    "Explain the answer validation pipeline",
    "What are the evaluation metrics used?",
]


def print_result(result: dict, elapsed: float) -> None:
    print("\n" + "─" * 60)
    print(f"✅  Answer  ({elapsed:.2f}s | confidence={result['validation'].confidence:.2f})")
    print("─" * 60)
    print(result["answer"])
    print("\n📚 Sources:", ", ".join(result["sources"]) or "none")
    print(f"🔄 Attempts: {result['attempts']}  |  Session: {result['session_id'][:8]}...")
    print("─" * 60 + "\n")


def interactive_loop(pipeline: RAGPipeline) -> None:
    print("\n💡 Sample queries (press Enter to skip):")
    for i, q in enumerate(SAMPLE_QUERIES, 1):
        print(f"  {i}. {q}")

    print("\nType your query (or 'quit' to exit):\n")
    while True:
        try:
            query = input("🔍 Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Allow selecting sample queries by number
        if query.isdigit() and 1 <= int(query) <= len(SAMPLE_QUERIES):
            query = SAMPLE_QUERIES[int(query) - 1]
            print(f"  → {query}")

        t0 = time.perf_counter()
        try:
            result = pipeline.run(query=query, top_k=10)
            elapsed = time.perf_counter() - t0
            print_result(result, elapsed)
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Modal RAG Demo CLI")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--query", help="Run a single query and exit")
    parser.add_argument("--ingest", help="Ingest documents before querying")
    args = parser.parse_args()

    print(BANNER)

    config = load_config(args.config)
    logger.disable("")  # Suppress debug logs in demo mode

    print("⚙️  Initialising pipeline...")
    pipeline = RAGPipeline(config)

    if args.ingest:
        print(f"📥 Ingesting from {args.ingest} ...")
        pipeline.ingest(args.ingest)
        print("✅ Ingestion complete\n")

    if args.query:
        t0 = time.perf_counter()
        result = pipeline.run(query=args.query, top_k=10)
        print_result(result, time.perf_counter() - t0)
    else:
        interactive_loop(pipeline)


if __name__ == "__main__":
    main()
