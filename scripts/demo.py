"""
scripts/demo.py
───────────────
Quick demonstration of the Multi-Modal RAG pipeline.
Ingests sample documents, runs queries, and prints results.

Usage:
    python scripts/demo.py
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from loguru import logger
from utils.logger import setup_logger

setup_logger(log_level="INFO")


def create_sample_docs():
    """Create sample text files for demonstration."""
    import tempfile, pathlib

    tmp = pathlib.Path(tempfile.mkdtemp())
    (tmp / "fraud_report.txt").write_text(
        "Fraud detection analysis Q3 2024.\n"
        "Anomaly detection models flagged 1,247 suspicious transactions.\n"
        "Transaction risk increased by 12% compared to Q2.\n"
        "Machine learning models achieved 94.3% precision.\n"
        "False positive rate was reduced to 2.1% from 5.8%.\n"
    )
    (tmp / "revenue_report.txt").write_text(
        "Revenue report Q3 2024.\n"
        "Total revenue: $48.2M, up 18% year-over-year.\n"
        "Enterprise segment grew 32% driven by new contracts.\n"
        "Customer retention rate: 91.4%.\n"
        "Churn rate decreased from 9.2% to 8.6%.\n"
    )
    return str(tmp)


def main():
    from retrieval.bm25 import BM25Retriever
    from retrieval.vector import VectorRetriever
    from retrieval.hybrid import HybridRetriever
    from embeddings.text_embedder import TextEmbedder
    from fusion.context_fusion import ContextFusion
    from utils.chunker import RecursiveChunker
    from validation.answer_validator import AnswerValidator
    from llm.query_understanding import QueryUnderstanding

    logger.info("=== Multi-Modal RAG Demo ===")

    # Setup
    embedder = TextEmbedder("all-MiniLM-L6-v2")
    chunker = RecursiveChunker(chunk_size=256, overlap=32)
    source_dir = create_sample_docs()

    # Ingest sample documents
    logger.info(f"Ingesting documents from {source_dir}")
    docs = []
    import pathlib
    for path in pathlib.Path(source_dir).glob("*.txt"):
        text = path.read_text()
        for chunk in chunker.chunk(text):
            import hashlib
            docs.append({
                "id": hashlib.md5(chunk.encode()).hexdigest()[:8],
                "content": chunk,
                "source": path.name,
                "type": "text",
                "embedding": embedder.embed(chunk),
            })

    logger.info(f"Ingested {len(docs)} chunks")

    # Build indexes
    bm25 = BM25Retriever()
    bm25.build(docs)

    vector = VectorRetriever(dim=embedder.dim)
    vector.add(docs)

    hybrid = HybridRetriever(bm25, vector)
    qu = QueryUnderstanding()
    fusion = ContextFusion(max_tokens=800)
    validator = AnswerValidator(min_confidence=0.0)  # no LLM in demo

    queries = [
        "What was the fraud detection precision rate?",
        "How did revenue change year-over-year?",
        "What is the customer churn rate?",
    ]

    print("\n" + "="*60)
    for query in queries:
        intent = qu.analyze(query)
        q_emb = embedder.embed(intent.rewritten_query)
        results = hybrid.search(query, q_emb, top_k=4)
        fused = fusion.fuse(results, q_emb)
        context = fusion.build_prompt_context(fused)

        print(f"\n🔍 Query: {query}")
        print(f"   Intent: {intent.intent} | Modality: {intent.modality}")
        print(f"   Retrieved {len(results)} chunks from: {set(d['source'] for d in results)}")
        print(f"   Context preview: {context[:200]}...")
        print()

    logger.success("Demo complete! Connect an LLM in config/config.yaml to generate answers.")


if __name__ == "__main__":
    main()
