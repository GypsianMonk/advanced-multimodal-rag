"""
utils/pipeline.py
─────────────────
Orchestrates the full Multi-Modal RAG pipeline end-to-end.
"""

from __future__ import annotations

from loguru import logger

from embeddings.text_embedder import TextEmbedder
from embeddings.image_embedder import ImageEmbedder
from retrieval.bm25 import BM25Retriever
from retrieval.vector import VectorRetriever
from retrieval.hybrid import HybridRetriever
from ranking.reranker import MultiModalReranker
from fusion.context_fusion import ContextFusion, ContextScorer
from llm.generator import LLMGenerator
from llm.query_understanding import QueryUnderstanding
from validation.answer_validator import AnswerValidator, ValidatedRAGPipeline
from memory.memory_store import MemoryManager
from ingestion.ingest import Ingestor


class RAGPipeline:
    def __init__(self, config: dict):
        logger.info("Initialising RAG pipeline...")
        self.cfg = config

        self.text_embedder = TextEmbedder(config["models"]["text_embedding"])
        self.image_embedder = ImageEmbedder(config["models"]["image_embedding"])

        self.bm25 = BM25Retriever()
        self.vector = VectorRetriever(dim=self.text_embedder.dim)
        self.hybrid = HybridRetriever(self.bm25, self.vector)

        self.reranker = MultiModalReranker(
            text_model=config["models"]["reranker"],
            image_embedder=self.image_embedder,
        )

        self.fusion = ContextFusion(
            max_tokens=config.get("max_context_tokens", 3000),
            scorer=ContextScorer(config.get("trusted_sources", [])),
        )

        self.generator = LLMGenerator(
            provider=config["llm"]["provider"],
            model=config["llm"]["model"],
        )

        self.query_understanding = QueryUnderstanding(generator=self.generator)

        validator = AnswerValidator(
            min_confidence=config.get("min_confidence", 0.6),
            generator=self.generator,
        )
        self.validated_pipeline = ValidatedRAGPipeline(
            generator=self.generator,
            validator=validator,
            max_retries=config.get("max_retries", 2),
        )

        self.memory = MemoryManager(
            backend=config.get("memory_backend", "memory"),
            redis_config=config.get("redis"),
        )

        self.ingestor = Ingestor(config)
        logger.success("RAG pipeline ready")

    def run(self, query: str, top_k: int = 10, modality_filter: str | None = None) -> dict:
        # 1. Query Understanding
        intent = self.query_understanding.analyze(query)
        effective_modality = modality_filter or (intent.modality if intent.intent == "visual" else None)

        # 2. Embed query
        query_embedding = self.text_embedder.embed(intent.rewritten_query)

        # 3. Hybrid retrieval
        candidates = self.hybrid.search(
            query=intent.rewritten_query,
            query_embedding=query_embedding,
            top_k=top_k * 3,
            modality_filter=effective_modality,
        )

        # 4. Rerank
        reranked = self.reranker.rerank(query, candidates, top_k=top_k)

        # 5. Context fusion
        fused = self.fusion.fuse(reranked, query_embedding)
        context_str = self.fusion.build_prompt_context(fused)

        # 6. Generate + validate
        result = self.validated_pipeline.run(
            query=query,
            context_chunks=reranked,
            context_str=context_str,
        )

        # 7. Memory
        session_id = self.memory.log_query(
            query=query,
            answer=result["answer"],
            context_docs=reranked,
            metadata={"intent": intent.intent, "modality": intent.modality},
        )

        return {
            **result,
            "session_id": session_id,
            "sources": list({d.get("source", "") for d in reranked}),
        }

    def ingest(self, source_path: str) -> None:
        docs = self.ingestor.ingest_directory(source_path)
        self.bm25.build(docs)
        self.vector.add(docs)
        logger.success(f"Ingested and indexed {len(docs)} documents from {source_path}")
