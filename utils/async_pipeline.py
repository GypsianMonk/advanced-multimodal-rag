"""
utils/async_pipeline.py
───────────────────────
Async version of the RAG pipeline using asyncio for concurrent
BM25 + vector retrieval, and non-blocking LLM calls.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from loguru import logger


_executor = ThreadPoolExecutor(max_workers=8)


async def run_in_thread(fn, *args, **kwargs):
    """Run a blocking function in a thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, lambda: fn(*args, **kwargs))


class AsyncRAGPipeline:
    """
    Async wrapper around the synchronous RAGPipeline.
    Runs BM25 and vector retrieval concurrently for lower latency.
    """

    def __init__(self, config: dict):
        from utils.pipeline import RAGPipeline
        self._sync = RAGPipeline(config)

    async def retrieve_concurrent(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int,
        modality_filter: str | None,
    ) -> list[dict]:
        """Run BM25 and vector search concurrently, then fuse."""
        broad_k = top_k * 3

        bm25_task = run_in_thread(
            self._sync.bm25.search, query, broad_k
        )
        vector_task = run_in_thread(
            self._sync.vector.search, query_embedding, broad_k
        )

        bm25_results, vector_results = await asyncio.gather(bm25_task, vector_task)

        if modality_filter:
            bm25_results = [r for r in bm25_results if r.get("type") == modality_filter]
            vector_results = [r for r in vector_results if r.get("type") == modality_filter]

        # Fuse via RRF
        from retrieval.hybrid import _rrf_score
        scores: dict[str, float] = {}
        doc_map: dict[str, dict] = {}
        w_bm25, w_vec = self._sync.hybrid.bm25_weight, self._sync.hybrid.vector_weight

        for rank, doc in enumerate(bm25_results):
            did = doc["id"]
            scores[did] = scores.get(did, 0.0) + w_bm25 * _rrf_score(rank)
            doc_map[did] = doc

        for rank, doc in enumerate(vector_results):
            did = doc["id"]
            scores[did] = scores.get(did, 0.0) + w_vec * _rrf_score(rank)
            doc_map[did] = doc

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{**doc_map[did], "hybrid_score": score} for did, score in ranked]

    async def run(
        self,
        query: str,
        top_k: int = 10,
        modality_filter: str | None = None,
    ) -> dict:
        # 1. Query understanding (fast, rule-based — run in thread for safety)
        intent = await run_in_thread(self._sync.query_understanding.analyze, query)
        effective_modality = modality_filter or (
            intent.modality if intent.intent == "visual" else None
        )

        # 2. Embed query
        query_embedding = await run_in_thread(
            self._sync.text_embedder.embed, intent.rewritten_query
        )

        # 3. Concurrent retrieval
        candidates = await self.retrieve_concurrent(
            query=intent.rewritten_query,
            query_embedding=query_embedding,
            top_k=top_k,
            modality_filter=effective_modality,
        )

        # 4. Rerank + fuse
        reranked = await run_in_thread(
            self._sync.reranker.rerank, query, candidates, top_k
        )
        fused = await run_in_thread(
            self._sync.fusion.fuse, reranked, query_embedding
        )
        context_str = self._sync.fusion.build_prompt_context(fused)

        # 5. Generate + validate
        result = await run_in_thread(
            self._sync.validated_pipeline.run, query, reranked, context_str
        )

        # 6. Memory
        session_id = self._sync.memory.log_query(
            query=query,
            answer=result["answer"],
            context_docs=reranked,
            metadata={"intent": intent.intent, "async": True},
        )

        logger.info(f"Async pipeline complete | session={session_id}")
        return {
            **result,
            "session_id": session_id,
            "sources": list({d.get("source", "") for d in reranked}),
        }
