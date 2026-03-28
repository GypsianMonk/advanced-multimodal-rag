"""
api/streaming.py
────────────────
Server-Sent Events (SSE) streaming endpoint for real-time
token-by-token answer delivery from the RAG pipeline.
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from loguru import logger

router = APIRouter()


class StreamQueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=50)
    modality_filter: str | None = None


async def _sse_event(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


async def stream_rag_response(
    query: str,
    top_k: int,
    modality_filter: str | None,
    pipeline,
) -> AsyncGenerator[str, None]:
    """
    Streams RAG pipeline stages as SSE events, then streams LLM tokens.
    Stages: query_understanding → retrieval → reranking → generating → done
    """
    try:
        # Stage 1: Query understanding
        yield await _sse_event({"stage": "query_understanding", "status": "running"})
        intent = pipeline.query_understanding.analyze(query)
        yield await _sse_event({
            "stage": "query_understanding",
            "status": "done",
            "intent": intent.intent,
            "modality": intent.modality,
            "expanded": intent.expanded_queries,
        })

        # Stage 2: Retrieval
        yield await _sse_event({"stage": "retrieval", "status": "running"})
        query_embedding = pipeline.text_embedder.embed(intent.rewritten_query)
        effective_modality = modality_filter or (intent.modality if intent.intent == "visual" else None)
        candidates = pipeline.hybrid.search(
            query=intent.rewritten_query,
            query_embedding=query_embedding,
            top_k=top_k * 3,
            modality_filter=effective_modality,
        )
        yield await _sse_event({
            "stage": "retrieval",
            "status": "done",
            "candidates": len(candidates),
        })

        # Stage 3: Reranking
        yield await _sse_event({"stage": "reranking", "status": "running"})
        reranked = pipeline.reranker.rerank(query, candidates, top_k=top_k)
        fused = pipeline.fusion.fuse(reranked, query_embedding)
        context_str = pipeline.fusion.build_prompt_context(fused)
        sources = list({d.get("source", "") for d in reranked})
        yield await _sse_event({
            "stage": "reranking",
            "status": "done",
            "final_docs": len(reranked),
            "sources": sources,
        })

        # Stage 4: Streaming generation (Anthropic streaming)
        yield await _sse_event({"stage": "generating", "status": "running"})

        import anthropic
        client = anthropic.Anthropic()

        from llm.generator import RAG_TEMPLATE, SYSTEM_PROMPT
        prompt = RAG_TEMPLATE.format(
            system=SYSTEM_PROMPT, context=context_str, query=query
        )

        full_answer = ""
        with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text_chunk in stream.text_stream:
                full_answer += text_chunk
                yield await _sse_event({"stage": "token", "token": text_chunk})

        # Stage 5: Validation
        from validation.answer_validator import AnswerValidator
        validator = AnswerValidator(min_confidence=0.5)
        val_result = validator.validate(query, full_answer, reranked, context_str)

        # Log to memory
        session_id = pipeline.memory.log_query(
            query=query,
            answer=full_answer,
            context_docs=reranked,
            metadata={"intent": intent.intent, "streamed": True},
        )

        yield await _sse_event({
            "stage": "done",
            "session_id": session_id,
            "confidence": val_result.confidence,
            "is_valid": val_result.is_valid,
            "sources": sources,
        })

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield await _sse_event({"stage": "error", "message": str(e)})


@router.post("/query/stream")
async def stream_query(req: StreamQueryRequest):
    """
    Streaming RAG endpoint via Server-Sent Events.
    Each event has a 'stage' field: query_understanding | retrieval |
    reranking | generating | token | done | error
    """
    from api.app import pipeline  # import at call time to avoid circular

    return StreamingResponse(
        stream_rag_response(
            query=req.query,
            top_k=req.top_k,
            modality_filter=req.modality_filter,
            pipeline=pipeline,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
