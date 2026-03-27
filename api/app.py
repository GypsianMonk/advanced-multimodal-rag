"""
api/app.py
──────────
FastAPI REST interface for the Multi-Modal RAG system.
Exposes /query, /ingest, /feedback, and /health endpoints.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from loguru import logger

from utils.helpers import load_config
from utils.pipeline import RAGPipeline

app = FastAPI(
    title="Advanced Multi-Modal RAG API",
    description="Production-grade RAG with hybrid retrieval, reranking, and answer validation",
    version="1.0.0",
)

config = load_config("config/config.yaml")
pipeline = RAGPipeline(config)


# ─── Request / Response Models ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000, example="What are the fraud trends for Q3?")
    top_k: int = Field(default=10, ge=1, le=50)
    modality_filter: str | None = Field(default=None, description="text | image | table | None")

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float
    session_id: str
    attempts: int

class FeedbackRequest(BaseModel):
    session_id: str
    rating: int = Field(..., ge=1, le=5)
    comment: str = ""

class IngestRequest(BaseModel):
    source_path: str
    file_type: str | None = None


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    try:
        result = pipeline.run(
            query=req.query,
            top_k=req.top_k,
            modality_filter=req.modality_filter,
        )
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            confidence=result["validation"].confidence,
            session_id=result["session_id"],
            attempts=result["attempts"],
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    try:
        pipeline.memory.record_feedback(req.session_id, req.rating, req.comment)
        return {"status": "recorded"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/ingest")
async def ingest(req: IngestRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(pipeline.ingest, req.source_path)
    return {"status": "ingestion_started", "source": req.source_path}


@app.get("/history")
async def history(limit: int = 20):
    return {"queries": pipeline.memory.get_recent_queries(limit=limit)}
