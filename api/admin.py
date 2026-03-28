"""
api/admin.py
────────────
Admin API endpoints for index management, system diagnostics,
and trace inspection. Mount under /admin with auth middleware.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from loguru import logger
import os

router = APIRouter(prefix="/admin", tags=["admin"])
security = HTTPBearer()

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "change-me-in-production")


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid admin token")
    return credentials.credentials


# ─── Models ──────────────────────────────────────────────────────────────────

class RebuildRequest(BaseModel):
    source_path: str
    confirm: bool = False


class PurgeRequest(BaseModel):
    confirm: bool = False


# ─── Endpoints ───────────────────────────────────────────────────────────────

@router.get("/health/detailed")
def detailed_health(token: str = Depends(verify_token)):
    """Full system health including index stats and memory."""
    from api.app import pipeline
    from utils.tracing import get_recent_traces

    index_stats = {}
    try:
        index_stats = {
            "bm25_docs": len(pipeline.bm25.documents),
            "faiss_vectors": pipeline.vector.index.ntotal,
        }
    except Exception as e:
        index_stats = {"error": str(e)}

    memory_stats = {}
    try:
        recent = pipeline.memory.get_recent_queries(limit=5)
        memory_stats = {
            "recent_queries": len(recent),
            "backend": pipeline.memory.backend,
        }
    except Exception as e:
        memory_stats = {"error": str(e)}

    return {
        "status": "ok",
        "index": index_stats,
        "memory": memory_stats,
        "recent_traces": len(get_recent_traces()),
    }


@router.get("/traces")
def list_traces(limit: int = 20, token: str = Depends(verify_token)):
    """Return recent pipeline execution traces."""
    from utils.tracing import get_recent_traces
    return {"traces": get_recent_traces(limit=limit)}


@router.get("/conversations")
def list_conversations(token: str = Depends(verify_token)):
    """List all active conversations."""
    try:
        from api.app import pipeline
        if hasattr(pipeline, "conversation_memory"):
            return {"conversations": pipeline.conversation_memory.list_conversations()}
        return {"conversations": [], "note": "conversation memory not enabled"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/index/rebuild")
def rebuild_index(req: RebuildRequest, token: str = Depends(verify_token)):
    """Full index rebuild from a source directory."""
    if not req.confirm:
        raise HTTPException(
            status_code=400,
            detail="Set confirm=true to proceed with full index rebuild"
        )
    try:
        from api.app import pipeline
        pipeline.ingest(req.source_path)
        return {"status": "rebuilt", "source": req.source_path}
    except Exception as e:
        logger.error(f"Rebuild failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/index/purge")
def purge_index(req: PurgeRequest, token: str = Depends(verify_token)):
    """Clear all indexed documents. Irreversible."""
    if not req.confirm:
        raise HTTPException(
            status_code=400,
            detail="Set confirm=true to purge all indexed data"
        )
    try:
        from api.app import pipeline
        import faiss
        pipeline.vector.index = faiss.IndexFlatIP(pipeline.vector.dim)
        pipeline.vector.documents = []
        pipeline.bm25.index = None
        pipeline.bm25.documents = []
        logger.warning("Index purged by admin request")
        return {"status": "purged"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
def system_stats(token: str = Depends(verify_token)):
    """System-level stats: query counts, avg latency, hit rates."""
    from utils.tracing import get_recent_traces
    traces = get_recent_traces(limit=100)
    if not traces:
        return {"message": "No traces available yet"}

    latencies = [t["total_ms"] for t in traces if "total_ms" in t]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    return {
        "total_queries_traced": len(traces),
        "avg_latency_ms": round(avg_latency, 2),
        "p50_ms": round(sorted(latencies)[len(latencies) // 2], 2) if latencies else 0,
        "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2) if latencies else 0,
    }
