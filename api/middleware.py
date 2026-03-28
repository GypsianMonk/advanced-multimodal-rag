"""
api/middleware.py
─────────────────
FastAPI middleware: request logging, latency tracking, API key auth,
and rate limiting.
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware

from utils.metrics import metrics


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        start = time.perf_counter()

        logger.info(f"[{request_id}] {request.method} {request.url.path}")

        response = await call_next(request)

        elapsed_ms = (time.perf_counter() - start) * 1000
        metrics.increment("total_requests")
        metrics.increment(f"requests_{response.status_code // 100}xx")

        logger.info(
            f"[{request_id}] {response.status_code} — {elapsed_ms:.1f}ms"
        )

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"
        return response


class APIKeyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, api_keys: set[str], exempt_paths: set[str] | None = None):
        super().__init__(app)
        self.api_keys = api_keys
        self.exempt_paths = exempt_paths or {"/health", "/docs", "/openapi.json"}

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.exempt_paths:
            return await call_next(request)

        api_key = request.headers.get("X-API-Key")
        if not api_key or api_key not in self.api_keys:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
            )
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple sliding-window rate limiter (in-memory, per IP)."""

    def __init__(self, app, max_requests: int = 60, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._windows: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window_start = now - self.window_seconds

        # Purge old timestamps
        self._windows[client_ip] = [
            t for t in self._windows[client_ip] if t > window_start
        ]

        if len(self._windows[client_ip]) >= self.max_requests:
            metrics.increment("rate_limited_requests")
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please slow down."},
                headers={"Retry-After": str(self.window_seconds)},
            )

        self._windows[client_ip].append(now)
        return await call_next(request)
