"""
app/middleware/rate_limit.py
Sliding-window rate limiter middleware (per IP address).

Limits REST API calls; WebSocket connections are excluded.
Uses an in-memory store - suitable for single-process deployment.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding-window rate limiter.

    Args:
        max_requests: Maximum allowed requests per minute per IP.
    """

    def __init__(self, app, max_requests: int = 120) -> None:
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = 60.0
        # ip → deque of timestamps
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Skip WebSocket upgrade requests
        if request.headers.get("upgrade", "").lower() == "websocket":
            return await call_next(request)

        client_ip = self._get_client_ip(request)
        now = time.monotonic()

        async with self._lock:
            timestamps = self._requests[client_ip]

            # Evict timestamps outside the sliding window
            cutoff = now - self.window_seconds
            while timestamps and timestamps[0] < cutoff:
                timestamps.popleft()

            if len(timestamps) >= self.max_requests:
                retry_after = int(self.window_seconds - (now - timestamps[0]))
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate limit exceeded.",
                        "retry_after_seconds": retry_after,
                    },
                    headers={"Retry-After": str(retry_after)},
                )

            timestamps.append(now)

        return await call_next(request)

    @staticmethod
    def _get_client_ip(request: Request) -> str:
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        if request.client:
            return request.client.host
        return "unknown"
