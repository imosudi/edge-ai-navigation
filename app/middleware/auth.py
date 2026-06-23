"""
app/middleware/auth.py
Simple API-key authentication middleware.

The key is passed via the X-API-Key header or ?api_key= query parameter.
Set via environment variable EDGE_AI_API_KEY or config file.
WebSocket and static file routes are exempt.
"""

from __future__ import annotations

import hmac
import logging
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# Routes that do NOT require authentication
_EXEMPT_PREFIXES = ("/static/", "/api/docs", "/api/redoc", "/openapi.json")
_EXEMPT_EXACT    = ("/",)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Validates X-API-Key header (or api_key query param) against the
    configured key using a constant-time comparison to prevent timing attacks.
    """

    def __init__(self, app, api_key: str) -> None:
        super().__init__(app)
        # Store the expected key as bytes for constant-time comparison
        self._key_bytes = api_key.encode()

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        path = request.url.path

        # Exempt public paths
        if path in _EXEMPT_EXACT or any(path.startswith(p) for p in _EXEMPT_PREFIXES):
            return await call_next(request)

        # Extract key from header or query parameter
        provided_key = (
            request.headers.get("X-API-Key")
            or request.query_params.get("api_key")
            or ""
        )

        if not self._valid_key(provided_key):
            logger.warning("Unauthorised request from %s to %s", request.client, path)
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid or missing API key."},
                headers={"WWW-Authenticate": "ApiKey"},
            )

        return await call_next(request)

    def _valid_key(self, provided: str) -> bool:
        """Constant-time comparison to prevent timing side-channel attacks."""
        return hmac.compare_digest(
            self._key_bytes,
            provided.encode(),
        )
