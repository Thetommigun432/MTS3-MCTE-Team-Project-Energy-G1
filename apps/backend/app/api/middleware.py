"""
API middleware for rate limiting, request size, and request ID.
"""

import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from app.core.config import get_settings
from app.core.errors import ErrorCode, error_response
from app.core.logging import get_logger, request_id_ctx
from app.core.telemetry import RATE_LIMIT_HIT, REQUEST_COUNT, REQUEST_LATENCY

logger = get_logger(__name__)


# =============================================================================
# Request ID Middleware
# =============================================================================


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Middleware to inject request ID into context and response headers."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get or generate request ID
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = f"req_{uuid.uuid4().hex[:16]}"

        # Set context variable for logging
        token = request_id_ctx.set(request_id)

        try:
            # Store in request state for access in route handlers
            request.state.request_id = request_id

            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            request_id_ctx.reset(token)


# =============================================================================
# Metrics Middleware
# =============================================================================


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to record request metrics."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        response = await call_next(request)

        duration = time.time() - start_time
        route = request.url.path
        method = request.method
        status = response.status_code

        REQUEST_COUNT.labels(method=method, route=route, status=status).inc()
        REQUEST_LATENCY.labels(method=method, route=route).observe(duration)

        return response


# =============================================================================
# Rate Limiting
# =============================================================================


@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""

    capacity: int
    tokens: float = field(default=0.0)
    last_refill: float = field(default_factory=time.time)
    refill_rate: float = 1.0  # tokens per second

    def consume(self) -> bool:
        """Try to consume a token. Returns True if successful."""
        now = time.time()
        elapsed = now - self.last_refill

        # Refill tokens
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Token bucket rate limiting middleware."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self._buckets: dict[str, TokenBucket] = {}
        self._settings = get_settings()
        self._user_limit, self._user_interval = self._parse_limit(
            self._settings.rate_limit_per_user
        )
        self._ip_limit, self._ip_interval = self._parse_limit(
            self._settings.rate_limit_per_ip
        )

    def _parse_limit(self, limit_str: str) -> tuple[int, int]:
        """Parse limit string like '60/minute' into (count, seconds)."""
        parts = limit_str.split("/")
        count = int(parts[0])
        interval_map = {"second": 1, "minute": 60, "hour": 3600}
        seconds = interval_map.get(parts[1], 60)
        return count, seconds

    def _get_bucket(self, key: str, limit: int, interval: int) -> TokenBucket:
        """Get or create a token bucket for the given key."""
        if key not in self._buckets:
            self._buckets[key] = TokenBucket(
                capacity=limit,
                tokens=float(limit),
                refill_rate=limit / interval,
            )
        return self._buckets[key]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health checks
        if request.url.path in ("/live", "/ready", "/health"):
            return await call_next(request)

        # Get key and limits
        user_id = getattr(request.state, "user_id", None)
        if user_id:
            key = f"user:{user_id}"
            limit, interval = self._user_limit, self._user_interval
            key_type = "user"
        else:
            # Use client IP
            client_ip = request.client.host if request.client else "unknown"
            key = f"ip:{client_ip}"
            limit, interval = self._ip_limit, self._ip_interval
            key_type = "ip"

        bucket = self._get_bucket(key, limit, interval)

        if not bucket.consume():
            RATE_LIMIT_HIT.labels(key_type=key_type).inc()
            request_id = getattr(request.state, "request_id", None)

            from app.core.errors import RateLimitError
            error = RateLimitError(
                message="Rate limit exceeded",
                retry_after=int(interval / limit),
            )

            return JSONResponse(
                status_code=429,
                content=error_response(error, request_id),
                headers={"Retry-After": str(int(interval / limit))},
            )

        return await call_next(request)


# =============================================================================
# Request Size Limit
# =============================================================================


class RequestSizeLimitMiddleware:
    """ASGI middleware to limit request body size."""

    def __init__(self, app: ASGIApp, max_bytes: int | None = None) -> None:
        self.app = app
        settings = get_settings()
        self.max_bytes = max_bytes or settings.max_body_bytes

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        content_length = None
        for header_name, header_value in scope.get("headers", []):
            if header_name == b"content-length":
                try:
                    content_length = int(header_value.decode())
                except (ValueError, UnicodeDecodeError):
                    pass
                break

        if content_length is not None and content_length > self.max_bytes:
            response = JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "code": ErrorCode.REQUEST_TOO_LARGE.value,
                        "message": f"Request body too large. Maximum size: {self.max_bytes} bytes",
                        "details": {"max_bytes": self.max_bytes},
                    },
                    "request_id": None,
                },
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
