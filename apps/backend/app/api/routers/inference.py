"""
Inference API endpoints.
"""

import hashlib
import time
from typing import Any

from fastapi import APIRouter, Header, Request

from app.api.deps import CurrentUserDep, RequestIdDep
from app.core.logging import get_logger
from app.core.telemetry import IDEMPOTENCY_CACHE_HIT, IDEMPOTENCY_CACHE_SIZE
from app.domain.inference import get_inference_service
from app.schemas.inference import InferRequest, InferResponse, ModelsListResponse

logger = get_logger(__name__)
router = APIRouter(tags=["Inference"])


# =============================================================================
# Idempotency Cache
# =============================================================================


class IdempotencyCache:
    """Simple in-memory idempotency cache."""

    def __init__(self, ttl_seconds: int = 600) -> None:
        self._cache: dict[str, tuple[float, dict]] = {}
        self._ttl = ttl_seconds

    def _key(self, user_id: str, idempotency_key: str) -> str:
        """Generate cache key."""
        return f"{user_id}:{idempotency_key}"

    def get(self, user_id: str, idempotency_key: str) -> dict | None:
        """Get cached response if exists and not expired."""
        key = self._key(user_id, idempotency_key)
        entry = self._cache.get(key)

        if entry is None:
            return None

        timestamp, response = entry
        if time.time() - timestamp > self._ttl:
            del self._cache[key]
            return None

        IDEMPOTENCY_CACHE_HIT.inc()
        return response

    def set(self, user_id: str, idempotency_key: str, response: dict) -> None:
        """Cache a response."""
        key = self._key(user_id, idempotency_key)
        self._cache[key] = (time.time(), response)
        IDEMPOTENCY_CACHE_SIZE.set(len(self._cache))

    def cleanup(self) -> int:
        """Remove expired entries. Returns count removed."""
        now = time.time()
        expired = [k for k, (ts, _) in self._cache.items() if now - ts > self._ttl]
        for k in expired:
            del self._cache[k]
        IDEMPOTENCY_CACHE_SIZE.set(len(self._cache))
        return len(expired)


_idempotency_cache = IdempotencyCache()


# =============================================================================
# Endpoints
# =============================================================================


@router.post("/infer", response_model=InferResponse)
async def infer(
    request: InferRequest,
    current_user: CurrentUserDep,
    request_id: RequestIdDep,
    idempotency_key: str | None = Header(alias="Idempotency-Key", default=None),
) -> InferResponse:
    """
    Run inference and persist prediction to InfluxDB.

    Implements Predict & Persist Strategy A:
    - Returns 200 only if prediction is persisted
    - Returns 503 if InfluxDB write fails

    Supports idempotency via Idempotency-Key header.
    """
    # Check idempotency cache
    if idempotency_key:
        cached = _idempotency_cache.get(current_user.user_id, idempotency_key)
        if cached:
            logger.debug(
                "Returning cached idempotent response",
                extra={"idempotency_key": idempotency_key},
            )
            return InferResponse(**cached)

    # Run inference
    service = get_inference_service()
    response = await service.infer_and_persist(
        request=request,
        token=current_user,
        request_id=request_id,
    )

    # Cache for idempotency
    if idempotency_key:
        _idempotency_cache.set(
            current_user.user_id,
            idempotency_key,
            response.model_dump(),
        )

    return response


@router.get("/models", response_model=ModelsListResponse)
async def list_models() -> ModelsListResponse:
    """
    List available models.

    Returns all models in the registry with their status.
    """
    service = get_inference_service()
    models = await service.list_models()

    return ModelsListResponse(
        models=models,
        count=len(models),
    )
