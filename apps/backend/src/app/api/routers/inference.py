"""
Inference API endpoints.
"""

import time
from typing import Any

from fastapi import APIRouter, Header, Request

from app.api.deps import CurrentUserDep, RequestIdDep
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.telemetry import IDEMPOTENCY_CACHE_HIT, IDEMPOTENCY_CACHE_SIZE
from app.domain.inference import get_inference_service
from app.infra.redis import get_redis_cache
from app.schemas.inference import InferRequest, InferResponse, ModelsListResponse, ModelMetricsResponse

logger = get_logger(__name__)
router = APIRouter(tags=["Inference"])


# =============================================================================
# Idempotency Cache
# =============================================================================


def _idempotency_key(user_id: str, key: str) -> str:
    """Generate cache key for idempotency."""
    return f"idempotency:{user_id}:{key}"


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
    Uses Redis cache with fallback to in-memory.
    """
    settings = get_settings()
    cache = get_redis_cache()

    # Check idempotency cache
    if idempotency_key:
        cache_key = _idempotency_key(current_user.user_id, idempotency_key)
        cached = await cache.get(cache_key)
        if cached:
            IDEMPOTENCY_CACHE_HIT.inc()
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
        cache_key = _idempotency_key(current_user.user_id, idempotency_key)
        await cache.set(
            cache_key,
            response.model_dump(),
            ttl=settings.idempotency_cache_ttl_seconds,
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


@router.get("/models/{model_id}/metrics", response_model=ModelMetricsResponse)
async def get_model_metrics(model_id: str) -> ModelMetricsResponse:
    """
    Get detailed metrics and configuration for a specific model.

    Returns architecture params, preprocessing config, and thresholds.
    """
    service = get_inference_service()
    entry = await service.get_model_details(model_id)
    
    return entry
