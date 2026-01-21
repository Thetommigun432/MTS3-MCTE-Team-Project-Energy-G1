"""
Inference service orchestrating the full inference workflow.
Implements Predict & Persist (Strategy A).
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

from starlette.concurrency import run_in_threadpool

from app.core.errors import InfluxError
from app.core.logging import get_logger, request_id_ctx
from app.core.security import TokenPayload
from app.domain.authz import require_appliance_access, require_building_access
from app.domain.inference.engine import get_inference_engine
from app.domain.inference.registry import get_model_registry
from app.infra.influx import get_influx_client
from app.schemas.inference import InferRequest, InferResponse, ModelInfo

logger = get_logger(__name__)


class InferenceService:
    """Service for running inference with persistence."""

    async def infer_and_persist(
        self,
        request: InferRequest,
        token: TokenPayload,
        request_id: str,
    ) -> InferResponse:
        """
        Run inference and persist the prediction to InfluxDB.

        Implements Strategy A: only return success if write succeeds.

        Raises:
            AuthorizationError: If user doesn't have access
            ValidationError: If input is invalid
            ModelError: If model loading/inference fails
            InfluxError: If persistence fails
        """
        start_time = time.time()

        # AuthZ checks
        await require_building_access(token, request.building_id)
        await require_appliance_access(token, request.building_id, request.appliance_id)

        # Get model
        engine = get_inference_engine()
        model, entry = engine.get_model(request.model_id, request.appliance_id)

        # Run inference in thread pool (don't block event loop)
        predicted_kw, confidence = await run_in_threadpool(
            engine.run_inference,
            model,
            entry,
            request.window,
        )

        inference_latency_ms = (time.time() - start_time) * 1000

        # Parse timestamp
        timestamp: datetime | None = None
        if request.timestamp:
            try:
                timestamp = datetime.fromisoformat(request.timestamp.replace("Z", "+00:00"))
            except ValueError:
                timestamp = None

        # Write to InfluxDB (this is the critical part of Strategy A)
        influx = get_influx_client()
        try:
            await influx.write_prediction(
                building_id=request.building_id,
                appliance_id=request.appliance_id,
                predicted_kw=predicted_kw,
                confidence=confidence,
                model_version=entry.model_version,
                user_id=token.user_id,
                request_id=request_id,
                latency_ms=inference_latency_ms,
                timestamp=timestamp,
            )
            persisted = True

        except InfluxError:
            # Re-raise - persistence failed, caller should return 503
            raise

        logger.info(
            "Inference completed",
            extra={
                "building_id": request.building_id,
                "appliance_id": request.appliance_id,
                "model_id": entry.model_id,
                "predicted_kw": predicted_kw,
                "latency_ms": inference_latency_ms,
            },
        )

        return InferResponse(
            predicted_kw=predicted_kw,
            confidence=confidence,
            model_version=entry.model_version,
            request_id=request_id,
            persisted=persisted,
        )

    async def list_models(self) -> list[ModelInfo]:
        """List all available models."""
        registry = get_model_registry()
        engine = get_inference_engine()
        loaded_models = set(engine.get_loaded_models())

        models: list[ModelInfo] = []
        for entry in registry.list_all():
            cache_key = f"{entry.model_id}:{entry.model_version}"
            models.append(
                ModelInfo(
                    model_id=entry.model_id,
                    model_version=entry.model_version,
                    appliance_id=entry.appliance_id,
                    architecture=entry.architecture,
                    input_window_size=entry.input_window_size,
                    is_active=entry.is_active,
                    cached=cache_key in loaded_models,
                )
            )

        return models

    async def reload_models(self) -> dict[str, Any]:
        """Reload model registry and clear cache."""
        registry = get_model_registry()
        engine = get_inference_engine()

        # Clear cache
        cleared = engine.clear_cache()

        # Reload registry
        registry.reload()

        # Validate
        errors = registry.validate()

        return {
            "models_count": len(registry.list_all()),
            "cache_cleared": cleared,
            "validation_errors": errors,
        }


# Global service instance
_service: InferenceService | None = None


def get_inference_service() -> InferenceService:
    """Get the global inference service instance."""
    global _service
    if _service is None:
        _service = InferenceService()
    return _service
