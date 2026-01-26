"""
Inference service orchestrating the full inference workflow.
Implements Predict & Persist (Strategy A) with multi-head model support.
"""

import time
from datetime import datetime, timezone
from typing import Any

from starlette.concurrency import run_in_threadpool

from app.core.errors import InfluxError, ModelError, ErrorCode
from app.core.logging import get_logger
from app.core.security import TokenPayload
from app.domain.authz import require_building_access
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
        Run multi-head inference and persist predictions to InfluxDB.

        Implements Strategy A: only return success if write succeeds.

        For multi-head models:
        - Returns predicted_kw and confidence as dicts mapping field_key to value
        - Persists ONE wide point with all predictions as fields

        Raises:
            AuthorizationError: If user doesn't have access
            ValidationError: If input is invalid
            ModelError: If model loading/inference fails
            InfluxError: If persistence fails
        """
        start_time = time.time()

        # AuthZ checks (building only for multi-head)
        await require_building_access(token, request.building_id)

        # Get model
        engine = get_inference_engine()
        model, entry = self._get_model_for_request(request)

        # Run multi-head inference in thread pool
        predictions = await run_in_threadpool(
            engine.run_inference_multi_head,
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

        # Write to InfluxDB using wide schema
        influx = get_influx_client()
        try:
            await influx.write_predictions_wide(
                building_id=request.building_id,
                predictions=predictions,
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

        # Convert predictions dict to response format
        predicted_kw_dict = {k: v[0] for k, v in predictions.items()}
        confidence_dict = {k: v[1] for k, v in predictions.items()}

        logger.info(
            "Multi-head inference completed",
            extra={
                "building_id": request.building_id,
                "model_id": entry.model_id,
                "heads_count": len(predictions),
                "latency_ms": inference_latency_ms,
            },
        )

        return InferResponse(
            predicted_kw=predicted_kw_dict,
            confidence=confidence_dict,
            model_version=entry.model_version,
            request_id=request_id,
            persisted=persisted,
        )

    def _get_model_for_request(self, request: InferRequest):
        """Get model for request, handling both explicit model_id and appliance_id lookup."""
        engine = get_inference_engine()
        registry = get_model_registry()

        if request.model_id:
            # Explicit model requested
            entry = registry.get(request.model_id)
            if not entry:
                raise ModelError(
                    code=ErrorCode.MODEL_NOT_FOUND,
                    message=f"Model not found: {request.model_id}",
                )
            model = engine._load_model(entry)
            return model, entry

        if request.appliance_id:
            # Get active model for specific appliance
            entry = registry.get_active_for_appliance(request.appliance_id)
            if not entry:
                raise ModelError(
                    code=ErrorCode.MODEL_NOT_FOUND,
                    message=f"No active model for appliance: {request.appliance_id}",
                )
            model = engine._load_model(entry)
            return model, entry

        # No model or appliance specified - get first active model
        all_entries = registry.list_all()
        for entry in all_entries:
            if entry.is_active:
                model = engine._load_model(entry)
                return model, entry

        raise ModelError(
            code=ErrorCode.MODEL_NOT_FOUND,
            message="No model specified and no active models available",
        )

    async def list_models(self) -> list[ModelInfo]:
        """List all available models with heads info."""
        from app.schemas.inference import HeadInfo
        
        registry = get_model_registry()
        engine = get_inference_engine()
        loaded_models = set(engine.get_loaded_models())

        models: list[ModelInfo] = []
        for entry in registry.list_all():
            cache_key = f"{entry.model_id}:{entry.model_version}"
            
            # Build heads list from registry entry
            heads = [
                HeadInfo(appliance_id=h.appliance_id, field_key=h.field_key)
                for h in entry.heads
            ] if entry.heads else []
            
            models.append(
                ModelInfo(
                    model_id=entry.model_id,
                    model_version=entry.model_version,
                    appliance_id=entry.appliance_id,
                    architecture=entry.architecture,
                    input_window_size=entry.input_window_size,
                    is_active=entry.is_active,
                    cached=cache_key in loaded_models,
                    heads=heads,
                    metrics=None,  # TODO: load from metrics.json if available
                )
            )

        return models


    async def get_model_details(self, model_id: str) -> dict[str, Any]:
        """Get detailed model information including architecture params."""
        from dataclasses import asdict
        
        registry = get_model_registry()
        engine = get_inference_engine()
        
        entry = registry.get(model_id)
        if not entry:
            raise ModelError(
                code=ErrorCode.MODEL_NOT_FOUND,
                message=f"Model not found: {model_id}",
            )
        
        loaded_models = set(engine.get_loaded_models())
        cache_key = f"{entry.model_id}:{entry.model_version}"
        
        # Convert preprocessing to dict
        preprocessing_dict = {
            "type": entry.preprocessing.type,
            "mean": entry.preprocessing.mean,
            "std": entry.preprocessing.std,
            "min": entry.preprocessing.min_val,
            "max": entry.preprocessing.max_val,
        }
        
        return {
            "model_id": entry.model_id,
            "model_version": entry.model_version,
            "appliance_id": entry.appliance_id,
            "architecture": entry.architecture,
            "architecture_params": entry.architecture_params,
            "input_window_size": entry.input_window_size,
            "preprocessing": preprocessing_dict,
            "is_active": entry.is_active,
            "cached": cache_key in loaded_models,
        }

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
