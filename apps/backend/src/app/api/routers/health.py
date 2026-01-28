"""
Health check endpoints.
"""

from fastapi import APIRouter, Response, status
from pydantic import BaseModel

from app.api.deps import RequestIdDep, SettingsDep
from app.core.config import get_settings, validate_production_settings
from app.core.errors import ErrorCode
from app.core.logging import get_logger
from app.domain.inference import get_inference_engine
from app.domain.inference.registry import get_model_registry
from app.infra.influx import get_influx_client
from app.infra.redis import get_redis_cache

logger = get_logger(__name__)
router = APIRouter(tags=["Health"])


@router.get("/live")
async def liveness(request_id: RequestIdDep):
    """
    Liveness probe - process is running.

    No external dependencies checked.
    """
    return {
        "status": "ok",
        "request_id": request_id,
    }


@router.get("/ready")
async def readiness(request_id: RequestIdDep, response: Response):
    """
    Readiness probe - dependencies are available.
    
    Returns 503 if dependencies are not healthy.

    Checks:
    - InfluxDB reachable
    - Predictions bucket exists
    - Model registry valid
    """
    checks = {}
    is_ready = True

    # Check Configuration (Critical for Prod)
    settings = get_settings()
    if settings.env == "prod":
        config_errors = validate_production_settings(settings)
        checks["config_valid"] = len(config_errors) == 0
        if config_errors:
            is_ready = False
            # We log these in main.py, but returning them here helps debugging if safe.
            # Avoid exposing secrets, but these messages are generic.
            checks["config_errors"] = config_errors

    # Check InfluxDB (Connection + Buckets)
    influx = get_influx_client()
    influx_status = await influx.verify_setup()
    
    checks["influxdb_connected"] = influx_status["connected"]
    checks["influx_bucket_pred"] = influx_status["bucket_pred"]
    checks["influx_bucket_raw"] = influx_status.get("bucket_raw", False)

    # Only require predictions bucket - raw_readings is optional (graceful degradation)
    if not influx_status["connected"] or not influx_status["bucket_pred"]:
        is_ready = False

    # Check registry
    registry = get_model_registry()
    checks["registry_loaded"] = registry.is_loaded
    # We require at least one model or just the registry to be loaded? 
    # For now just loaded is enough, empty registry is valid state but useless.
    if not registry.is_loaded:
        is_ready = False
        
    checks["models_count"] = len(registry.list_all())

    # Check Redis (non-blocking - gracefully degrades to in-memory)
    redis_cache = get_redis_cache()
    checks["redis_available"] = not redis_cache.is_using_fallback
    # Redis unavailability doesn't fail readiness (graceful degradation)

    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "unavailable",
            "checks": checks,
            "request_id": request_id,
        }

    return {
        "status": "ok",
        "checks": checks,
        "request_id": request_id,
    }


@router.get("/health")
async def health(request_id: RequestIdDep, settings: SettingsDep):
    """
    Rich health information.

    In production, this endpoint may be protected or return limited info.
    """
    influx = get_influx_client()
    registry = get_model_registry()
    engine = get_inference_engine()

    # Basic info (always included)
    info = {
        "status": "ok",
        "request_id": request_id,
        "environment": settings.env,
    }

    # Detailed info (dev/test only)
    if settings.env != "prod":
        influx_ok = await influx.ping()
        info["details"] = {
            "influxdb": {
                "reachable": influx_ok,
                "url": settings.influx_url,
            },
            "models": {
                "registry_loaded": registry.is_loaded,
                "total_models": len(registry.list_all()),
                "loaded_models": engine.get_loaded_models(),
            },
        }

    return info
