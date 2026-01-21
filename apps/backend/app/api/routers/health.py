"""
Health check endpoints.
"""

from fastapi import APIRouter, Response, status

from app.api.deps import RequestIdDep, SettingsDep
from app.domain.inference import get_inference_engine, get_model_registry
from app.infra.influx import get_influx_client

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
    - Both buckets exist (raw + predictions)
    - Model registry valid
    """
    checks = {}
    is_ready = True

    # Check InfluxDB (Connection + Buckets)
    influx = get_influx_client()
    influx_status = await influx.verify_setup()
    
    checks["influxdb_connected"] = influx_status["connected"]
    checks["influx_bucket_raw"] = influx_status["bucket_raw"]
    checks["influx_bucket_pred"] = influx_status["bucket_pred"]

    if not all(influx_status.values()):
        is_ready = False

    # Check registry
    registry = get_model_registry()
    checks["registry_loaded"] = registry.is_loaded
    # We require at least one model or just the registry to be loaded? 
    # For now just loaded is enough, empty registry is valid state but useless.
    if not registry.is_loaded:
        is_ready = False
        
    checks["models_count"] = len(registry.list_all())

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
