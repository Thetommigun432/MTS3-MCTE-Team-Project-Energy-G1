"""
Health check endpoints.
"""

from fastapi import APIRouter, Request

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
async def readiness(request_id: RequestIdDep):
    """
    Readiness probe - dependencies are available.

    Checks:
    - InfluxDB reachable
    - Predictions bucket exists
    - Model registry valid
    """
    checks = {}
    ready = True

    # Check InfluxDB
    influx = get_influx_client()
    influx_ok = await influx.ping()
    checks["influxdb"] = "ok" if influx_ok else "unavailable"
    if not influx_ok:
        ready = False

    # Check predictions bucket
    settings = get_influx_client()
    bucket_ok = await influx.bucket_exists("predictions")
    checks["predictions_bucket"] = "ok" if bucket_ok else "missing"
    if not bucket_ok:
        ready = False

    # Check registry
    registry = get_model_registry()
    checks["registry"] = "ok" if registry.is_loaded else "not_loaded"
    checks["models_count"] = len(registry.list_all())

    return {
        "status": "ok" if ready else "unavailable",
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
