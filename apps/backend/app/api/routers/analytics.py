"""
Analytics API endpoints.
"""

from fastapi import APIRouter, Query

from app.api.deps import CurrentUserDep, RequestIdDep
from app.core.security import TokenPayload
from app.domain.authz import require_building_access
from app.infra.influx import get_influx_client
from app.schemas.analytics import (
    PredictionsResponse,
    ReadingsResponse,
    Resolution,
)

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/readings", response_model=ReadingsResponse)
async def get_readings(
    current_user: CurrentUserDep,
    request_id: RequestIdDep,
    building_id: str = Query(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$"),
    start: str = Query(..., description="Start time (ISO8601 or relative like -7d)"),
    end: str = Query(..., description="End time (ISO8601 or relative like now())"),
    appliance_id: str | None = Query(None, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$"),
    resolution: Resolution = Query(default=Resolution.ONE_MINUTE),
) -> ReadingsResponse:
    """
    Get sensor readings for a building.

    Time range can be specified as:
    - Relative: -7d, -1h, -30m
    - ISO8601: 2024-01-15T00:00:00Z
    """
    # AuthZ check
    await require_building_access(current_user, building_id)

    # Query InfluxDB
    influx = get_influx_client()
    data = await influx.query_readings(
        building_id=building_id,
        appliance_id=appliance_id,
        start=start,
        end=end,
        resolution=resolution,
    )

    return ReadingsResponse(
        building_id=building_id,
        appliance_id=appliance_id,
        start=start,
        end=end,
        resolution=resolution.value,
        data=data,
        count=len(data),
    )


@router.get("/predictions", response_model=PredictionsResponse)
async def get_predictions(
    current_user: CurrentUserDep,
    request_id: RequestIdDep,
    building_id: str = Query(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$"),
    start: str = Query(..., description="Start time (ISO8601 or relative like -7d)"),
    end: str = Query(..., description="End time (ISO8601 or relative like now())"),
    appliance_id: str | None = Query(None, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$"),
    resolution: Resolution = Query(default=Resolution.ONE_MINUTE),
) -> PredictionsResponse:
    """
    Get predictions for a building.

    Time range can be specified as:
    - Relative: -7d, -1h, -30m
    - ISO8601: 2024-01-15T00:00:00Z
    """
    # AuthZ check
    await require_building_access(current_user, building_id)

    # Query InfluxDB
    influx = get_influx_client()
    data = await influx.query_predictions(
        building_id=building_id,
        appliance_id=appliance_id,
        start=start,
        end=end,
        resolution=resolution,
    )

    return PredictionsResponse(
        building_id=building_id,
        appliance_id=appliance_id,
        start=start,
        end=end,
        resolution=resolution.value,
        data=data,
        count=len(data),
    )
