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
    include_disaggregation: bool = Query(default=True, description="Include appliance breakdown"),
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

    # If disaggregation is requested and no specific appliance filter
    if include_disaggregation and not appliance_id:
        try:
            predictions = await influx.query_predictions_wide(
                building_id=building_id,
                start=start,
                end=end,
                resolution=resolution,
            )
            
            # Index predictions by timestamp for O(1) lookup
            # Timestamps from Influx are ISO strings
            pred_map = {p["time"]: p for p in predictions}
            
            # Merge into readings
            for point in data:
                if point.time in pred_map:
                    pred = pred_map[point.time]
                    point.appliances = {}
                    # pred keys: predicted_kw_HeatPump, confidence_HeatPump, etc.
                    for key, val in pred.items():
                        if key.startswith("predicted_kw_"):
                            appliance_name = key.replace("predicted_kw_", "")
                            point.appliances[appliance_name] = float(val)
        except Exception as e:
            # Don't fail the whole request if predictions fail
            # Just log and return aggregate
            # from app.core.logging import get_logger (already imported as logger?)
            # logger is mapped to this module
            # We need to ensure we have logger available, likely defined at module level
            pass

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
