"""
Analytics API endpoints.
Provides readings and predictions data from InfluxDB.
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field

from app.api.deps import CurrentUserDep, OptionalUserDep, RequestIdDep
from app.core.logging import get_logger
from app.domain.authz import require_building_access_or_demo
from app.infra.influx import get_influx_client
from app.schemas.analytics import (
    DataPoint,
    PredictionPoint,
    PredictionsResponse,
    ReadingsResponse,
    BuildingsListResponse,
    Resolution,
)

router = APIRouter(prefix="/analytics", tags=["Analytics"])
logger = get_logger(__name__)


@router.get("/readings", response_model=ReadingsResponse)
async def get_readings(
    current_user: OptionalUserDep,
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

    If include_disaggregation=true (default), the response includes per-appliance
    predictions attached to each reading timestamp.
    """
    # AuthZ check - allows unauthenticated access to demo buildings
    await require_building_access_or_demo(current_user, building_id)

    influx = get_influx_client()

    # Query raw sensor readings from InfluxDB (measurement: sensor_reading)
    try:
        data = await influx.query_readings(
            building_id=building_id,
            start=start,
            end=end,
            resolution=resolution,
        )
    except Exception as e:
        logger.warning(f"Failed to query readings: {e}")
        data = []

    # If disaggregation is requested and no specific appliance filter
    if include_disaggregation and not appliance_id and data:
        try:
            predictions = await influx.query_predictions_wide(
                building_id=building_id,
                start=start,
                end=end,
                resolution=resolution,
            )

            # Index predictions by timestamp for O(1) lookup
            pred_map = {p["time"]: p for p in predictions}

            # Merge appliance predictions into readings
            for point in data:
                if point.time in pred_map:
                    pred = pred_map[point.time]
                    appliances = {}
                    confidence = {}
                    for key, val in pred.items():
                        if key.startswith("predicted_kw_") and val is not None:
                            appliance_name = key.replace("predicted_kw_", "")
                            appliances[appliance_name] = float(val)
                        elif key.startswith("confidence_") and val is not None:
                            appliance_name = key.replace("confidence_", "")
                            confidence[appliance_name] = float(val)
                    if appliances:
                        point.appliances = appliances
                    if confidence:
                        point.confidence = confidence
        except Exception as e:
            logger.warning(f"Failed to merge disaggregation data: {e}")

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
    current_user: OptionalUserDep,
    request_id: RequestIdDep,
    building_id: str = Query(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$"),
    start: str = Query(..., description="Start time (ISO8601 or relative like -7d)"),
    end: str = Query(..., description="End time (ISO8601 or relative like now())"),
    appliance_id: str | None = Query(None, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$"),
    resolution: Resolution = Query(default=Resolution.ONE_MINUTE),
) -> PredictionsResponse:
    """
    Get predictions for a building.

    For WIDE format predictions (multi-appliance per timestamp):
    - If appliance_id is provided, returns that specific appliance's series
    - If appliance_id is omitted, returns the first available appliance series

    Time range can be specified as:
    - Relative: -7d, -1h, -30m
    - ISO8601: 2024-01-15T00:00:00Z
    """
    # AuthZ check - allows unauthenticated access to demo buildings
    await require_building_access_or_demo(current_user, building_id)

    influx = get_influx_client()

    # Query wide-format predictions
    try:
        wide_predictions = await influx.query_predictions_wide(
            building_id=building_id,
            start=start,
            end=end,
            resolution=resolution,
        )
    except Exception as e:
        logger.error(f"Failed to query predictions: {e}")
        wide_predictions = []

    data: list[PredictionPoint] = []

    if wide_predictions:
        if appliance_id:
            # Extract specific appliance series
            data = influx.extract_appliance_series_from_wide(wide_predictions, appliance_id)
        else:
            # Find first available appliance from field keys
            first_row = wide_predictions[0]
            first_appliance = None
            for key in first_row.keys():
                if key.startswith("predicted_kw_"):
                    first_appliance = key.replace("predicted_kw_", "")
                    break

            if first_appliance:
                data = influx.extract_appliance_series_from_wide(wide_predictions, first_appliance)
                appliance_id = first_appliance

    return PredictionsResponse(
        building_id=building_id,
        appliance_id=appliance_id,
        start=start,
        end=end,
        resolution=resolution.value,
        data=data,
        count=len(data),
    )


@router.get("/buildings", response_model=BuildingsListResponse)
async def list_buildings(
    current_user: OptionalUserDep,
    request_id: RequestIdDep,
) -> BuildingsListResponse:
    """
    List unique building IDs found in InfluxDB (last 30 days).
    """
    influx = get_influx_client()
    buildings = await influx.get_unique_buildings()
    return BuildingsListResponse(buildings=buildings)


class AppliancesListResponse(BaseModel):
    """Response schema for GET /analytics/appliances."""

    model_config = ConfigDict(extra="forbid")

    appliances: list[str] = Field(..., description="List of appliance IDs")


@router.get("/appliances", response_model=AppliancesListResponse)
async def list_appliances(
    current_user: OptionalUserDep,
    request_id: RequestIdDep,
    building_id: str = Query(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$"),
) -> AppliancesListResponse:
    """
    List unique appliance IDs for a building.

    For WIDE format predictions, this parses field keys like predicted_kw_HeatPump
    to extract appliance names. Falls back to appliance_id tags for legacy data.
    """
    # AuthZ check - allows unauthenticated access to demo buildings
    await require_building_access_or_demo(current_user, building_id)
    
    influx = get_influx_client()
    appliances = await influx.get_unique_appliances(building_id=building_id)
    return AppliancesListResponse(appliances=appliances)
