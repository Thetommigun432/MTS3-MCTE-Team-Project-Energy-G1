"""
Pydantic v2 schemas for analytics endpoints.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Resolution(str, Enum):
    """Allowed resolution values for analytics queries."""

    ONE_SECOND = "1s"
    ONE_MINUTE = "1m"
    FIFTEEN_MINUTES = "15m"


class AnalyticsQueryParams(BaseModel):
    """Query parameters for analytics endpoints."""

    model_config = ConfigDict(extra="forbid")

    building_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Building ID",
    )
    appliance_id: str | None = Field(
        None,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Optional appliance ID filter",
    )
    start: str = Field(
        ...,
        description="Start time (ISO8601 or relative like -7d)",
    )
    end: str = Field(
        ...,
        description="End time (ISO8601 or relative like now())",
    )
    resolution: Resolution = Field(
        default=Resolution.ONE_MINUTE,
        description="Data resolution: 1s, 1m, or 15m",
    )


class DataPoint(BaseModel):
    """Single data point in a time series."""

    model_config = ConfigDict(extra="allow")

    time: str = Field(..., description="ISO8601 timestamp")
    value: float = Field(..., description="Value at this timestamp")


class ReadingsResponse(BaseModel):
    """Response schema for GET /analytics/readings."""

    model_config = ConfigDict(extra="forbid")

    building_id: str = Field(..., description="Building ID")
    appliance_id: str | None = Field(None, description="Appliance ID if filtered")
    start: str = Field(..., description="Query start time")
    end: str = Field(..., description="Query end time")
    resolution: str = Field(..., description="Data resolution")
    data: list[DataPoint] = Field(..., description="Time series data")
    count: int = Field(..., description="Number of data points")


class PredictionPoint(BaseModel):
    """Single prediction data point."""

    model_config = ConfigDict(extra="allow")

    time: str = Field(..., description="ISO8601 timestamp")
    predicted_kw: float = Field(..., description="Predicted power in kW")
    confidence: float | None = Field(None, description="Prediction confidence")
    model_version: str | None = Field(None, description="Model version")


class PredictionsResponse(BaseModel):
    """Response schema for GET /analytics/predictions."""

    model_config = ConfigDict(extra="forbid")

    building_id: str = Field(..., description="Building ID")
    appliance_id: str | None = Field(None, description="Appliance ID if filtered")
    start: str = Field(..., description="Query start time")
    end: str = Field(..., description="Query end time")
    resolution: str = Field(..., description="Data resolution")
    data: list[PredictionPoint] = Field(..., description="Prediction time series")
    count: int = Field(..., description="Number of data points")
