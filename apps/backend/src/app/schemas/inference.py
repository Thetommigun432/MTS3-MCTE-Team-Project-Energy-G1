"""
Pydantic v2 schemas for inference endpoints.
"""

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.common import ValidatedId


class InferRequest(BaseModel):
    """Request schema for POST /infer endpoint."""

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
        min_length=1,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Appliance ID (optional for multi-head models)",
    )
    window: list[float] = Field(
        ...,
        description="Input window of power readings (1000 floats by default)",
    )
    timestamp: str | None = Field(
        None,
        description="Optional ISO8601 timestamp for the prediction",
    )
    model_id: str | None = Field(
        None,
        description="Optional model ID (if absent, use default active model)",
    )

    @field_validator("window")
    @classmethod
    def validate_window(cls, v: list[float]) -> list[float]:
        """Validate window data: check for NaN/Inf values."""
        for i, val in enumerate(v):
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"Window contains invalid value at index {i}: {val}")
        return v

    @model_validator(mode="after")
    def validate_window_length(self) -> "InferRequest":
        """Validate window length (default 1000, but configurable per model)."""
        # Note: actual length validation happens in the service layer
        # based on the model's input_window_size configuration
        if len(self.window) < 1:
            raise ValueError("Window must contain at least 1 value")
        if len(self.window) > 10000:
            raise ValueError("Window exceeds maximum length of 10000")
        return self


class InferResponse(BaseModel):
    """
    Response schema for POST /infer endpoint (success).
    
    For multi-head models, predicted_kw and confidence are dicts
    mapping appliance field_key to value.
    """

    model_config = ConfigDict(extra="forbid")

    predicted_kw: dict[str, float] = Field(
        ...,
        description="Predicted power in kW per appliance (e.g., {'fridge': 0.05, 'oven': 0.0})",
    )
    confidence: dict[str, float] = Field(
        ...,
        description="Prediction confidence per appliance (e.g., {'fridge': 0.98, 'oven': 0.63})",
    )
    model_version: str = Field(..., description="Model version used")
    request_id: str = Field(..., description="Request ID for tracing")
    persisted: bool = Field(..., description="Whether prediction was persisted to InfluxDB")


class HeadInfo(BaseModel):
    """Output head information for multi-head models."""

    model_config = ConfigDict(extra="forbid")

    appliance_id: str = Field(..., description="Appliance identifier (e.g., 'HeatPump')")
    field_key: str = Field(..., description="InfluxDB field key (e.g., 'HeatPump')")


class ModelMetrics(BaseModel):
    """Basic model performance metrics."""

    model_config = ConfigDict(extra="forbid")

    mae: float | None = Field(None, description="Mean Absolute Error (kW)")
    rmse: float | None = Field(None, description="Root Mean Square Error (kW)")
    f1_score: float | None = Field(None, description="F1 score for on/off classification")
    accuracy: float | None = Field(None, description="Classification accuracy")


class ModelInfo(BaseModel):
    """Model information schema."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(..., description="Model ID")
    model_version: str = Field(..., description="Model version")
    appliance_id: str = Field(..., description="Primary appliance ID (or 'multi' for multi-head)")
    architecture: str = Field(..., description="Model architecture (e.g., CNNTransformer)")
    input_window_size: int = Field(..., description="Required input window size")
    is_active: bool = Field(..., description="Whether this is the active model")
    cached: bool = Field(default=False, description="Whether model is loaded in cache")
    heads: list[HeadInfo] = Field(default_factory=list, description="Output heads (empty for single-head models)")
    metrics: ModelMetrics | None = Field(None, description="Performance metrics (if available)")


class ModelsListResponse(BaseModel):
    """Response schema for GET /models endpoint."""

    model_config = ConfigDict(extra="forbid")

    models: list[ModelInfo] = Field(..., description="List of available models")
    count: int = Field(..., description="Total number of models")


class ModelMetricsResponse(BaseModel):
    """Response schema for GET /models/{id}/metrics endpoint."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(..., description="Model ID")
    model_version: str = Field(..., description="Model version")
    appliance_id: str = Field(..., description="Associated appliance ID")
    architecture: str = Field(..., description="Model architecture")
    architecture_params: dict[str, Any] = Field(
        default_factory=dict, description="Architecture parameters (n_blocks, hidden_channels, etc.)"
    )
    input_window_size: int = Field(..., description="Required input window size")
    preprocessing: dict[str, Any] = Field(
        default_factory=dict, description="Preprocessing config (type, max, etc.)"
    )
    is_active: bool = Field(..., description="Whether this is the active model")
    cached: bool = Field(default=False, description="Whether model is loaded in cache")
