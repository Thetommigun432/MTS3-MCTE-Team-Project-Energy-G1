"""
E2E Testing schemas for Railway pipeline validation.

These schemas define the request/response models for E2E probe endpoints
that allow testing the deployed pipeline without exposing internal infrastructure.
"""

from typing import Any

from pydantic import BaseModel, Field


class E2EInjectRequest(BaseModel):
    """Request to inject a sample into the pipeline via Redis pub/sub."""

    run_id: str = Field(..., description="Unique identifier for this test run")
    timestamp: float = Field(..., description="Unix timestamp for the sample")
    power_watts: float = Field(..., description="Total power in watts")
    building_id: str = Field(default="building_1", description="Building identifier")
    voltage: float = Field(default=230.0, description="Voltage in volts")
    current: float | None = Field(default=None, description="Current in amperes (calculated if not provided)")
    power_factor: float = Field(default=0.95, description="Power factor (0-1)")


class E2EInjectResponse(BaseModel):
    """Response from injecting a sample."""

    status: str = Field(..., description="Status: 'ok' or 'error'")
    run_id: str = Field(..., description="Echo of the run_id for confirmation")
    preprocessed: list[float] = Field(..., description="7-element feature vector from preprocessing")
    redis_published: bool = Field(..., description="Whether the sample was published to Redis")
    channel: str = Field(..., description="Redis channel the sample was published to")


class E2EPreprocessRequest(BaseModel):
    """Request to test preprocessing without side effects."""

    timestamp: float = Field(..., description="Unix timestamp")
    power_watts: float = Field(..., description="Power in watts")


class E2EPreprocessResponse(BaseModel):
    """Response from preprocessing test."""

    features: list[float] = Field(..., description="7-element feature vector")
    feature_names: list[str] = Field(
        ...,
        description="Names of each feature in order",
    )


class E2ERedisBufferResponse(BaseModel):
    """Response from Redis buffer status check."""

    building_id: str = Field(..., description="Building identifier")
    features_key: str = Field(..., description="Redis key for features buffer")
    buffer_length: int = Field(..., description="Current number of samples in buffer")
    window_size: int = Field(..., description="Required window size for inference")
    buffer_full: bool = Field(..., description="Whether buffer has enough samples")
    oldest_timestamp: float | None = Field(None, description="Timestamp of oldest sample")
    newest_timestamp: float | None = Field(None, description="Timestamp of newest sample")


class E2EInfluxStatusResponse(BaseModel):
    """Response from InfluxDB status check."""

    found: bool = Field(..., description="Whether records with run_id were found")
    run_id: str = Field(..., description="The run_id that was queried")
    records_count: int = Field(..., description="Number of records found")
    sample_record: dict[str, Any] | None = Field(None, description="First matching record (sanitized)")
    query_time_ms: float = Field(..., description="Query execution time in milliseconds")
