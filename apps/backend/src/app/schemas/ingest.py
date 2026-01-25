"""
Schemas for data ingestion endpoints.
"""

from datetime import datetime
from typing import List

from pydantic import BaseModel, Field, field_validator


class IngestReading(BaseModel):
    """A single power reading to ingest."""

    building_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Building identifier",
    )
    ts: datetime = Field(
        ...,
        description="Timestamp of the reading (ISO8601)",
    )
    aggregate_kw: float = Field(
        ...,
        ge=0,
        description="Aggregate power consumption in kilowatts",
    )

    @field_validator("aggregate_kw")
    @classmethod
    def validate_power(cls, v: float) -> float:
        """Ensure power value is reasonable."""
        if v > 1000:  # 1 MW seems like a reasonable max for residential
            raise ValueError("aggregate_kw exceeds maximum reasonable value (1000 kW)")
        return v


class IngestBatchRequest(BaseModel):
    """Batch of readings to ingest."""

    readings: List[IngestReading] = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="List of readings to ingest (max 10,000 per batch)",
    )


class IngestBatchResponse(BaseModel):
    """Response from batch ingestion."""

    ingested: int = Field(..., description="Number of readings ingested")
    errors: int = Field(default=0, description="Number of readings that failed")
    message: str = Field(default="OK", description="Status message")
