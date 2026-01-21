"""
Pydantic v2 schemas for common types.
"""

import re
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ID pattern validation: alphanumeric, dash, underscore, max 64 chars
ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def validate_id(value: str, field_name: str = "id") -> str:
    """Validate an ID matches the required pattern."""
    if not ID_PATTERN.match(value):
        raise ValueError(
            f"{field_name} must be 1-64 alphanumeric characters, dashes, or underscores"
        )
    return value


class ErrorDetail(BaseModel):
    """Error detail in API response."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional error details")


class ErrorResponse(BaseModel):
    """Standard error response schema."""

    model_config = ConfigDict(extra="forbid")

    error: ErrorDetail
    request_id: str | None = Field(None, description="Request ID for tracing")


class PaginationParams(BaseModel):
    """Pagination parameters for list endpoints."""

    offset: int = Field(default=0, ge=0, description="Number of items to skip")
    limit: int = Field(default=100, ge=1, le=1000, description="Maximum items to return")


# Type alias for validated IDs
ValidatedId = Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")]
