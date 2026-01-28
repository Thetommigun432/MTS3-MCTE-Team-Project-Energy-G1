"""
Safe Flux query templates for InfluxDB.
Prevents injection by using fixed templates with validated parameters.
"""

import re
from datetime import datetime, timezone
from typing import Literal

from app.core.errors import ErrorCode, ValidationError
from app.schemas.analytics import Resolution


# ID validation pattern
ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

# Relative time pattern (e.g., -7d, -1h, -30m) - minus sign required
RELATIVE_TIME_PATTERN = re.compile(r"^-\d+[smhdw]$")

# ISO8601 datetime pattern (simplified)
ISO_DATETIME_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?)?$"
)


def validate_id(value: str, field_name: str = "id") -> str:
    """
    Validate an ID matches the safe pattern.

    Raises:
        ValidationError: If ID is invalid
    """
    if not ID_PATTERN.match(value):
        raise ValidationError(
            code=ErrorCode.VALIDATION_INVALID_ID,
            message=f"Invalid {field_name}: must be 1-64 alphanumeric characters, dashes, or underscores",
            details={field_name: value},
        )
    return value


def validate_and_convert_time(value: str, field_name: str = "time") -> str:
    """
    Validate and convert a time parameter to Flux-compatible format.

    Accepts:
    - Relative time: -7d, -1h, -30m, etc.
    - ISO8601 datetime: 2024-01-15T00:00:00Z
    - now()

    Returns:
        Flux-compatible time string (quoted for ISO, unquoted for relative)

    Raises:
        ValidationError: If time format is invalid
    """
    if value == "now()":
        return "now()"

    # Check relative time format
    if RELATIVE_TIME_PATTERN.match(value):
        return value

    # Check ISO8601 format
    if ISO_DATETIME_PATTERN.match(value):
        # Return as quoted string for Flux
        return value

    raise ValidationError(
        code=ErrorCode.VALIDATION_INVALID_TIMESTAMP,
        message=f"Invalid {field_name}: expected relative time (-7d) or ISO8601 datetime",
        details={field_name: value},
    )


def resolution_to_flux(resolution: Resolution) -> str:
    """Convert Resolution enum to Flux duration string."""
    mapping = {
        Resolution.ONE_SECOND: "1s",
        Resolution.ONE_MINUTE: "1m",
        Resolution.FIFTEEN_MINUTES: "15m",
    }
    return mapping[resolution]


def build_readings_query(
    bucket: str,
    building_id: str,
    appliance_id: str | None,
    start: str,
    end: str,
    resolution: Resolution,
) -> str:
    """
    Build a safe Flux query for reading sensor data.

    All parameters are validated before interpolation.
    """
    # Validate all inputs
    validate_id(building_id, "building_id")
    if appliance_id:
        validate_id(appliance_id, "appliance_id")

    start_flux = validate_and_convert_time(start, "start")
    end_flux = validate_and_convert_time(end, "end")
    resolution_flux = resolution_to_flux(resolution)

    # Format time for Flux range()
    # Relative times like -7d are used as-is
    # ISO8601 timestamps need to be wrapped for time()
    def format_time_param(t: str) -> str:
        if t == "now()" or RELATIVE_TIME_PATTERN.match(t):
            return t
        return f'time(v: "{t}")'

    start_param = format_time_param(start_flux)
    end_param = format_time_param(end_flux)

    # Build query with validated parameters
    # Using double quotes for string literals in Flux
    appliance_filter = ""
    if appliance_id:
        appliance_filter = f'  |> filter(fn: (r) => r.appliance_id == "{appliance_id}")\n'

    query = f'''from(bucket: "{bucket}")
  |> range(start: {start_param}, stop: {end_param})
  |> filter(fn: (r) => r._measurement == "sensor_reading")
  |> filter(fn: (r) => r.building_id == "{building_id}")
{appliance_filter}  |> aggregateWindow(every: {resolution_flux}, fn: mean, createEmpty: false)
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
'''
    return query


def build_predictions_query(
    bucket: str,
    building_id: str,
    appliance_id: str | None,
    start: str,
    end: str,
    resolution: Resolution,
) -> str:
    """
    Build a safe Flux query for reading prediction data.

    All parameters are validated before interpolation.
    """
    # Validate all inputs
    validate_id(building_id, "building_id")
    if appliance_id:
        validate_id(appliance_id, "appliance_id")

    start_flux = validate_and_convert_time(start, "start")
    end_flux = validate_and_convert_time(end, "end")
    resolution_flux = resolution_to_flux(resolution)

    def format_time_param(t: str) -> str:
        if t == "now()" or RELATIVE_TIME_PATTERN.match(t):
            return t
        return f'time(v: "{t}")'

    start_param = format_time_param(start_flux)
    end_param = format_time_param(end_flux)

    appliance_filter = ""
    if appliance_id:
        appliance_filter = f'  |> filter(fn: (r) => r.appliance_id == "{appliance_id}")\n'

    query = f'''from(bucket: "{bucket}")
  |> range(start: {start_param}, stop: {end_param})
  |> filter(fn: (r) => r._measurement == "prediction")
  |> filter(fn: (r) => r.building_id == "{building_id}")
{appliance_filter}  |> aggregateWindow(every: {resolution_flux}, fn: mean, createEmpty: false)
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
'''
    return query


def build_predictions_wide_query(
    bucket: str,
    building_id: str,
    start: str,
    end: str,
    resolution: Resolution,
) -> str:
    """
    Build a safe Flux query for reading wide-format prediction data.
    
    Filters to only numeric fields (predicted_kw_*, confidence_*, latency_ms)
    to allow mean() aggregation. String fields like request_id and user_id
    are excluded.
    """
    validate_id(building_id, "building_id")

    start_flux = validate_and_convert_time(start, "start")
    end_flux = validate_and_convert_time(end, "end")
    resolution_flux = resolution_to_flux(resolution)

    def format_time_param(t: str) -> str:
        if t == "now()" or RELATIVE_TIME_PATTERN.match(t):
            return t
        return f'time(v: "{t}")'

    start_param = format_time_param(start_flux)
    end_param = format_time_param(end_flux)

    # IMPORTANT: measurement name must match write operations (client.py uses "prediction" singular)
    # Filter to only numeric fields to allow mean() aggregation
    # String fields (request_id, user_id) would cause "unsupported input type for mean aggregate: string"
    query = f'''from(bucket: "{bucket}")
  |> range(start: {start_param}, stop: {end_param})
  |> filter(fn: (r) => r._measurement == "prediction")
  |> filter(fn: (r) => r.building_id == "{building_id}")
  |> filter(fn: (r) => r._field =~ /^(predicted_kw_|confidence_|latency_ms)/)
  |> aggregateWindow(every: {resolution_flux}, fn: mean, createEmpty: false)
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
'''
    return query
