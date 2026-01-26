"""
Unit tests for Flux query builder.
"""

import pytest

from app.core.errors import ValidationError
from app.infra.influx.queries import (
    build_predictions_query,
    build_readings_query,
    validate_and_convert_time,
    validate_id,
)
from app.schemas.analytics import Resolution


class TestValidateId:
    """Tests for ID validation."""

    def test_valid_ids(self):
        """Test valid ID patterns."""
        valid_ids = [
            "building_123",
            "app-456",
            "heatpump",
            "A1B2C3",
            "a" * 64,  # Max length
        ]
        for id_val in valid_ids:
            result = validate_id(id_val, "test_id")
            assert result == id_val

    def test_invalid_ids_rejected(self):
        """Test invalid IDs are rejected."""
        invalid_ids = [
            "building 123",  # Space
            "app/456",  # Slash
            "heat.pump",  # Dot
            "a" * 65,  # Too long
            "",  # Empty
            "app@123",  # Special char
        ]
        for id_val in invalid_ids:
            with pytest.raises(ValidationError):
                validate_id(id_val, "test_id")


class TestValidateTime:
    """Tests for time validation."""

    def test_relative_times(self):
        """Test relative time formats."""
        valid = ["-7d", "-1h", "-30m", "-60s", "-2w"]
        for t in valid:
            result = validate_and_convert_time(t, "test")
            assert result == t

    def test_iso_times(self):
        """Test ISO8601 time formats."""
        valid = [
            "2024-01-15",
            "2024-01-15T00:00:00Z",
            "2024-01-15T12:30:00+00:00",
        ]
        for t in valid:
            result = validate_and_convert_time(t, "test")
            assert result == t

    def test_now(self):
        """Test now() is valid."""
        result = validate_and_convert_time("now()", "test")
        assert result == "now()"

    def test_invalid_times_rejected(self):
        """Test invalid time formats are rejected."""
        invalid = [
            "yesterday",
            "last week",
            "7d",  # Missing minus
            "2024/01/15",  # Wrong separator
        ]
        for t in invalid:
            with pytest.raises(ValidationError):
                validate_and_convert_time(t, "test")


class TestBuildQueries:
    """Tests for query builders."""

    def test_readings_query_basic(self):
        """Test basic readings query."""
        query = build_readings_query(
            bucket="predictions",
            building_id="bldg_123",
            appliance_id=None,
            start="-7d",
            end="now()",
            resolution=Resolution.ONE_MINUTE,
        )
        assert 'bucket: "predictions"' in query
        assert 'building_id == "bldg_123"' in query
        assert "aggregateWindow(every: 1m" in query

    def test_readings_query_with_appliance(self):
        """Test readings query with appliance filter."""
        query = build_readings_query(
            bucket="predictions",
            building_id="bldg_123",
            appliance_id="heatpump",
            start="-7d",
            end="now()",
            resolution=Resolution.ONE_MINUTE,
        )
        assert 'appliance_id == "heatpump"' in query

    def test_predictions_query_resolution(self):
        """Test predictions query with different resolutions."""
        for resolution, expected in [
            (Resolution.ONE_SECOND, "1s"),
            (Resolution.ONE_MINUTE, "1m"),
            (Resolution.FIFTEEN_MINUTES, "15m"),
        ]:
            query = build_predictions_query(
                bucket="predictions",
                building_id="bldg_123",
                appliance_id=None,
                start="-7d",
                end="now()",
                resolution=resolution,
            )
            assert f"aggregateWindow(every: {expected}" in query

    def test_query_sanitizes_ids(self):
        """Test that invalid IDs are rejected before query building."""
        with pytest.raises(ValidationError):
            build_readings_query(
                bucket="test",
                building_id="invalid id with spaces",
                appliance_id=None,
                start="-7d",
                end="now()",
                resolution=Resolution.ONE_MINUTE,
            )
