"""
Unit tests for input validation.
"""

import math
import pytest
from pydantic import ValidationError

from app.schemas.inference import InferRequest


class TestInferRequest:
    """Tests for InferRequest validation."""

    def test_valid_request(self):
        """Test valid inference request."""
        request = InferRequest(
            building_id="bldg_123",
            appliance_id="app_456",
            window=[1.0] * 1000,
        )
        assert request.building_id == "bldg_123"
        assert request.appliance_id == "app_456"
        assert len(request.window) == 1000

    def test_window_with_nan_rejected(self):
        """Test that NaN values in window are rejected."""
        window = [1.0] * 999 + [float("nan")]
        with pytest.raises(ValidationError) as exc_info:
            InferRequest(
                building_id="bldg_123",
                appliance_id="app_456",
                window=window,
            )
        assert "invalid value" in str(exc_info.value).lower()

    def test_window_with_inf_rejected(self):
        """Test that Inf values in window are rejected."""
        window = [1.0] * 999 + [float("inf")]
        with pytest.raises(ValidationError) as exc_info:
            InferRequest(
                building_id="bldg_123",
                appliance_id="app_456",
                window=window,
            )
        assert "invalid value" in str(exc_info.value).lower()

    def test_window_with_negative_inf_rejected(self):
        """Test that -Inf values in window are rejected."""
        window = [1.0] * 999 + [float("-inf")]
        with pytest.raises(ValidationError) as exc_info:
            InferRequest(
                building_id="bldg_123",
                appliance_id="app_456",
                window=window,
            )
        assert "invalid value" in str(exc_info.value).lower()

    def test_empty_window_rejected(self):
        """Test that empty window is rejected."""
        with pytest.raises(ValidationError):
            InferRequest(
                building_id="bldg_123",
                appliance_id="app_456",
                window=[],
            )

    def test_oversized_window_rejected(self):
        """Test that windows over 10000 are rejected."""
        with pytest.raises(ValidationError):
            InferRequest(
                building_id="bldg_123",
                appliance_id="app_456",
                window=[1.0] * 10001,
            )

    def test_invalid_building_id_rejected(self):
        """Test that invalid building IDs are rejected."""
        with pytest.raises(ValidationError):
            InferRequest(
                building_id="bldg 123",  # Space not allowed
                appliance_id="app_456",
                window=[1.0] * 1000,
            )

    def test_invalid_appliance_id_rejected(self):
        """Test that invalid appliance IDs are rejected."""
        with pytest.raises(ValidationError):
            InferRequest(
                building_id="bldg_123",
                appliance_id="app/456",  # Slash not allowed
                window=[1.0] * 1000,
            )

    def test_optional_fields(self):
        """Test optional fields."""
        request = InferRequest(
            building_id="bldg_123",
            appliance_id="app_456",
            window=[1.0] * 1000,
            timestamp="2024-01-15T00:00:00Z",
            model_id="model_v1",
        )
        assert request.timestamp == "2024-01-15T00:00:00Z"
        assert request.model_id == "model_v1"
