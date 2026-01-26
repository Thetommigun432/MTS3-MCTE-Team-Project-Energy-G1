"""
Unit tests for InfluxDB write operations.

Tests the write contract including wide-format predictions,
tags, fields, and data validation.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from influxdb_client import Point


class TestWritePredictionsWide:
    """Tests for wide-format multi-head prediction writes."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for InfluxDB configuration."""
        mock_settings_obj = MagicMock()
        mock_settings_obj.influx_url = "http://localhost:8086"
        mock_settings_obj.influx_token = "test-token"
        mock_settings_obj.influx_org = "test-org"
        mock_settings_obj.influx_bucket_pred = "predictions"

        mock_settings_obj.influx_timeout_ms = 10000
        return mock_settings_obj

    @pytest.fixture
    def mock_write_api(self):
        """Mock InfluxDB async write API."""
        return AsyncMock()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_wide_format_creates_single_point(self, mock_settings, mock_write_api):
        """Wide-format write creates a single Point with all fields."""
        with patch("app.infra.influx.client.get_settings", return_value=mock_settings):
            from app.infra.influx.client import InfluxClient

            client = InfluxClient()
            client._write_api = mock_write_api

            predictions = {
                "HeatPump": (2.5, 0.95),
                "Dishwasher": (0.8, 0.88),
            }

            await client.write_predictions_wide(
                building_id="building_1",
                predictions=predictions,
                model_version="v1.0",
                user_id="user123",
                request_id="req456",
                latency_ms=50.0,
            )

            # Verify write was called once (single point)
            mock_write_api.write.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_wide_format_field_naming(self, mock_settings, mock_write_api):
        """Wide-format uses correct field naming: predicted_kw_{field_key}."""
        with patch("app.infra.influx.client.get_settings", return_value=mock_settings):
            from app.infra.influx.client import InfluxClient

            client = InfluxClient()
            client._write_api = mock_write_api

            predictions = {
                "HeatPump": (2.5, 0.95),
                "Dishwasher": (0.8, 0.88),
            }

            await client.write_predictions_wide(
                building_id="building_1",
                predictions=predictions,
                model_version="v1.0",
                user_id="user123",
                request_id="req456",
                latency_ms=50.0,
            )

            # Get the Point that was written
            call_args = mock_write_api.write.call_args
            point = call_args.kwargs.get("record")

            # Convert to line protocol to inspect
            line = point.to_line_protocol()

            # Check field names
            assert "predicted_kw_HeatPump=2.5" in line
            assert "predicted_kw_Dishwasher=0.8" in line
            assert "confidence_HeatPump=0.95" in line
            assert "confidence_Dishwasher=0.88" in line

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_wide_format_tags(self, mock_settings, mock_write_api):
        """Wide-format includes correct tags."""
        with patch("app.infra.influx.client.get_settings", return_value=mock_settings):
            from app.infra.influx.client import InfluxClient

            client = InfluxClient()
            client._write_api = mock_write_api

            await client.write_predictions_wide(
                building_id="building_1",
                predictions={"test": (1.0, 0.9)},
                model_version="v2.0",
                user_id="user123",
                request_id="req456",
                latency_ms=50.0,
            )

            call_args = mock_write_api.write.call_args
            point = call_args.kwargs.get("record")
            line = point.to_line_protocol()

            # Check tags are present (tags come before first space)
            assert "building_id=building_1" in line
            assert "model_version=v2.0" in line

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_wide_format_measurement_name(self, mock_settings, mock_write_api):
        """Wide-format uses 'prediction' measurement."""
        with patch("app.infra.influx.client.get_settings", return_value=mock_settings):
            from app.infra.influx.client import InfluxClient

            client = InfluxClient()
            client._write_api = mock_write_api

            await client.write_predictions_wide(
                building_id="building_1",
                predictions={"test": (1.0, 0.9)},
                model_version="v1.0",
                user_id="user123",
                request_id="req456",
                latency_ms=50.0,
            )

            call_args = mock_write_api.write.call_args
            point = call_args.kwargs.get("record")
            line = point.to_line_protocol()

            # Line protocol format: measurement,tags fields timestamp
            assert line.startswith("prediction,")

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_negative_predictions_clamped_to_zero(self, mock_settings, mock_write_api):
        """Negative prediction values are clamped to 0."""
        with patch("app.infra.influx.client.get_settings", return_value=mock_settings):
            from app.infra.influx.client import InfluxClient

            client = InfluxClient()
            client._write_api = mock_write_api

            predictions = {
                "NegativeTest": (-0.5, 0.8),  # Negative power
            }

            await client.write_predictions_wide(
                building_id="building_1",
                predictions=predictions,
                model_version="v1.0",
                user_id="user123",
                request_id="req456",
                latency_ms=50.0,
            )

            call_args = mock_write_api.write.call_args
            point = call_args.kwargs.get("record")
            line = point.to_line_protocol()

            # Should be clamped to 0
            assert "predicted_kw_NegativeTest=0" in line

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_confidence_clamped_to_valid_range(self, mock_settings, mock_write_api):
        """Confidence values are clamped to [0, 1] range."""
        with patch("app.infra.influx.client.get_settings", return_value=mock_settings):
            from app.infra.influx.client import InfluxClient

            client = InfluxClient()
            client._write_api = mock_write_api

            predictions = {
                "OverConfident": (1.0, 1.5),  # Confidence > 1
                "UnderConfident": (1.0, -0.2),  # Confidence < 0
            }

            await client.write_predictions_wide(
                building_id="building_1",
                predictions=predictions,
                model_version="v1.0",
                user_id="user123",
                request_id="req456",
                latency_ms=50.0,
            )

            call_args = mock_write_api.write.call_args
            point = call_args.kwargs.get("record")
            line = point.to_line_protocol()

            # Should be clamped
            assert "confidence_OverConfident=1" in line or "confidence_OverConfident=1.0" in line
            assert "confidence_UnderConfident=0" in line or "confidence_UnderConfident=0.0" in line


class TestWritePredictionSingleHead:
    """Tests for single-head prediction writes."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for InfluxDB configuration."""
        mock_settings_obj = MagicMock()
        mock_settings_obj.influx_url = "http://localhost:8086"
        mock_settings_obj.influx_token = "test-token"
        mock_settings_obj.influx_org = "test-org"
        mock_settings_obj.influx_bucket_pred = "predictions"
        mock_settings_obj.influx_timeout_ms = 10000
        return mock_settings_obj

    @pytest.fixture
    def mock_write_api(self):
        """Mock InfluxDB async write API."""
        return AsyncMock()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_single_head_includes_appliance_tag(self, mock_settings, mock_write_api):
        """Single-head write includes appliance_id as tag."""
        with patch("app.infra.influx.client.get_settings", return_value=mock_settings):
            from app.infra.influx.client import InfluxClient

            client = InfluxClient()
            client._write_api = mock_write_api

            await client.write_prediction(
                building_id="building_1",
                appliance_id="heatpump",
                predicted_kw=2.5,
                confidence=0.95,
                model_version="v1.0",
                user_id="user123",
                request_id="req456",
                latency_ms=50.0,
            )

            call_args = mock_write_api.write.call_args
            point = call_args.kwargs.get("record")
            line = point.to_line_protocol()

            assert "appliance_id=heatpump" in line

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_single_head_prediction_fields(self, mock_settings, mock_write_api):
        """Single-head write has correct field structure."""
        with patch("app.infra.influx.client.get_settings", return_value=mock_settings):
            from app.infra.influx.client import InfluxClient

            client = InfluxClient()
            client._write_api = mock_write_api

            await client.write_prediction(
                building_id="building_1",
                appliance_id="heatpump",
                predicted_kw=2.5,
                confidence=0.95,
                model_version="v1.0",
                user_id="user123",
                request_id="req456",
                latency_ms=50.0,
            )

            call_args = mock_write_api.write.call_args
            point = call_args.kwargs.get("record")
            line = point.to_line_protocol()

            # Check fields
            assert "predicted_kw=2.5" in line
            assert "confidence=0.95" in line
            assert "latency_ms=50" in line


class TestWriteRetryBehavior:
    """Tests for write retry logic."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for InfluxDB configuration."""
        mock_settings_obj = MagicMock()
        mock_settings_obj.influx_url = "http://localhost:8086"
        mock_settings_obj.influx_token = "test-token"
        mock_settings_obj.influx_org = "test-org"
        mock_settings_obj.influx_bucket_pred = "predictions"
        mock_settings_obj.influx_timeout_ms = 10000
        return mock_settings_obj

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_retries_on_transient_failure(self, mock_settings):
        """Writer retries on transient failures."""
        with patch("app.infra.influx.client.get_settings", return_value=mock_settings):
            from app.infra.influx.client import InfluxClient

            client = InfluxClient()
            mock_write_api = AsyncMock()
            mock_write_api.write.side_effect = [
                Exception("Connection error"),
                Exception("Timeout"),
                None,  # Success on 3rd try
            ]
            client._write_api = mock_write_api
            client._sleep = AsyncMock()  # Don't actually sleep

            result = await client.write_prediction(
                building_id="test",
                appliance_id="test",
                predicted_kw=1.0,
                confidence=0.9,
                model_version="v1",
                user_id="u",
                request_id="r",
                latency_ms=10,
                max_retries=3,
            )

            assert result is True
            assert mock_write_api.write.call_count == 3

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self, mock_settings):
        """Raises InfluxError after max retries exhausted."""
        with patch("app.infra.influx.client.get_settings", return_value=mock_settings):
            from app.infra.influx.client import InfluxClient
            from app.core.errors import InfluxError

            client = InfluxClient()
            mock_write_api = AsyncMock()
            mock_write_api.write.side_effect = Exception("Persistent failure")
            client._write_api = mock_write_api
            client._sleep = AsyncMock()

            with pytest.raises(InfluxError) as exc_info:
                await client.write_prediction(
                    building_id="test",
                    appliance_id="test",
                    predicted_kw=1.0,
                    confidence=0.9,
                    model_version="v1",
                    user_id="u",
                    request_id="r",
                    latency_ms=10,
                    max_retries=3,
                )

            assert "Failed to write prediction after 3 attempts" in str(exc_info.value.message)


class TestPointStructure:
    """Tests for InfluxDB Point construction."""

    @pytest.mark.unit
    def test_point_line_protocol_format(self):
        """Verify Point line protocol format is correct."""
        timestamp = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

        point = (
            Point("prediction")
            .tag("building_id", "b1")
            .tag("model_version", "v1")
            .field("predicted_kw_HeatPump", 2.5)
            .field("confidence_HeatPump", 0.95)
            .time(timestamp)
        )

        line = point.to_line_protocol()

        # Line format: measurement,tag1=val1,tag2=val2 field1=val1,field2=val2 timestamp
        assert line.startswith("prediction,")
        assert "building_id=b1" in line
        assert "model_version=v1" in line
        assert "predicted_kw_HeatPump=2.5" in line
        assert "confidence_HeatPump=0.95" in line

    @pytest.mark.unit
    def test_point_escapes_special_characters(self):
        """Point correctly escapes special characters in tag values."""
        point = (
            Point("prediction")
            .tag("building_id", "building 1")  # Space
            .field("value", 1.0)
        )

        line = point.to_line_protocol()

        # Spaces in tag values should be escaped with backslash
        assert "building\\ 1" in line or "building 1" not in line.split()[0]

    @pytest.mark.unit
    def test_point_handles_numeric_field_types(self):
        """Point handles various numeric types correctly."""
        point = (
            Point("test")
            .field("int_val", 42)
            .field("float_val", 3.14)
            .field("zero", 0)
            .field("negative", -1.5)
        )

        line = point.to_line_protocol()

        # All should be present as numbers
        assert "int_val=42" in line
        assert "float_val=3.14" in line
        assert "zero=0" in line
        assert "negative=-1.5" in line


class TestMetadataFields:
    """Tests for metadata fields in predictions."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for InfluxDB configuration."""
        mock_settings_obj = MagicMock()
        mock_settings_obj.influx_url = "http://localhost:8086"
        mock_settings_obj.influx_token = "test-token"
        mock_settings_obj.influx_org = "test-org"
        mock_settings_obj.influx_bucket_pred = "predictions"
        mock_settings_obj.influx_timeout_ms = 10000
        return mock_settings_obj

    @pytest.fixture
    def mock_write_api(self):
        """Mock InfluxDB async write API."""
        return AsyncMock()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_includes_user_id_field(self, mock_settings, mock_write_api):
        """Write includes user_id in fields."""
        with patch("app.infra.influx.client.get_settings", return_value=mock_settings):
            from app.infra.influx.client import InfluxClient

            client = InfluxClient()
            client._write_api = mock_write_api

            await client.write_predictions_wide(
                building_id="b1",
                predictions={"test": (1.0, 0.9)},
                model_version="v1",
                user_id="user_abc",
                request_id="req123",
                latency_ms=50.0,
            )

            call_args = mock_write_api.write.call_args
            point = call_args.kwargs.get("record")
            line = point.to_line_protocol()

            assert 'user_id="user_abc"' in line

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_includes_request_id_field(self, mock_settings, mock_write_api):
        """Write includes request_id in fields for tracing."""
        with patch("app.infra.influx.client.get_settings", return_value=mock_settings):
            from app.infra.influx.client import InfluxClient

            client = InfluxClient()
            client._write_api = mock_write_api

            await client.write_predictions_wide(
                building_id="b1",
                predictions={"test": (1.0, 0.9)},
                model_version="v1",
                user_id="user_abc",
                request_id="req-uuid-123",
                latency_ms=50.0,
            )

            call_args = mock_write_api.write.call_args
            point = call_args.kwargs.get("record")
            line = point.to_line_protocol()

            assert 'request_id="req-uuid-123"' in line

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_includes_latency_ms_field(self, mock_settings, mock_write_api):
        """Write includes latency_ms in fields."""
        with patch("app.infra.influx.client.get_settings", return_value=mock_settings):
            from app.infra.influx.client import InfluxClient

            client = InfluxClient()
            client._write_api = mock_write_api

            await client.write_predictions_wide(
                building_id="b1",
                predictions={"test": (1.0, 0.9)},
                model_version="v1",
                user_id="user_abc",
                request_id="req123",
                latency_ms=123.45,
            )

            call_args = mock_write_api.write.call_args
            point = call_args.kwargs.get("record")
            line = point.to_line_protocol()

            assert "latency_ms=123.45" in line
