"""
Component tests for POST /ingest/readings endpoint.

Tests batch ingestion with mocked InfluxDB and Redis dependencies.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock
from datetime import datetime, timezone


@pytest.fixture
def mock_influx_client():
    """Mock InfluxDB client."""
    client = MagicMock()
    client.write_point = AsyncMock(return_value=True)
    return client


@pytest.fixture
def mock_redis_xadd():
    """Mock Redis XADD for stream writes."""
    return AsyncMock(return_value="1234567890-0")


@pytest.fixture
def mock_settings_no_token():
    """Mock settings without ingestion token (open access)."""
    settings = MagicMock()
    settings.ingest_token = None
    settings.redis_stream_key = "nilm:readings"
    settings.pipeline_enabled = True
    return settings


@pytest.fixture
def mock_settings_with_token():
    """Mock settings with ingestion token."""
    settings = MagicMock()
    settings.ingest_token = "secret-ingest-token"
    settings.redis_stream_key = "nilm:readings"
    settings.pipeline_enabled = True
    return settings


@pytest.fixture
def test_client(mock_influx_client, mock_redis_xadd, mock_settings_no_token):
    """Create test client with mocked dependencies."""
    with patch("app.api.routers.ingest.get_influx_client", return_value=mock_influx_client), \
         patch("app.api.routers.ingest.xadd_reading", mock_redis_xadd), \
         patch("app.api.routers.ingest.get_settings", return_value=mock_settings_no_token):
        from app.main import app
        with TestClient(app) as client:
            yield client


class TestIngestReadingsEndpoint:
    """Tests for POST /ingest/readings."""

    @pytest.mark.component
    def test_ingest_single_reading_returns_202(self, test_client):
        """Single reading is ingested and returns 202 ACCEPTED."""
        response = test_client.post(
            "/api/ingest/readings",
            json={
                "readings": [
                    {
                        "building_id": "building_1",
                        "ts": "2024-01-15T12:00:00Z",
                        "aggregate_kw": 2.5,
                    }
                ]
            }
        )

        assert response.status_code == 202

    @pytest.mark.component
    def test_ingest_single_reading_response_shape(self, test_client):
        """Response has correct schema."""
        response = test_client.post(
            "/api/ingest/readings",
            json={
                "readings": [
                    {
                        "building_id": "building_1",
                        "ts": "2024-01-15T12:00:00Z",
                        "aggregate_kw": 2.5,
                    }
                ]
            }
        )

        data = response.json()

        assert "ingested" in data
        assert "errors" in data
        assert data["ingested"] == 1
        assert data["errors"] == 0

    @pytest.mark.component
    def test_ingest_batch_readings(self, test_client):
        """Batch of readings is ingested."""
        readings = [
            {
                "building_id": "building_1",
                "ts": f"2024-01-15T12:0{i}:00Z",
                "aggregate_kw": 2.0 + i * 0.1
            }
            for i in range(5)
        ]

        response = test_client.post(
            "/api/ingest/readings",
            json={"readings": readings}
        )

        assert response.status_code == 202
        data = response.json()
        assert data["ingested"] == 5
        assert data["errors"] == 0

    @pytest.mark.component
    def test_ingest_calls_influx_write(self, mock_influx_client, mock_redis_xadd, mock_settings_no_token):
        """Ingestion writes to InfluxDB."""
        with patch("app.api.routers.ingest.get_influx_client", return_value=mock_influx_client), \
             patch("app.api.routers.ingest.xadd_reading", mock_redis_xadd), \
             patch("app.api.routers.ingest.get_settings", return_value=mock_settings_no_token):
            from app.main import app
            with TestClient(app) as client:
                client.post(
                    "/api/ingest/readings",
                    json={
                        "readings": [
                            {
                                "building_id": "building_1",
                                "ts": "2024-01-15T12:00:00Z",
                                "aggregate_kw": 2.5,
                            }
                        ]
                    }
                )

                # Verify Influx write was called
                mock_influx_client.write_point.assert_called_once()

    @pytest.mark.component
    def test_ingest_calls_redis_stream(self, mock_influx_client, mock_redis_xadd, mock_settings_no_token):
        """Ingestion publishes to Redis stream."""
        with patch("app.api.routers.ingest.get_influx_client", return_value=mock_influx_client), \
             patch("app.api.routers.ingest.xadd_reading", mock_redis_xadd), \
             patch("app.api.routers.ingest.get_settings", return_value=mock_settings_no_token):
            from app.main import app
            with TestClient(app) as client:
                client.post(
                    "/api/ingest/readings",
                    json={
                        "readings": [
                            {
                                "building_id": "building_1",
                                "ts": "2024-01-15T12:00:00Z",
                                "aggregate_kw": 2.5,
                            }
                        ]
                    }
                )

                # Verify Redis XADD was called
                mock_redis_xadd.assert_called_once()


class TestIngestPartialFailure:
    """Tests for partial failure handling."""

    @pytest.mark.component
    def test_partial_failure_reports_correctly(self, mock_redis_xadd, mock_settings_no_token):
        """Partial failures are reported in response."""
        mock_influx = MagicMock()
        # First two succeed, third fails
        mock_influx.write_point = AsyncMock(
            side_effect=[None, None, Exception("Write failed")]
        )

        with patch("app.api.routers.ingest.get_influx_client", return_value=mock_influx), \
             patch("app.api.routers.ingest.xadd_reading", mock_redis_xadd), \
             patch("app.api.routers.ingest.get_settings", return_value=mock_settings_no_token):
            from app.main import app
            with TestClient(app) as client:
                readings = [
                    {"building_id": "b1", "ts": f"2024-01-15T12:0{i}:00Z", "aggregate_kw": 2.0}
                    for i in range(3)
                ]

                response = client.post("/api/ingest/readings", json={"readings": readings})

                assert response.status_code == 202
                data = response.json()
                assert data["ingested"] == 2
                assert data["errors"] == 1

    @pytest.mark.component
    def test_all_failures_returns_500(self, mock_redis_xadd, mock_settings_no_token):
        """All failures returns 500."""
        mock_influx = MagicMock()
        mock_influx.write_point = AsyncMock(side_effect=Exception("Write failed"))

        with patch("app.api.routers.ingest.get_influx_client", return_value=mock_influx), \
             patch("app.api.routers.ingest.xadd_reading", mock_redis_xadd), \
             patch("app.api.routers.ingest.get_settings", return_value=mock_settings_no_token):
            from app.main import app
            with TestClient(app) as client:
                response = client.post(
                    "/api/ingest/readings",
                    json={
                        "readings": [
                            {"building_id": "b1", "ts": "2024-01-15T12:00:00Z", "aggregate_kw": 2.0}
                        ]
                    }
                )

                assert response.status_code == 500

    @pytest.mark.component
    def test_redis_failure_does_not_fail_request(self, mock_influx_client, mock_settings_no_token):
        """Redis failure doesn't fail the request if Influx succeeded."""
        mock_redis_fail = AsyncMock(side_effect=Exception("Redis unavailable"))

        with patch("app.api.routers.ingest.get_influx_client", return_value=mock_influx_client), \
             patch("app.api.routers.ingest.xadd_reading", mock_redis_fail), \
             patch("app.api.routers.ingest.get_settings", return_value=mock_settings_no_token):
            from app.main import app
            with TestClient(app) as client:
                response = client.post(
                    "/api/ingest/readings",
                    json={
                        "readings": [
                            {"building_id": "b1", "ts": "2024-01-15T12:00:00Z", "aggregate_kw": 2.0}
                        ]
                    }
                )

                # Should still succeed since Influx write worked
                assert response.status_code == 202
                data = response.json()
                assert data["ingested"] == 1


class TestIngestValidation:
    """Tests for request validation."""

    @pytest.mark.component
    def test_empty_readings_array_accepted(self, test_client):
        """Empty readings array is handled gracefully."""
        response = test_client.post(
            "/api/ingest/readings",
            json={"readings": []}
        )

        # Empty batch should not error
        assert response.status_code == 202
        data = response.json()
        assert data["ingested"] == 0
        assert data["errors"] == 0

    @pytest.mark.component
    def test_missing_readings_field_returns_422(self, test_client):
        """Missing readings field returns validation error."""
        response = test_client.post(
            "/api/ingest/readings",
            json={}
        )

        assert response.status_code == 422

    @pytest.mark.component
    def test_invalid_building_id_returns_422(self, test_client):
        """Invalid building_id format returns validation error."""
        response = test_client.post(
            "/api/ingest/readings",
            json={
                "readings": [
                    {
                        "building_id": "invalid building!@#",  # Contains invalid chars
                        "ts": "2024-01-15T12:00:00Z",
                        "aggregate_kw": 2.0,
                    }
                ]
            }
        )

        assert response.status_code == 422

    @pytest.mark.component
    def test_missing_timestamp_returns_422(self, test_client):
        """Missing timestamp returns validation error."""
        response = test_client.post(
            "/api/ingest/readings",
            json={
                "readings": [
                    {
                        "building_id": "building_1",
                        "aggregate_kw": 2.0,
                    }
                ]
            }
        )

        assert response.status_code == 422


class TestIngestToken:
    """Tests for ingestion token authentication."""

    @pytest.mark.component
    def test_requires_token_when_configured(self, mock_influx_client, mock_redis_xadd, mock_settings_with_token):
        """Token is required when configured in settings."""
        with patch("app.api.routers.ingest.get_influx_client", return_value=mock_influx_client), \
             patch("app.api.routers.ingest.xadd_reading", mock_redis_xadd), \
             patch("app.api.routers.ingest.get_settings", return_value=mock_settings_with_token):
            from app.main import app
            with TestClient(app) as client:
                # Without token
                response = client.post(
                    "/api/ingest/readings",
                    json={
                        "readings": [
                            {"building_id": "b1", "ts": "2024-01-15T12:00:00Z", "aggregate_kw": 2.0}
                        ]
                    }
                )

                assert response.status_code == 401

    @pytest.mark.component
    def test_accepts_correct_token(self, mock_influx_client, mock_redis_xadd, mock_settings_with_token):
        """Correct token is accepted."""
        with patch("app.api.routers.ingest.get_influx_client", return_value=mock_influx_client), \
             patch("app.api.routers.ingest.xadd_reading", mock_redis_xadd), \
             patch("app.api.routers.ingest.get_settings", return_value=mock_settings_with_token):
            from app.main import app
            with TestClient(app) as client:
                response = client.post(
                    "/api/ingest/readings",
                    json={
                        "readings": [
                            {"building_id": "b1", "ts": "2024-01-15T12:00:00Z", "aggregate_kw": 2.0}
                        ]
                    },
                    headers={"X-Ingest-Token": "secret-ingest-token"}
                )

                assert response.status_code == 202

    @pytest.mark.component
    def test_rejects_wrong_token(self, mock_influx_client, mock_redis_xadd, mock_settings_with_token):
        """Wrong token is rejected."""
        with patch("app.api.routers.ingest.get_influx_client", return_value=mock_influx_client), \
             patch("app.api.routers.ingest.xadd_reading", mock_redis_xadd), \
             patch("app.api.routers.ingest.get_settings", return_value=mock_settings_with_token):
            from app.main import app
            with TestClient(app) as client:
                response = client.post(
                    "/api/ingest/readings",
                    json={
                        "readings": [
                            {"building_id": "b1", "ts": "2024-01-15T12:00:00Z", "aggregate_kw": 2.0}
                        ]
                    },
                    headers={"X-Ingest-Token": "wrong-token"}
                )

                assert response.status_code == 401
