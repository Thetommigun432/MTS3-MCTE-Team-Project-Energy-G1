"""
Component tests for analytics API endpoints.

Tests GET /analytics/* endpoints with mocked dependencies.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock


@pytest.fixture
def mock_user():
    """Mock authenticated user token payload."""
    user = MagicMock()
    user.user_id = "test_user"
    user.sub = "test_user"
    user.aud = "authenticated"
    user.email = "test@example.com"
    return user


@pytest.fixture
def mock_influx_client():
    """Mock InfluxDB client with sample data."""
    client = AsyncMock()

    # Mock readings query
    from app.schemas.analytics import DataPoint
    client.query_readings.return_value = [
        DataPoint(time="2024-01-15T12:00:00Z", value=2.5),
        DataPoint(time="2024-01-15T12:01:00Z", value=2.6),
        DataPoint(time="2024-01-15T12:02:00Z", value=2.4),
    ]

    # Mock predictions query
    from app.schemas.analytics import PredictionPoint
    client.query_predictions.return_value = [
        PredictionPoint(
            time="2024-01-15T12:00:00Z",
            predicted_kw=0.8,
            confidence=0.95,
            model_version="v1.0",
        ),
        PredictionPoint(
            time="2024-01-15T12:01:00Z",
            predicted_kw=0.9,
            confidence=0.92,
            model_version="v1.0",
        ),
    ]

    # Mock wide predictions (for disaggregation)
    client.query_predictions_wide.return_value = [
        {
            "time": "2024-01-15T12:00:00Z",
            "predicted_kw_HeatPump": 1.5,
            "predicted_kw_Dishwasher": 0.8,
            "confidence_HeatPump": 0.95,
            "confidence_Dishwasher": 0.88,
        }
    ]

    # Mock buildings list
    client.get_unique_buildings.return_value = ["building_1", "building_2", "building_3"]

    # Mock appliances list
    client.get_unique_appliances.return_value = ["HeatPump", "Dishwasher", "WashingMachine"]

    return client


@pytest.fixture
def mock_building_access():
    """Mock building access authorization."""
    async def _mock_require_access(user, building_id):
        # Always allow access in tests
        return None

    return _mock_require_access


@pytest.fixture
def test_client(mock_user, mock_influx_client, mock_building_access):
    """Create test client with mocked dependencies."""
    with patch("app.api.deps.get_current_user", return_value=mock_user), \
         patch("app.api.routers.analytics.get_influx_client", return_value=mock_influx_client), \
         patch("app.api.routers.analytics.require_building_access", mock_building_access):
        from app.main import app
        with TestClient(app) as client:
            yield client


class TestReadingsEndpoint:
    """Tests for GET /analytics/readings."""

    @pytest.mark.component
    def test_readings_returns_200(self, test_client):
        """GET /analytics/readings returns 200 with valid params."""
        response = test_client.get(
            "/analytics/readings",
            params={
                "building_id": "building_1",
                "start": "-7d",
                "end": "now()",
            },
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == 200

    @pytest.mark.component
    def test_readings_response_shape(self, test_client):
        """Response has correct schema."""
        response = test_client.get(
            "/analytics/readings",
            params={
                "building_id": "building_1",
                "start": "-7d",
                "end": "now()",
            },
            headers={"Authorization": "Bearer test_token"}
        )

        data = response.json()

        assert "building_id" in data
        assert "data" in data
        assert "count" in data
        assert data["building_id"] == "building_1"
        assert isinstance(data["data"], list)

    @pytest.mark.component
    def test_readings_data_points_structure(self, test_client):
        """Data points have correct structure."""
        response = test_client.get(
            "/analytics/readings",
            params={
                "building_id": "building_1",
                "start": "-7d",
                "end": "now()",
            },
            headers={"Authorization": "Bearer test_token"}
        )

        data = response.json()

        assert data["count"] == 3
        assert len(data["data"]) == 3

        point = data["data"][0]
        assert "time" in point
        assert "value" in point
        assert point["time"] == "2024-01-15T12:00:00Z"
        assert point["value"] == 2.5

    @pytest.mark.component
    def test_readings_requires_building_id(self, test_client):
        """Request without building_id returns 422."""
        response = test_client.get(
            "/analytics/readings",
            params={
                "start": "-7d",
                "end": "now()",
            },
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == 422

    @pytest.mark.component
    def test_readings_requires_start(self, test_client):
        """Request without start returns 422."""
        response = test_client.get(
            "/analytics/readings",
            params={
                "building_id": "building_1",
                "end": "now()",
            },
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == 422


class TestPredictionsEndpoint:
    """Tests for GET /analytics/predictions."""

    @pytest.mark.component
    def test_predictions_returns_200(self, test_client):
        """GET /analytics/predictions returns 200 with valid params."""
        response = test_client.get(
            "/analytics/predictions",
            params={
                "building_id": "building_1",
                "start": "-7d",
                "end": "now()",
            },
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == 200

    @pytest.mark.component
    def test_predictions_response_shape(self, test_client):
        """Response has correct schema."""
        response = test_client.get(
            "/analytics/predictions",
            params={
                "building_id": "building_1",
                "start": "-7d",
                "end": "now()",
            },
            headers={"Authorization": "Bearer test_token"}
        )

        data = response.json()

        assert "building_id" in data
        assert "data" in data
        assert "count" in data

    @pytest.mark.component
    def test_predictions_data_points_structure(self, test_client):
        """Prediction points have correct structure."""
        response = test_client.get(
            "/analytics/predictions",
            params={
                "building_id": "building_1",
                "start": "-7d",
                "end": "now()",
            },
            headers={"Authorization": "Bearer test_token"}
        )

        data = response.json()

        assert data["count"] == 2
        point = data["data"][0]

        assert "time" in point
        assert "predicted_kw" in point
        assert "confidence" in point
        assert point["predicted_kw"] == 0.8
        assert point["confidence"] == 0.95


class TestBuildingsEndpoint:
    """Tests for GET /analytics/buildings."""

    @pytest.mark.component
    def test_buildings_returns_200(self, test_client):
        """GET /analytics/buildings returns 200."""
        response = test_client.get(
            "/analytics/buildings",
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == 200

    @pytest.mark.component
    def test_buildings_response_shape(self, test_client):
        """Response has correct schema."""
        response = test_client.get(
            "/analytics/buildings",
            headers={"Authorization": "Bearer test_token"}
        )

        data = response.json()

        assert "buildings" in data
        assert isinstance(data["buildings"], list)

    @pytest.mark.component
    def test_buildings_returns_list(self, test_client):
        """Returns list of building IDs."""
        response = test_client.get(
            "/analytics/buildings",
            headers={"Authorization": "Bearer test_token"}
        )

        data = response.json()

        assert data["buildings"] == ["building_1", "building_2", "building_3"]


class TestAppliancesEndpoint:
    """Tests for GET /analytics/appliances."""

    @pytest.mark.component
    def test_appliances_returns_200(self, test_client):
        """GET /analytics/appliances returns 200."""
        response = test_client.get(
            "/analytics/appliances",
            params={"building_id": "building_1"},
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == 200

    @pytest.mark.component
    def test_appliances_response_shape(self, test_client):
        """Response has correct schema."""
        response = test_client.get(
            "/analytics/appliances",
            params={"building_id": "building_1"},
            headers={"Authorization": "Bearer test_token"}
        )

        data = response.json()

        assert "appliances" in data
        assert isinstance(data["appliances"], list)

    @pytest.mark.component
    def test_appliances_returns_list(self, test_client):
        """Returns list of appliance IDs for building."""
        response = test_client.get(
            "/analytics/appliances",
            params={"building_id": "building_1"},
            headers={"Authorization": "Bearer test_token"}
        )

        data = response.json()

        assert data["appliances"] == ["HeatPump", "Dishwasher", "WashingMachine"]

    @pytest.mark.component
    def test_appliances_requires_building_id(self, test_client):
        """Request without building_id returns 422."""
        response = test_client.get(
            "/analytics/appliances",
            headers={"Authorization": "Bearer test_token"}
        )

        assert response.status_code == 422


class TestAuthenticationRequired:
    """Tests that endpoints require authentication."""

    @pytest.fixture
    def unauthenticated_client(self, mock_influx_client, mock_building_access):
        """Create test client without mocked auth."""
        with patch("app.api.routers.analytics.get_influx_client", return_value=mock_influx_client), \
             patch("app.api.routers.analytics.require_building_access", mock_building_access):
            from app.main import app
            with TestClient(app) as client:
                yield client

    @pytest.mark.component
    def test_readings_requires_auth(self, unauthenticated_client):
        """GET /analytics/readings returns 401 without auth."""
        response = unauthenticated_client.get(
            "/analytics/readings",
            params={
                "building_id": "building_1",
                "start": "-7d",
                "end": "now()",
            }
        )

        assert response.status_code == 401

    @pytest.mark.component
    def test_predictions_requires_auth(self, unauthenticated_client):
        """GET /analytics/predictions returns 401 without auth."""
        response = unauthenticated_client.get(
            "/analytics/predictions",
            params={
                "building_id": "building_1",
                "start": "-7d",
                "end": "now()",
            }
        )

        assert response.status_code == 401

    @pytest.mark.component
    def test_buildings_requires_auth(self, unauthenticated_client):
        """GET /analytics/buildings returns 401 without auth."""
        response = unauthenticated_client.get("/analytics/buildings")

        assert response.status_code == 401


class TestResolutionParameter:
    """Tests for resolution query parameter."""

    @pytest.mark.component
    def test_default_resolution_is_1m(self, test_client, mock_influx_client):
        """Default resolution is 1 minute."""
        test_client.get(
            "/analytics/readings",
            params={
                "building_id": "building_1",
                "start": "-7d",
                "end": "now()",
            },
            headers={"Authorization": "Bearer test_token"}
        )

        # Check that influx was called with default resolution
        mock_influx_client.query_readings.assert_called_once()
        call_kwargs = mock_influx_client.query_readings.call_args.kwargs
        assert call_kwargs["resolution"].value == "1m"

    @pytest.mark.component
    def test_custom_resolution(self, test_client, mock_influx_client):
        """Custom resolution parameter is passed to query."""
        test_client.get(
            "/analytics/readings",
            params={
                "building_id": "building_1",
                "start": "-7d",
                "end": "now()",
                "resolution": "15m",
            },
            headers={"Authorization": "Bearer test_token"}
        )

        mock_influx_client.query_readings.assert_called_once()
        call_kwargs = mock_influx_client.query_readings.call_args.kwargs
        assert call_kwargs["resolution"].value == "15m"
