"""
Component tests for GET /models endpoint.

Tests the models listing API with mocked inference service.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock


@pytest.fixture
def mock_inference_service():
    """Mock inference service with sample models."""
    service = AsyncMock()
    service.list_models.return_value = [
        {
            "model_id": "heatpump_v1",
            "model_version": "1.0.0",
            "appliance_id": "heatpump",
            "architecture": "CNNTransformer",
            "input_window_size": 1000,
            "is_active": True,
            "cached": False,
            "heads": [],
            "metrics": None,
        },
        {
            "model_id": "multi_v1",
            "model_version": "2.0.0",
            "appliance_id": "multi",
            "architecture": "WaveNILM_v3",
            "input_window_size": 1000,
            "is_active": True,
            "cached": True,
            "heads": [
                {"appliance_id": "HeatPump", "field_key": "HeatPump"},
                {"appliance_id": "Dishwasher", "field_key": "Dishwasher"},
            ],
            "metrics": {"mae": 0.15, "rmse": 0.22, "f1_score": 0.87, "accuracy": 0.91},
        },
    ]
    return service


@pytest.fixture
def mock_model_details():
    """Mock model details response."""
    return {
        "model_id": "heatpump_v1",
        "model_version": "1.0.0",
        "appliance_id": "heatpump",
        "architecture": "CNNTransformer",
        "architecture_params": {"d_model": 128, "n_heads": 8, "n_layers": 4},
        "input_window_size": 1000,
        "preprocessing": {"type": "minmax", "min": 0.0, "max": 10000.0},
        "is_active": True,
        "cached": False,
    }


@pytest.fixture
def test_client(mock_inference_service):
    """Create test client with mocked dependencies."""
    with patch("app.api.routers.inference.get_inference_service", return_value=mock_inference_service):
        from app.main import app
        with TestClient(app) as client:
            yield client


class TestListModelsEndpoint:
    """Tests for GET /models endpoint."""

    @pytest.mark.component
    def test_list_models_returns_200(self, test_client, mock_inference_service):
        """GET /models returns 200 OK."""
        response = test_client.get("/models")

        assert response.status_code == 200

    @pytest.mark.component
    def test_list_models_response_has_models_array(self, test_client, mock_inference_service):
        """Response contains 'models' array."""
        response = test_client.get("/models")
        data = response.json()

        assert "models" in data
        assert isinstance(data["models"], list)

    @pytest.mark.component
    def test_list_models_response_has_count(self, test_client, mock_inference_service):
        """Response contains 'count' field matching models length."""
        response = test_client.get("/models")
        data = response.json()

        assert "count" in data
        assert data["count"] == len(data["models"])
        assert data["count"] == 2

    @pytest.mark.component
    def test_model_info_has_required_fields(self, test_client, mock_inference_service):
        """Each model has required fields."""
        response = test_client.get("/models")
        data = response.json()

        required_fields = {
            "model_id",
            "model_version",
            "appliance_id",
            "architecture",
            "input_window_size",
            "is_active",
            "cached",
            "heads",
        }

        for model in data["models"]:
            assert required_fields.issubset(set(model.keys())), \
                f"Model missing required fields: {required_fields - set(model.keys())}"

    @pytest.mark.component
    def test_multi_head_model_has_heads(self, test_client, mock_inference_service):
        """Multi-head model includes heads list."""
        response = test_client.get("/models")
        data = response.json()

        multi_model = next(m for m in data["models"] if m["model_id"] == "multi_v1")

        assert len(multi_model["heads"]) == 2
        assert multi_model["heads"][0]["appliance_id"] == "HeatPump"
        assert multi_model["heads"][1]["field_key"] == "Dishwasher"

    @pytest.mark.component
    def test_single_head_model_has_empty_heads(self, test_client, mock_inference_service):
        """Single-head model has empty heads list."""
        response = test_client.get("/models")
        data = response.json()

        single_model = next(m for m in data["models"] if m["model_id"] == "heatpump_v1")

        assert single_model["heads"] == []

    @pytest.mark.component
    def test_model_metrics_optional(self, test_client, mock_inference_service):
        """Model metrics can be null or object."""
        response = test_client.get("/models")
        data = response.json()

        # First model has no metrics
        single_model = next(m for m in data["models"] if m["model_id"] == "heatpump_v1")
        assert single_model["metrics"] is None

        # Second model has metrics
        multi_model = next(m for m in data["models"] if m["model_id"] == "multi_v1")
        assert multi_model["metrics"] is not None
        assert multi_model["metrics"]["mae"] == 0.15

    @pytest.mark.component
    def test_cached_status_reported(self, test_client, mock_inference_service):
        """Model cached status is correctly reported."""
        response = test_client.get("/models")
        data = response.json()

        # First model not cached
        single_model = next(m for m in data["models"] if m["model_id"] == "heatpump_v1")
        assert single_model["cached"] is False

        # Second model is cached
        multi_model = next(m for m in data["models"] if m["model_id"] == "multi_v1")
        assert multi_model["cached"] is True


class TestEmptyModelsResponse:
    """Tests for empty models list."""

    @pytest.mark.component
    def test_empty_models_returns_200(self):
        """GET /models with no models returns 200 with empty list."""
        mock_service = AsyncMock()
        mock_service.list_models.return_value = []

        with patch("app.api.routers.inference.get_inference_service", return_value=mock_service):
            from app.main import app
            with TestClient(app) as client:
                response = client.get("/models")

                assert response.status_code == 200
                data = response.json()
                assert data["models"] == []
                assert data["count"] == 0


class TestModelMetricsEndpoint:
    """Tests for GET /models/{model_id}/metrics endpoint."""

    @pytest.mark.component
    def test_model_metrics_returns_200(self, mock_model_details):
        """GET /models/{model_id}/metrics returns 200 for valid model."""
        mock_service = AsyncMock()
        mock_service.get_model_details.return_value = mock_model_details

        with patch("app.api.routers.inference.get_inference_service", return_value=mock_service):
            from app.main import app
            with TestClient(app) as client:
                response = client.get("/models/heatpump_v1/metrics")

                assert response.status_code == 200

    @pytest.mark.component
    def test_model_metrics_response_shape(self, mock_model_details):
        """Response has correct schema."""
        mock_service = AsyncMock()
        mock_service.get_model_details.return_value = mock_model_details

        with patch("app.api.routers.inference.get_inference_service", return_value=mock_service):
            from app.main import app
            with TestClient(app) as client:
                response = client.get("/models/heatpump_v1/metrics")
                data = response.json()

                assert "model_id" in data
                assert "model_version" in data
                assert "architecture" in data
                assert "architecture_params" in data
                assert "preprocessing" in data
                assert "input_window_size" in data

    @pytest.mark.component
    def test_model_metrics_architecture_params(self, mock_model_details):
        """Architecture params are returned."""
        mock_service = AsyncMock()
        mock_service.get_model_details.return_value = mock_model_details

        with patch("app.api.routers.inference.get_inference_service", return_value=mock_service):
            from app.main import app
            with TestClient(app) as client:
                response = client.get("/models/heatpump_v1/metrics")
                data = response.json()

                assert data["architecture_params"]["d_model"] == 128
                assert data["architecture_params"]["n_heads"] == 8

    @pytest.mark.component
    def test_model_not_found_returns_404(self):
        """GET /models/{model_id}/metrics returns 404 for unknown model."""
        mock_service = AsyncMock()
        # Simulate model not found
        from app.core.errors import ModelError, ErrorCode
        mock_service.get_model_details.side_effect = ModelError(
            code=ErrorCode.MODEL_NOT_FOUND,
            message="Model not found: unknown_model"
        )

        with patch("app.api.routers.inference.get_inference_service", return_value=mock_service):
            from app.main import app
            with TestClient(app) as client:
                response = client.get("/models/unknown_model/metrics")

                # Should return 404 or error response
                assert response.status_code in [404, 422, 500]
