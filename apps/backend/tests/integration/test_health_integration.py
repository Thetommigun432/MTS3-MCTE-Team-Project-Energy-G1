
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch, AsyncMock
from app.main import app

# Create a test client
client = TestClient(app)

def test_health_endpoint():
    """Verify standard liveness probe."""
    response = client.get("/live")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "request_id" in data

@patch("app.api.routers.health.get_model_registry")
@patch("app.api.routers.health.get_redis_cache")
@patch("app.api.routers.health.get_influx_client") 
def test_ready_endpoint_mocked(mock_get_influx, mock_get_redis, mock_get_registry):
    """
    Verify readiness probe checks dependencies.
    We mock the clients to simulate success without needing real DBs running.
    """
    # Mock Registry
    mock_registry = MagicMock()
    mock_registry.is_loaded = True
    mock_registry.list_all.return_value = ["model1"]
    mock_get_registry.return_value = mock_registry

    # Mock Redis ping
    mock_redis_instance = MagicMock()
    mock_redis_instance.is_using_fallback = False
    mock_get_redis.return_value = mock_redis_instance

    # Mock Influx ready
    mock_influx_instance = AsyncMock()
    mock_influx_instance.verify_setup.return_value = {
        "connected": True,
        "bucket_raw": True,
        "bucket_pred": True
    }
    mock_get_influx.return_value = mock_influx_instance

    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "checks" in data
    assert data["checks"]["influxdb_connected"] is True
    assert data["checks"]["redis_available"] is True
    assert data["checks"]["registry_loaded"] is True

@patch("app.api.routers.health.get_model_registry")
@patch("app.api.routers.health.get_influx_client")
@patch("app.api.routers.health.get_redis_cache")
def test_ready_endpoint_redis_failure(mock_get_redis, mock_get_influx, mock_get_registry):
    """Verify degradation if Redis fails."""
    # Mock Registry
    mock_registry = MagicMock()
    mock_registry.is_loaded = True
    mock_registry.list_all.return_value = ["model1"]
    mock_get_registry.return_value = mock_registry

    # Mock Influx ready (must stay healthy for 200 OK)
    mock_influx_instance = AsyncMock()
    mock_influx_instance.verify_setup.return_value = {
        "connected": True,
        "bucket_raw": True,
        "bucket_pred": True
    }
    mock_get_influx.return_value = mock_influx_instance

    # Mock Redis failure (fallback mode)
    mock_redis_instance = MagicMock()
    mock_redis_instance.is_using_fallback = True
    mock_get_redis.return_value = mock_redis_instance
    
    # We allow /ready to pass even if Redis is fallback (graceful degradation)
    response = client.get("/ready")
    
    assert response.status_code == 200
    data = response.json()
    assert data["checks"]["redis_available"] is False
