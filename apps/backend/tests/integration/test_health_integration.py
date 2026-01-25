
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
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

@patch("app.api.routers.health.get_redis_cache")
@patch("app.api.routers.health.get_influx_client") 
def test_ready_endpoint_mocked(mock_get_influx, mock_get_redis):
    """
    Verify readiness probe checks dependencies.
    We mock the clients to simulate success without needing real DBs running.
    """
    # Mock Redis ping
    mock_redis_instance = MagicMock()
    mock_redis_instance.is_using_fallback = False
    mock_get_redis.return_value = mock_redis_instance

    # Mock Influx ready
    mock_influx_instance = AsyncMock()
    # verify_setup returns a dict map of status
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

@patch("app.api.routers.health.get_redis_cache")
def test_ready_endpoint_redis_failure(mock_get_redis):
    """Verify degradation if Redis fails."""
    # Mock Redis failure (fallback mode)
    mock_redis_instance = MagicMock()
    mock_redis_instance.is_using_fallback = True
    mock_get_redis.return_value = mock_redis_instance
    
    # We allow /ready to pass even if Redis is fallback (graceful degradation)
    response = client.get("/ready")
    # Should be 200 OK but check indicates fallback?
    # Actually ready endpoint checks redis_available = not is_using_fallback
    # But does not fail readiness if redis is unavailable (lines 83-84 in health.py)
    
    assert response.status_code == 200
    data = response.json()
    assert data["checks"]["redis_available"] is False
