import pytest
import sys
from pathlib import Path
from unittest.mock import patch, AsyncMock

# Add backend app to python path for imports
backend_path = Path(__file__).parent.parent.parent
sys.path.append(str(backend_path))

@pytest.fixture(scope="session")
def integration_root():
    """Return the root of the integration tests."""
    return Path(__file__).parent

@pytest.fixture(autouse=True)
def mock_startup_dependencies():
    """Mock external service initialization to allow app startup in tests."""
    with patch("app.main.init_influx_client", new_callable=AsyncMock), \
         patch("app.main.get_influx_client") as mock_get_influx, \
         patch("app.main.init_redis_cache", new_callable=AsyncMock), \
         patch("app.main.init_supabase_client"), \
         patch("app.main.init_model_registry"), \
         patch("app.domain.pipeline.redis_inference_worker.RedisInferenceWorker") as MockWorker:
        
        # Configure Influx Mock to handle ensure_predictions_bucket await
        mock_influx_instance = AsyncMock()
        mock_get_influx.return_value = mock_influx_instance
        
        # Configure Worker Mock
        mock_worker_instance = AsyncMock()
        MockWorker.return_value = mock_worker_instance
        
        yield
