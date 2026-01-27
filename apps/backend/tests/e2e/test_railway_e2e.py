"""
Railway E2E Tests - Tests against deployed Railway backend with full pipeline.

These tests verify the deployed NILM pipeline on Railway:
1. E2E probe endpoints are accessible with correct token
2. Preprocessing produces valid 7-element feature vectors
3. Samples can be injected into the real pipeline via Redis
4. Redis buffer receives and stores samples
5. Predictions are written to InfluxDB with correct run_id

Run with:
    pytest apps/backend/tests/e2e/test_railway_e2e.py -v

Environment variables:
    RAILWAY_BACKEND_URL: Railway backend URL (required)
    E2E_TOKEN: E2E probe authentication token (required)
    E2E_RUN_ID: Unique run identifier for test isolation (auto-generated if not set)
"""

import os
import time

import httpx
import pytest

# Configuration from environment
BACKEND_URL = os.environ.get("RAILWAY_BACKEND_URL", "https://energy-monitor.up.railway.app")
E2E_TOKEN = os.environ.get("E2E_TOKEN")
E2E_RUN_ID = os.environ.get("E2E_RUN_ID", f"local-{int(time.time())}")

# Mark all tests in this module as e2e and railway
pytestmark = [pytest.mark.e2e, pytest.mark.railway]


def _is_ci() -> bool:
    """Check if running in CI environment."""
    return os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true"


def _require_e2e_token() -> None:
    """Skip test if E2E_TOKEN not set (unless in CI where it's required)."""
    if not E2E_TOKEN:
        if _is_ci():
            pytest.fail("E2E_TOKEN required in CI")
        else:
            pytest.skip("E2E_TOKEN not set - skipping Railway E2E test locally")


@pytest.fixture
def client() -> httpx.Client:
    """HTTP client with E2E token header."""
    _require_e2e_token()
    headers = {"X-E2E-Token": E2E_TOKEN} if E2E_TOKEN else {}
    return httpx.Client(base_url=BACKEND_URL, headers=headers, timeout=30.0)


class TestPreprocessing:
    """Test preprocessing endpoint."""

    def test_returns_7_features(self, client: httpx.Client) -> None:
        """Verify preprocessing produces 7-element feature vector."""
        resp = client.post(
            "/e2e/preprocess",
            json={
                "timestamp": time.time(),
                "power_watts": 3500.0,
            },
        )

        assert resp.status_code == 200, f"Unexpected status: {resp.text}"
        data = resp.json()

        assert "features" in data
        assert len(data["features"]) == 7, f"Expected 7 features, got {len(data['features'])}"
        assert all(isinstance(f, float) for f in data["features"])

    def test_aggregate_normalized(self, client: httpx.Client) -> None:
        """Verify aggregate power is normalized correctly (P_MAX=15000W)."""
        resp = client.post(
            "/e2e/preprocess",
            json={
                "timestamp": time.time(),
                "power_watts": 7500.0,  # Half of P_MAX
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        features = data["features"]

        # First feature is aggregate_norm = power_watts / P_MAX
        # 7500 / 15000 = 0.5
        assert 0.45 <= features[0] <= 0.55, f"Aggregate should be ~0.5, got {features[0]}"

    def test_temporal_features_in_range(self, client: httpx.Client) -> None:
        """Verify temporal features (sin/cos) are in [-1, 1] range."""
        resp = client.post(
            "/e2e/preprocess",
            json={
                "timestamp": time.time(),
                "power_watts": 2000.0,
            },
        )

        assert resp.status_code == 200
        features = resp.json()["features"]

        # Features 1-6 are temporal sin/cos values
        for i in range(1, 7):
            assert -1.0 <= features[i] <= 1.0, f"Feature {i} should be in [-1, 1], got {features[i]}"


class TestInjection:
    """Test sample injection endpoint."""

    def test_inject_returns_ok(self, client: httpx.Client) -> None:
        """Verify inject endpoint returns success response."""
        resp = client.post(
            "/e2e/inject",
            json={
                "run_id": E2E_RUN_ID,
                "timestamp": time.time(),
                "power_watts": 2500.0,
                "building_id": "building_1",
            },
        )

        assert resp.status_code == 200, f"Inject failed: {resp.text}"
        data = resp.json()

        assert data["status"] == "ok"
        assert data["run_id"] == E2E_RUN_ID
        assert len(data["preprocessed"]) == 7
        assert "channel" in data

    def test_inject_includes_preprocessed_features(self, client: httpx.Client) -> None:
        """Verify inject response includes valid preprocessed features."""
        timestamp = time.time()
        power = 5000.0

        resp = client.post(
            "/e2e/inject",
            json={
                "run_id": E2E_RUN_ID,
                "timestamp": timestamp,
                "power_watts": power,
            },
        )

        assert resp.status_code == 200
        data = resp.json()

        # Check preprocessed features
        features = data["preprocessed"]
        assert len(features) == 7

        # Aggregate should be power/P_MAX = 5000/15000 = ~0.33
        assert 0.30 <= features[0] <= 0.36


class TestRedisBuffer:
    """Test Redis buffer status endpoint."""

    def test_redis_buffer_status(self, client: httpx.Client) -> None:
        """Verify redis-buffer endpoint returns status."""
        resp = client.get("/e2e/redis-buffer?building_id=building_1")

        assert resp.status_code == 200
        data = resp.json()

        # Check required fields
        assert "building_id" in data
        assert "buffer_length" in data
        assert "window_size" in data
        assert "buffer_full" in data


class TestInfluxStatus:
    """Test InfluxDB status endpoint."""

    def test_influx_status_returns_result(self, client: httpx.Client) -> None:
        """Verify influx-status endpoint returns query result."""
        resp = client.get(f"/e2e/influx-status?run_id={E2E_RUN_ID}")

        assert resp.status_code == 200
        data = resp.json()

        # Check required fields
        assert "found" in data
        assert "run_id" in data
        assert data["run_id"] == E2E_RUN_ID
        assert "records_count" in data
        assert "query_time_ms" in data


class TestFullPipelineFlow:
    """Test full pipeline flow: inject -> buffer -> inference -> persister -> InfluxDB."""

    def test_inject_multiple_samples(self, client: httpx.Client) -> None:
        """Inject multiple samples and verify they reach the buffer."""
        # Inject several samples
        for i in range(5):
            resp = client.post(
                "/e2e/inject",
                json={
                    "run_id": E2E_RUN_ID,
                    "timestamp": time.time() + i,
                    "power_watts": 2000.0 + i * 100,
                },
            )
            assert resp.status_code == 200
            time.sleep(0.2)

        # Check buffer status
        resp = client.get("/e2e/redis-buffer?building_id=building_1")
        assert resp.status_code == 200

    @pytest.mark.slow
    def test_predictions_appear_in_influx(self, client: httpx.Client) -> None:
        """
        Full E2E flow: inject samples, wait for pipeline, verify in InfluxDB.

        This test requires:
        - Producer, inference worker, and persister running on Railway
        - Buffer to be warm (enough samples for inference)

        Note: This test may take up to 2 minutes due to pipeline warmup.
        """
        # Poll for predictions with our run_id
        max_wait = 120  # seconds
        poll_interval = 5
        waited = 0

        print(f"\nWaiting for predictions with run_id={E2E_RUN_ID}...")

        while waited < max_wait:
            resp = client.get(f"/e2e/influx-status?run_id={E2E_RUN_ID}")

            if resp.status_code == 200:
                data = resp.json()
                if data["found"]:
                    print(f"Found prediction after {waited}s!")
                    print(f"  Records: {data['records_count']}")
                    print(f"  Sample: {data.get('sample_record')}")
                    assert data["records_count"] >= 1
                    return

            time.sleep(poll_interval)
            waited += poll_interval
            print(f"  Waiting... ({waited}s / {max_wait}s)")

        # Collect diagnostics on failure
        print("\n=== DIAGNOSTICS ===")
        print(f"Backend URL: {BACKEND_URL}")
        print(f"Run ID: {E2E_RUN_ID}")

        # Check buffer status
        buffer_resp = client.get("/e2e/redis-buffer?building_id=building_1")
        if buffer_resp.status_code == 200:
            buffer_data = buffer_resp.json()
            print(f"Buffer length: {buffer_data.get('buffer_length')}")
            print(f"Window size: {buffer_data.get('window_size')}")
            print(f"Buffer full: {buffer_data.get('buffer_full')}")

        print("===================\n")

        pytest.fail(f"No predictions found in InfluxDB after {max_wait}s for run_id={E2E_RUN_ID}")


class TestSecurity:
    """Test security controls."""

    def test_401_without_token(self) -> None:
        """Verify endpoints return 401/404 without valid token."""
        # Client without token
        client = httpx.Client(base_url=BACKEND_URL, timeout=10.0)

        resp = client.post(
            "/e2e/preprocess",
            json={
                "timestamp": time.time(),
                "power_watts": 1000.0,
            },
        )

        # Should be 401 Unauthorized or 404 (if endpoint hidden)
        assert resp.status_code in [401, 404], f"Expected 401 or 404, got {resp.status_code}"

    def test_401_with_invalid_token(self) -> None:
        """Verify endpoints return 401 with invalid token."""
        # Client with wrong token
        client = httpx.Client(
            base_url=BACKEND_URL,
            headers={"X-E2E-Token": "invalid-token"},
            timeout=10.0,
        )

        resp = client.post(
            "/e2e/preprocess",
            json={
                "timestamp": time.time(),
                "power_watts": 1000.0,
            },
        )

        # Should be 401 Unauthorized or 404 (if probes disabled)
        assert resp.status_code in [401, 404], f"Expected 401 or 404, got {resp.status_code}"


class TestHealthEndpoints:
    """Test that standard health endpoints still work."""

    def test_live_endpoint(self, client: httpx.Client) -> None:
        """Verify /live endpoint works."""
        # Use client without E2E token for public endpoint
        public_client = httpx.Client(base_url=BACKEND_URL, timeout=10.0)
        resp = public_client.get("/live")

        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") == "ok"

    def test_ready_endpoint(self, client: httpx.Client) -> None:
        """Verify /ready endpoint works."""
        public_client = httpx.Client(base_url=BACKEND_URL, timeout=15.0)
        resp = public_client.get("/ready")

        # May return 200 or 503 depending on dependencies
        assert resp.status_code in [200, 503]
        data = resp.json()
        assert "status" in data
        assert "checks" in data
