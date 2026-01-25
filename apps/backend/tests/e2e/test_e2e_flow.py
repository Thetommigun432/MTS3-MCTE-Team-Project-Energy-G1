"""
End-to-end tests for the NILM pipeline.

These tests require a running Docker stack with all services:
- InfluxDB (time-series database)
- Redis (message queue)
- Backend API (FastAPI)
- Producer (data ingestion)
- Inference (ML predictions)
- Persister (write predictions to InfluxDB)

Run with: pytest -m e2e -v
"""

import pytest
import os
import time
import httpx
from influxdb_client import InfluxDBClient

# Configuration from environment - required in CI
INFLUX_URL = os.environ.get("INFLUX_URL", "http://localhost:8086")
INFLUX_TOKEN = os.environ.get("INFLUX_TOKEN")
INFLUX_ORG = os.environ.get("INFLUX_ORG", "energy-monitor")
INFLUX_BUCKET_PRED = os.environ.get("INFLUX_BUCKET_PRED", "predictions")
INFLUX_MEASUREMENT = os.environ.get("INFLUX_MEASUREMENT_PRED", "prediction")
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
E2E_RUN_ID = os.environ.get("E2E_RUN_ID")

# Skip E2E tests if required env vars are missing locally
def _is_ci():
    return os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true"


def _require_influx_token():
    """Skip if INFLUX_TOKEN not set and not in CI."""
    if not INFLUX_TOKEN:
        if _is_ci():
            pytest.fail("INFLUX_TOKEN required in CI")
        else:
            pytest.skip("INFLUX_TOKEN not set - skipping E2E test locally")


@pytest.fixture
def influx_client():
    _require_influx_token()
    
    client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    
    # Ensure bucket exists (resilience against slow init)
    try:
        buckets_api = client.buckets_api()
        bucket = buckets_api.find_bucket_by_name(INFLUX_BUCKET_PRED)
        if not bucket:
            print(f"Creating missing bucket: {INFLUX_BUCKET_PRED}")
            buckets_api.create_bucket(bucket_name=INFLUX_BUCKET_PRED, org=INFLUX_ORG)
            print("Bucket created. Waiting for consistency...")
            time.sleep(2)
    except Exception as e:
        print(f"Warning: Failed to ensure bucket exists: {e}")

    yield client
    client.close()


def _retry_with_backoff(fn, max_attempts=10, initial_delay=1.0, max_delay=10.0):
    """
    Execute fn() with exponential backoff retry.
    
    Args:
        fn: Callable that returns (success: bool, result: Any)
        max_attempts: Maximum retry attempts
        initial_delay: Starting delay in seconds
        max_delay: Maximum delay between retries
    
    Returns:
        The result from fn() on success, or raises the last exception
    """
    delay = initial_delay
    last_error = None
    
    for attempt in range(max_attempts):
        try:
            success, result = fn()
            if success:
                return result
        except Exception as e:
            last_error = e
        
        if attempt < max_attempts - 1:
            time.sleep(delay)
            delay = min(delay * 2, max_delay)  # Exponential backoff with cap
    
    if last_error:
        raise last_error
    return None


@pytest.mark.e2e
def test_pipeline_produces_predictions(influx_client):
    """
    Verifies the complete NILM pipeline:
    1. Data Producer sends data to Redis (implied if predictions exist)
    2. Inference Service processes data and generates predictions (implied)
    3. Prediction Persister writes to InfluxDB (validated here)

    Uses E2E_RUN_ID for test isolation when running in CI.
    """
    print(f"Waiting for pipeline to generate predictions...")
    print(f"Target: Bucket={INFLUX_BUCKET_PRED}, Measurement={INFLUX_MEASUREMENT}, RunID={E2E_RUN_ID}")
    
    query_api = influx_client.query_api()
    
    # Build Query
    query = f'''
        from(bucket: "{INFLUX_BUCKET_PRED}")
        |> range(start: -24h)
        |> filter(fn: (r) => r["_measurement"] == "{INFLUX_MEASUREMENT}")
    '''
    
    if E2E_RUN_ID:
        query += f' |> filter(fn: (r) => r["run_id"] == "{E2E_RUN_ID}")'
        
    query += ' |> limit(n: 1)'
    
    found_record = None
    last_error = None
    
    def try_query():
        nonlocal found_record, last_error
        try:
            tables = query_api.query(query)
            for table in tables:
                if len(table.records) > 0:
                    found_record = table.records[0]
                    return True, found_record
        except Exception as e:
            last_error = e
        return False, None
    
    # Retry with exponential backoff: 1s, 2s, 4s, 8s, 10s (capped)
    # Total max wait: ~60s across 15 attempts
    try:
        result = _retry_with_backoff(try_query, max_attempts=15, initial_delay=1.0, max_delay=10.0)
        if result:
            print(f"FOUND PREDICTION: {result.values}")
            
            # Validate prediction schema
            values = result.values
            assert "_measurement" in values, "Missing _measurement field"
            # Note: field structure depends on wide vs narrow format
            
            return  # Success!
    except Exception as e:
        last_error = e
        
    # Failure path
    print("\n=== E2E TEST FAILED ===")
    print(f"Influx URL: {INFLUX_URL}")
    print(f"Bucket: {INFLUX_BUCKET_PRED}")
    print(f"Measurement: {INFLUX_MEASUREMENT}")
    print(f"Run ID: {E2E_RUN_ID}")
    print(f"Last Query Error: {last_error}")
    print("Hint: Check docker compose logs for 'nilm-persister' or 'nilm-inference'.")
    print("=======================\n")
        
    pytest.fail(f"Pipeline failed to produce predictions. RunID: {E2E_RUN_ID}")


@pytest.mark.e2e
def test_backend_live():
    """Verify backend is reachable via /live endpoint."""
    
    def try_live_check():
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{BACKEND_URL}/live")
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("status") == "ok":
                        return True, data
        except httpx.RequestError:
            pass
        return False, None
    
    try:
        result = _retry_with_backoff(try_live_check, max_attempts=10, initial_delay=1.0, max_delay=5.0)
        if result:
            print(f"Backend live check passed: {result}")
            return
    except Exception as e:
        print(f"Backend live check failed: {e}")

    print(f"\n=== BACKEND LIVE CHECK FAILED ===")
    print(f"Backend URL: {BACKEND_URL}")
    print("Hint: Check docker compose logs for 'backend' service.")
    print("==================================\n")
    pytest.fail(f"Backend not reachable at {BACKEND_URL}/live")
