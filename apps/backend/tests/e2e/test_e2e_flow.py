
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
import requests
from influxdb_client import InfluxDBClient

# Configuration from environment with sensible defaults
INFLUX_URL = os.environ.get("INFLUX_URL", "http://localhost:8086")
INFLUX_TOKEN = os.environ.get("INFLUX_TOKEN", "influx-admin-token-2026-secure")
INFLUX_ORG = os.environ.get("INFLUX_ORG", "energy-monitor")
INFLUX_BUCKET_PRED = os.environ.get("INFLUX_BUCKET_PRED", "predictions")
INFLUX_MEASUREMENT = os.environ.get("INFLUX_MEASUREMENT_PRED", "prediction")
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
E2E_RUN_ID = os.environ.get("E2E_RUN_ID")

@pytest.fixture
def influx_client():
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
    
    # Wait up to 60 seconds (increased from 30s)
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
    
    found = False
    last_error = None
    
    for attempt in range(20):  # 20 * 3s = 60s
        try:
            tables = query_api.query(query)
            # Search across all returned tables/records
            for table in tables:
                if len(table.records) > 0:
                    found = True
                    record = table.records[0]
                    print(f"FOUND PREDICTION: {record.values}")
                    break
            if found:
                break
        except Exception as e:
            last_error = e
            # print(f"Query attempt {attempt} failed: {e}")
            
        time.sleep(3)
        
    if not found:
        print("\n=== E2E TEST FAILED ===")
        print(f"Influx URL: {INFLUX_URL}")
        print(f"Bucket: {INFLUX_BUCKET_PRED}")
        print(f"Measurement: {INFLUX_MEASUREMENT}")
        print(f"Run ID: {E2E_RUN_ID}")
        print(f"Last Query Error: {last_error}")
        print("Hint: Check docker compose logs for 'nilm-persister' or 'nilm-inference'.")
        print("=======================\n")
        
    assert found, f"Pipeline failed to produce predictions. RunID: {E2E_RUN_ID}"

@pytest.mark.e2e
def test_backend_live():
    """Verify backend is reachable via /live endpoint."""
    max_attempts = 10
    last_error = None

    for attempt in range(max_attempts):
        try:
            resp = requests.get(f"{BACKEND_URL}/live", timeout=5)
            assert resp.status_code == 200
            data = resp.json()
            assert data.get("status") == "ok", f"Unexpected status: {data}"
            print(f"Backend live check passed on attempt {attempt + 1}")
            return
        except requests.exceptions.ConnectionError as e:
            last_error = e
            if attempt < max_attempts - 1:
                time.sleep(2)
                continue
        except requests.exceptions.Timeout as e:
            last_error = e
            if attempt < max_attempts - 1:
                time.sleep(2)
                continue

    print(f"\n=== BACKEND LIVE CHECK FAILED ===")
    print(f"Backend URL: {BACKEND_URL}")
    print(f"Last Error: {last_error}")
    print("Hint: Check docker compose logs for 'backend' service.")
    print("==================================\n")
    pytest.fail(f"Backend not reachable at {BACKEND_URL}/live after {max_attempts} attempts")
