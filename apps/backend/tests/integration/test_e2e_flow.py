
import pytest
import os
import time
import requests
from influxdb_client import InfluxDBClient

# Configuration
INFLUX_URL = os.environ.get("INFLUX_URL", "http://localhost:8086")
INFLUX_TOKEN = os.environ.get("INFLUX_TOKEN", "influx-admin-token-2026-secure")
INFLUX_ORG = os.environ.get("INFLUX_ORG", "energy-monitor")
INFLUX_BUCKET_PRED = os.environ.get("INFLUX_BUCKET_PRED", "predictions")
BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:8000")

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
    except Exception as e:
        print(f"Warning: Failed to ensure bucket exists: {e}")

    yield client
    client.close()

def test_pipeline_produces_predictions(influx_client):
    """
    Verifies that:
    1. Data Producer sends data (implied if predictions exist)
    2. Inference Service processes data (implied)
    3. Prediction Persister writes to InfluxDB (validated here)
    """
    print("Waiting for pipeline to generate predictions...")
    
    # Wait up to 30 seconds for data to flow
    query_api = influx_client.query_api()
    query = f'''
        from(bucket: "{INFLUX_BUCKET_PRED}")
        |> range(start: -10m)
        |> filter(fn: (r) => r["_measurement"] == "nilm_predictions")
        |> limit(n: 1)
    '''
    
    found = False
    for _ in range(15):
        try:
            tables = query_api.query(query)
            if tables and len(tables[0].records) > 0:
                found = True
                record = tables[0].records[0]
                print(f"FOUND PREDICTION: {record.values}")
                break
        except Exception as e:
            print(f"Query error: {e}")
            
        time.sleep(2)
        
    assert found, "Pipeline failed to produce predictions in InfluxDB within timeout."

def test_backend_live():
    """Verify backend is reachable."""
    try:
        resp = requests.get(f"{BACKEND_URL}/live")
        assert resp.status_code == 200
    except requests.exceptions.ConnectionError:
        pytest.fail("Backend not reachable")
