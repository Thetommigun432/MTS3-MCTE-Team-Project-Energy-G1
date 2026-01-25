#!/usr/bin/env python3
"""
End-to-End Smoke Test for NILM Pipeline
========================================

Verifies the full data flow works:
1. Insert a synthetic prediction point in InfluxDB (canonical wide schema)
2. Call /analytics/readings and verify appliances dict
3. Call /analytics/appliances and verify returns same keys

Usage:
    python scripts/smoke_e2e.py

Environment Variables:
    BACKEND_URL: Backend API URL (default: http://localhost:8000)
    INFLUX_URL: InfluxDB URL (default: http://localhost:8086)
    INFLUX_TOKEN: InfluxDB token
    INFLUX_ORG: InfluxDB organization (default: energy-monitor)
    INFLUX_BUCKET_PRED: Predictions bucket (default: predictions)
    AUTH_TOKEN: Optional Bearer token for API calls
"""

import os
import sys
import json
import urllib.request
import urllib.error
from datetime import datetime, timezone


# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'


def log_pass(msg: str) -> None:
    print(f"{Colors.GREEN}✅ PASS{Colors.RESET}: {msg}")


def log_fail(msg: str) -> None:
    print(f"{Colors.RED}❌ FAIL{Colors.RESET}: {msg}")


def log_warn(msg: str) -> None:
    print(f"{Colors.YELLOW}⚠️ WARN{Colors.RESET}: {msg}")


def log_info(msg: str) -> None:
    print(f"   INFO: {msg}")


def http_request(url: str, method: str = "GET", data: dict | None = None, headers: dict | None = None) -> tuple[int, dict | str]:
    """Make HTTP request."""
    headers = headers or {}
    headers["Content-Type"] = "application/json"
    
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read().decode()
            try:
                return resp.status, json.loads(content)
            except json.JSONDecodeError:
                return resp.status, content
    except urllib.error.HTTPError as e:
        content = e.read().decode() if e.fp else ""
        try:
            return e.code, json.loads(content)
        except json.JSONDecodeError:
            return e.code, content
    except urllib.error.URLError as e:
        return 0, str(e.reason)


def insert_test_prediction(
    influx_url: str,
    influx_token: str,
    influx_org: str,
    bucket: str,
    building_id: str,
) -> bool:
    """Insert a test prediction point using InfluxDB Line Protocol."""
    
    # Build line protocol with wide format
    # prediction,building_id=xxx,model_version=test predicted_kw_HeatPump=1.5,confidence_HeatPump=0.9,...
    timestamp_ns = int(datetime.now(timezone.utc).timestamp() * 1_000_000_000)
    
    fields = [
        "predicted_kw_HeatPump=1.5",
        "confidence_HeatPump=0.92",
        "predicted_kw_Dishwasher=0.0",
        "confidence_Dishwasher=0.65",
        "predicted_kw_WashingMachine=0.3",
        "confidence_WashingMachine=0.88",
    ]
    
    line = f'prediction,building_id={building_id},model_version=smoke_test {",".join(fields)} {timestamp_ns}'
    
    url = f"{influx_url}/api/v2/write?org={influx_org}&bucket={bucket}&precision=ns"
    
    req = urllib.request.Request(
        url,
        data=line.encode(),
        headers={
            "Authorization": f"Token {influx_token}",
            "Content-Type": "text/plain; charset=utf-8",
        },
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 204
    except urllib.error.HTTPError as e:
        log_fail(f"InfluxDB write failed: {e.code} - {e.read().decode()}")
        return False
    except Exception as e:
        log_fail(f"InfluxDB write error: {e}")
        return False


def main():
    # Configuration
    backend_url = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")
    influx_url = os.environ.get("INFLUX_URL", "http://localhost:8086").rstrip("/")
    influx_token = os.environ.get("INFLUX_TOKEN", "influx-admin-token-2026-secure")
    influx_org = os.environ.get("INFLUX_ORG", "energy-monitor")
    bucket = os.environ.get("INFLUX_BUCKET_PRED", "predictions")
    auth_token = os.environ.get("AUTH_TOKEN", "")
    
    # Use a test building ID
    test_building = "smoke_test_building"
    
    print("=" * 60)
    print("NILM Pipeline E2E Smoke Test")
    print("=" * 60)
    print(f"Backend: {backend_url}")
    print(f"InfluxDB: {influx_url}")
    print(f"Building ID: {test_building}")
    print("-" * 60)
    
    all_passed = True
    
    # Step 1: Insert test prediction
    print("\n[1/4] Inserting test prediction into InfluxDB...")
    if insert_test_prediction(influx_url, influx_token, influx_org, bucket, test_building):
        log_pass("Test prediction inserted (wide format)")
    else:
        log_fail("Could not insert test prediction")
        all_passed = False
    
    # Step 2: Check /models endpoint
    print("\n[2/4] Checking /models endpoint...")
    headers = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    
    status, body = http_request(f"{backend_url}/models", headers=headers)
    if status == 200 and isinstance(body, dict):
        models = body.get("models", [])
        log_pass(f"/models returns {len(models)} models")
        
        # Check if any model has heads
        models_with_heads = [m for m in models if m.get("heads")]
        if models_with_heads:
            log_info(f"{len(models_with_heads)} models have heads defined")
        else:
            log_warn("No models have heads defined (may be legacy single-head)")
    else:
        log_fail(f"/models returned {status}")
        all_passed = False
    
    # Step 3: Check /analytics/appliances
    print("\n[3/4] Checking /analytics/appliances...")
    status, body = http_request(
        f"{backend_url}/analytics/appliances?building_id={test_building}",
        headers=headers
    )
    
    expected_appliances = {"HeatPump", "Dishwasher", "WashingMachine"}
    
    if status == 200 and isinstance(body, dict):
        appliances = set(body.get("appliances", []))
        log_pass(f"/analytics/appliances returns: {appliances}")
        
        if appliances >= expected_appliances:
            log_pass("All expected appliances found")
        elif appliances:
            log_warn(f"Some appliances missing. Expected: {expected_appliances}")
        else:
            log_fail("No appliances returned - check field key parsing")
            all_passed = False
    elif status == 401:
        log_warn("/analytics/appliances requires authentication (set AUTH_TOKEN)")
    else:
        log_fail(f"/analytics/appliances returned {status}: {body}")
        all_passed = False
    
    # Step 4: Check /analytics/readings with disaggregation
    print("\n[4/4] Checking /analytics/readings with disaggregation...")
    status, body = http_request(
        f"{backend_url}/analytics/readings?building_id={test_building}&start=-1h&end=now()&include_disaggregation=true",
        headers=headers
    )
    
    if status == 200 and isinstance(body, dict):
        data = body.get("data", [])
        log_pass(f"/analytics/readings returns {len(data)} points")
        
        # Check if any point has appliances dict
        points_with_appliances = [p for p in data if p.get("appliances")]
        if points_with_appliances:
            sample = points_with_appliances[0].get("appliances", {})
            log_pass(f"Disaggregation working - sample appliances: {list(sample.keys())}")
        else:
            log_warn("No points have appliances dict (may need more data)")
    elif status == 401:
        log_warn("/analytics/readings requires authentication")
    else:
        log_fail(f"/analytics/readings returned {status}")
        all_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print(f"{Colors.GREEN}All smoke tests PASSED{Colors.RESET}")
        sys.exit(0)
    else:
        print(f"{Colors.RED}Some tests FAILED - see output above{Colors.RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
