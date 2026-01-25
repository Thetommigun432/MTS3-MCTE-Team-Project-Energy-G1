#!/usr/bin/env python3
"""
NILM Pipeline Smoke Test
=========================

Verifies the end-to-end pipeline works:
1. API is alive (/live)
2. API is ready (/ready)
3. Models are registered (/models)
4. Ingest endpoint accepts readings (/ingest/readings)

Usage:
    python scripts/smoke_pipeline.py

Environment Variables:
    BACKEND_URL: Backend API URL (default: http://localhost:8000)
    INGEST_TOKEN: Optional auth token for ingest endpoint
"""

import os
import sys
import time
import json
import urllib.request
import urllib.error
from datetime import datetime, timezone


def log(msg: str) -> None:
    """Print timestamped log message."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def request(url: str, method: str = "GET", data: dict | None = None, headers: dict | None = None) -> tuple[int, dict | str]:
    """Make HTTP request and return (status_code, response_body)."""
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


def main():
    base_url = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")
    ingest_token = os.environ.get("INGEST_TOKEN", "")
    
    log(f"Smoke testing: {base_url}")
    print("-" * 50)
    
    all_passed = True
    
    # 1. Check /live
    log("Checking /live...")
    status, body = request(f"{base_url}/live")
    if status == 200:
        log(f"  ✅ /live: OK")
    else:
        log(f"  ❌ /live: {status} - {body}")
        all_passed = False
    
    # 2. Check /ready
    log("Checking /ready...")
    status, body = request(f"{base_url}/ready")
    if status == 200:
        log(f"  ✅ /ready: OK")
        if isinstance(body, dict):
            for component, state in body.get("components", {}).items():
                symbol = "✅" if state.get("healthy") else "⚠️"
                log(f"      {symbol} {component}: {state.get('status', 'unknown')}")
    else:
        log(f"  ⚠️ /ready: {status} (may need dependencies)")
    
    # 3. Check /models
    log("Checking /models...")
    status, body = request(f"{base_url}/models")
    if status == 200:
        models = body.get("models", []) if isinstance(body, dict) else []
        log(f"  ✅ /models: {len(models)} models registered")
        for m in models[:3]:
            active = "🟢" if m.get("is_active") else "⚪"
            log(f"      {active} {m.get('model_id')} ({m.get('architecture')})")
        if len(models) > 3:
            log(f"      ... and {len(models) - 3} more")
    else:
        log(f"  ❌ /models: {status} - {body}")
        all_passed = False
    
    # 4. Check /ingest/readings (POST with sample data)
    log("Checking /ingest/readings...")
    sample_readings = {
        "building_id": "smoke-test",
        "readings": [
            {"power_watts": 500.0, "timestamp": datetime.now(timezone.utc).isoformat()},
            {"power_watts": 510.0},
            {"power_watts": 505.0},
        ]
    }
    
    headers = {}
    if ingest_token:
        headers["Authorization"] = f"Bearer {ingest_token}"
    
    status, body = request(f"{base_url}/ingest/readings", method="POST", data=sample_readings, headers=headers)
    if status in (200, 201, 202):
        log(f"  ✅ /ingest/readings: Accepted ({status})")
    elif status == 401:
        log(f"  ⚠️ /ingest/readings: Auth required (set INGEST_TOKEN)")
    else:
        log(f"  ❌ /ingest/readings: {status} - {body}")
        all_passed = False
    
    print("-" * 50)
    if all_passed:
        log("🎉 All smoke tests passed!")
        sys.exit(0)
    else:
        log("⚠️ Some tests failed - check output above")
        sys.exit(1)


if __name__ == "__main__":
    main()
