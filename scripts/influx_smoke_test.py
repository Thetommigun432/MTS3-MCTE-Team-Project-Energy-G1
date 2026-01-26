#!/usr/bin/env python3
"""
InfluxDB Smoke Test Script
--------------------------
Verifies connectivity, write, and query capability of an InfluxDB instance.
Can be run against local or remote instances via environment variables.

Usage:
    export INFLUX_URL=http://localhost:8086
    export INFLUX_TOKEN=my-token
    export INFLUX_ORG=my-org
    export INFLUX_BUCKET=my-bucket
    python scripts/influx_smoke_test.py
"""

import os
import sys
import time
from datetime import datetime
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

def get_env(name, default=None):
    val = os.getenv(name, default)
    if val is None:
        print(f"Error: Environment variable {name} is not set.")
        sys.exit(1)
    return val

def main():
    print("=== InfluxDB Smoke Test ===")
    
    # 1. Configuration
    url = get_env("INFLUX_URL", "http://localhost:8086")
    token = get_env("INFLUX_TOKEN")
    org = get_env("INFLUX_ORG", "energy-monitor")
    bucket = get_env("INFLUX_BUCKET", "predictions")
    
    print(f"Target: {url}")
    print(f"Org:    {org}")
    print(f"Bucket: {bucket}")
    
    # 2. Connect
    print("\n[1/3] Connecting...", end=" ", flush=True)
    try:
        client = InfluxDBClient(url=url, token=token, org=org, timeout=10000)
        ready = client.ping()
        if not ready:
            print("FAILED (Ping returned False)")
            sys.exit(1)
        print("OK")
    except Exception as e:
        print(f"FAILED\nError: {e}")
        sys.exit(1)

    # 3. Write
    print("[2/3] Writing test point...", end=" ", flush=True)
    try:
        write_api = client.write_api(write_options=SYNCHRONOUS)
        point = (
            Point("smoke_test")
            .tag("host", "smoke-script")
            .field("value", 1.0)
            .time(datetime.utcnow())
        )
        write_api.write(bucket=bucket, org=org, record=point)
        print("OK")
    except Exception as e:
        print(f"FAILED\nError: {e}")
        sys.exit(1)

    # 4. Query
    print("[3/3] Querying test point...", end=" ", flush=True)
    try:
        query_api = client.query_api()
        # Query specifically for our measurement in the last minute
        query = f'''
        from(bucket: "{bucket}")
            |> range(start: -1m)
            |> filter(fn: (r) => r._measurement == "smoke_test")
            |> filter(fn: (r) => r.host == "smoke-script")
            |> last()
        '''
        tables = query_api.query(query, org=org)
        
        found = False
        for table in tables:
            for record in table.records:
                if record.get_value() == 1.0:
                    found = True
                    break
            if found: break
            
        if found:
            print("OK")
        else:
            print("FAILED (Point not found in query results)")
            sys.exit(1)
            
    except Exception as e:
        print(f"FAILED\nError: {e}")
        sys.exit(1)

    print("\n[SUCCESS] Smoke test passed!")
    client.close()

if __name__ == "__main__":
    main()
