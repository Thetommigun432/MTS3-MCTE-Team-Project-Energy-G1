"""
Seed pipeline with data from y_test.npy.
Reads the test file, synthesizes aggregate power, and ingests into backend.
"""

import argparse
import asyncio
import numpy as np
import httpx
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Any

# Configure backend URL
DEFAULT_BACKEND_URL = "http://localhost:8000"

async def ingest_batch(client: httpx.AsyncClient, url: str, readings: List[Dict[str, Any]], token: str | None) -> int:
    """Send a batch of readings to the backend."""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Ingest-Token"] = token
        
    try:
        response = await client.post(
            f"{url}/ingest/readings",
            json={"readings": readings},
            headers=headers,
            timeout=30.0
        )
        response.raise_for_status()
        data = response.json()
        return data.get("ingested", 0)
    except httpx.HTTPError as e:
        print(f"Error ingesting batch: {e}")
        if hasattr(e, "response") and e.response:
            print(f"Response: {e.response.text}")
        return 0

async def seed_data(
    npy_path: str,
    backend_url: str,
    seconds: int,
    building_id: str,
    token: str | None
) -> None:
    """Read numpy file and seed data."""
    print(f"Loading data from {npy_path}...")
    try:
        # Load y_test: shape (N, 1024, 11)
        # Using mmap_mode='r' to avoid loading everything into RAM if large
        y = np.load(npy_path, mmap_mode='r')
    except Exception as e:
        print(f"Failed to load numpy file: {e}")
        return

    print(f"Data shape: {y.shape}")
    
    # Synthesize aggregate: sum across appliance axis (last axis)
    # y is normalized, so we need to scale if the model expects scaled input logic
    # But wait, looking at the previous plan, the transformer adapter handles scaling if needed?
    # Actually, y_test.npy from the prompt says "float32 (per-appliance ground truth sequences)"
    # Usually these are normalized. The prompt for implementation said:
    # "aggregate_norm = y.sum(axis=-1) -> shape (N, 1024)"
    # "Convert to kW: aggregate_kw = aggregate_norm * 13.5118 (P_MAX_kW)"
    
    P_MAX_KW = 13.5118
    
    # We take the first N windows to cover 'seconds'
    # Flattening logic: we want a continuous stream.
    # The windows in y_test might be overlapping or disparate.
    # For a demo, let's just flatten the aggregate of the first few samples to make a long stream.
    
    # Calculate how many windows we need
    # 1024 points per window.
    total_points_needed = seconds
    windows_needed = (total_points_needed // 1024) + 1
    
    if windows_needed > y.shape[0]:
        print(f"Warning: Requested {seconds}s but file only has {y.shape[0]*1024} points. Using max available.")
        windows_needed = y.shape[0]
        
    print(f"Processing {windows_needed} windows to generate stream...")
    
    # Extract and process
    subset = y[:windows_needed] # (W, 1024, 11)
    # Sum appliances to get aggregate
    agg_norm = subset.sum(axis=-1) # (W, 1024)
    # Flatten
    agg_flat_norm = agg_norm.flatten()
    # Apply scale
    agg_flat_kw = agg_flat_norm * P_MAX_KW
    
    # Trim to requested seconds
    if len(agg_flat_kw) > seconds:
        agg_flat_kw = agg_flat_kw[:seconds]
        
    total_points = len(agg_flat_kw)
    print(f"Prepared {total_points} readings.")
    
    # Generate timestamps
    # Ending now, going back specific amount
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(seconds=total_points)
    
    print(f"Time range: {start_time.isoformat()} -> {end_time.isoformat()}")
    
    # Create batches
    batch_size = 500
    total_ingested = 0
    
    async with httpx.AsyncClient() as client:
        for i in range(0, total_points, batch_size):
            batch_data = agg_flat_kw[i : i + batch_size]
            
            readings = []
            for j, kw in enumerate(batch_data):
                # Calculate timestamp for this point
                # i + j is the offset from 0
                point_time = start_time + timedelta(seconds=(i + j))
                
                readings.append({
                    "building_id": building_id,
                    "ts": point_time.isoformat(),
                    "aggregate_kw": float(kw)
                })
            
            count = await ingest_batch(client, backend_url, readings, token)
            total_ingested += count
            if (i // batch_size) % 10 == 0:
                print(f"Ingested {total_ingested}/{total_points}...")
                
    print(f"Seeding complete! Total ingested: {total_ingested}")
    print("\nTo view data, query the analytics endpoint:")
    print(f"GET {backend_url}/analytics/readings?building_id={building_id}&start=-2h&end=now()&resolution=1m&include_disaggregation=true")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed NILM pipeline from y_test.npy")
    parser.add_argument("--path", default="./y_test.npy", help="Path to y_test.npy")
    parser.add_argument("--backend-url", default=DEFAULT_BACKEND_URL, help="Backend URL")
    parser.add_argument("--seconds", type=int, default=7200, help="Number of seconds/points to ingest (default 2h)")
    parser.add_argument("--building-id", default="demo", help="Target building ID")
    parser.add_argument("--ingest-token", default=None, help="Ingestion token (X-Ingest-Token)")
    
    args = parser.parse_args()
    
    if not Path(args.path).exists():
        print(f"Error: File {args.path} not found.")
        exit(1)
        
    asyncio.run(seed_data(
        args.path,
        args.backend_url,
        args.seconds,
        args.building_id,
        args.ingest_token
    ))
