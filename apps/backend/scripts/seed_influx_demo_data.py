#!/usr/bin/env python3
"""
Seed InfluxDB with demo data and run inference to populate predictions.

Usage:
    python scripts/seed_influx_demo_data.py --backend-url http://localhost:8000 --building-id demo

Requires:
    ENV=test
    TEST_JWT_SECRET=<secret>

This script:
1. Loads fixture aggregate power data
2. Sends each window to the /infer endpoint
3. Predictions are written to InfluxDB by the backend
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
import jwt
import numpy as np


def generate_test_jwt(secret: str, user_id: str = "demo-user") -> str:
    """Generate HS256 JWT for test mode."""
    payload = {
        "sub": user_id,
        "role": "authenticated",
        "exp": datetime.now(timezone.utc) + timedelta(hours=1),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, secret, algorithm="HS256")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Seed InfluxDB with demo predictions")
    parser.add_argument(
        "--backend-url",
        default="http://localhost:8000",
        help="Backend URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--building-id",
        default="demo",
        help="Building ID for demo data (default: demo)",
    )
    parser.add_argument(
        "--jwt-secret",
        default=os.environ.get("TEST_JWT_SECRET", "test-secret"),
        help="JWT secret for authentication (default from TEST_JWT_SECRET env var)",
    )
    parser.add_argument(
        "--model-id",
        default="transformer_hybrid_v1",
        help="Model ID to use for inference (default: transformer_hybrid_v1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without making requests",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Influx Demo Data Seeder")
    print("=" * 60)
    print(f"Backend URL: {args.backend_url}")
    print(f"Building ID: {args.building_id}")
    print(f"Model ID: {args.model_id}")

    # Load fixture
    fixtures_dir = Path(__file__).parent.parent / "tests" / "fixtures"
    aggregate_path = fixtures_dir / "aggregate_kw_sample.npy"

    if not aggregate_path.exists():
        print(f"ERROR: Fixture not found: {aggregate_path}")
        print("Run `python scripts/build_fixture_from_y_test.py` first.")
        sys.exit(1)

    aggregate = np.load(aggregate_path)
    print(f"Loaded {len(aggregate)} windows from fixtures")

    if args.dry_run:
        print("\n[DRY RUN] Would send the following requests:")
        for i, window in enumerate(aggregate):
            print(f"  [{i+1}/{len(aggregate)}] Window shape: {window.shape}, "
                  f"sum: {window.sum():.2f} kW")
        print("\nExiting (dry run)")
        return

    # Generate JWT
    token = generate_test_jwt(args.jwt_secret)
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Verify backend is alive
        try:
            r = await client.get(f"{args.backend_url}/live")
            print(f"\nBackend status: {r.status_code}")
            if r.status_code != 200:
                print(f"Backend not responding correctly: {r.text}")
                sys.exit(1)
        except httpx.ConnectError:
            print(f"ERROR: Cannot connect to backend at {args.backend_url}")
            print("Make sure the backend is running: uvicorn app.main:app --reload --port 8000")
            sys.exit(1)

        # Run inference for each sample
        now = datetime.now(timezone.utc)
        success_count = 0
        error_count = 0

        print(f"\nSending {len(aggregate)} inference requests...")

        for i, window in enumerate(aggregate):
            # Space samples 10 minutes apart, going backwards from now
            timestamp = (now - timedelta(minutes=10 * (len(aggregate) - i - 1))).isoformat()

            payload = {
                "building_id": args.building_id,
                "window": window.tolist(),
                "model_id": args.model_id,
                "timestamp": timestamp,
            }

            try:
                r = await client.post(
                    f"{args.backend_url}/infer",
                    json=payload,
                    headers=headers,
                )

                if r.status_code == 200:
                    data = r.json()
                    total_kw = sum(data.get("predicted_kw", {}).values())
                    print(f"  [{i+1}/{len(aggregate)}] OK - Total predicted: {total_kw:.2f} kW")
                    success_count += 1
                else:
                    print(f"  [{i+1}/{len(aggregate)}] Error {r.status_code}: {r.text[:100]}")
                    error_count += 1

            except Exception as e:
                print(f"  [{i+1}/{len(aggregate)}] Exception: {e}")
                error_count += 1

    print("=" * 60)
    print(f"Completed: {success_count} success, {error_count} errors")
    if success_count > 0:
        print(f"Frontend should now show predictions for building '{args.building_id}'")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
