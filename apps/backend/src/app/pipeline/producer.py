"""
Data Producer - Simulates real-time sensor data
================================================

Sends raw power data to Redis for the inference service.
Can be used for testing or connected to real sensors.

Usage:
    python data_producer.py --source simulation
    python data_producer.py --source csv --file building_data.csv
    python data_producer.py --source mqtt --broker mqtt.example.com
"""

import os
import sys
import json
import time
import random
import argparse
from datetime import datetime, timezone
from typing import Generator, Dict

import redis


def simulate_building_power() -> Generator[Dict, None, None]:
    """
    Generate synthetic building power data.
    
    Simulates realistic patterns:
    - Base load (refrigerator, standby)
    - Time-of-day patterns (morning/evening peaks)
    - Random appliance activations
    """
    base_load = 300  # Watts (fridge, standby)
    
    # Appliance profiles (power, duration_seconds, probability_per_hour)
    appliances = {
        'heat_pump': (2500, 1800, 0.3),
        'washing_machine': (1500, 2400, 0.05),
        'dishwasher': (1800, 3600, 0.04),
        'dryer': (2000, 2700, 0.03),
        'electric_oven': (2500, 1800, 0.1),
        'microwave': (1200, 180, 0.2),
        'kettle': (2000, 120, 0.15),
    }
    
    active_appliances = {}
    
    while True:
        now = datetime.now(timezone.utc)
        hour = now.hour + now.minute / 60.0
        
        # Time-of-day multiplier (peaks at 7-9 AM and 6-9 PM)
        time_factor = 1.0
        if 7 <= hour <= 9:
            time_factor = 1.5
        elif 18 <= hour <= 21:
            time_factor = 1.8
        elif 23 <= hour or hour <= 6:
            time_factor = 0.6
        
        # Check for new appliance activations
        for app_name, (power, duration, prob) in appliances.items():
            if app_name not in active_appliances:
                # Probability adjusted for 1-second intervals
                if random.random() < (prob / 3600) * time_factor:
                    active_appliances[app_name] = {
                        'power': power * random.uniform(0.8, 1.2),
                        'remaining': duration * random.uniform(0.8, 1.2)
                    }
        
        # Calculate total power
        total_power = base_load
        
        # Add active appliances
        to_remove = []
        for app_name, state in active_appliances.items():
            total_power += state['power']
            state['remaining'] -= 1
            if state['remaining'] <= 0:
                to_remove.append(app_name)
        
        for app in to_remove:
            del active_appliances[app]
        
        # Add noise
        total_power += random.gauss(0, 50)
        total_power = max(100, total_power)
        
        yield {
            'timestamp': time.time(),
            'power_total': round(total_power, 1),
            'voltage': round(230 + random.gauss(0, 2), 1),
            'current': round(total_power / 230, 2),
            'power_factor': round(min(1.0, max(0.8, 0.95 + random.gauss(0, 0.02))), 3)
        }



def read_csv_file(file_path: str) -> Generator[Dict, None, None]:
    """
    Read power data from CSV file.
    
    Supports two formats:
        1. Clean format: time, aggregate_kw (from clean_csv_for_producer.py)
        2. Raw format: time, building_* columns
    """
    import pandas as pd
    
    print(f"Loading CSV: {file_path}")
    df = pd.read_csv(file_path)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")
    
    # Detect format
    if 'aggregate_kw' in df.columns:
        # Clean format
        power_col = 'aggregate_kw'
        print(f"  Using clean format: {power_col}")
    else:
        # Raw format - find building column that is not all zeros
        building_cols = [col for col in df.columns if col.startswith('building_')]
        power_col = None
        for col in building_cols:
            if df[col].sum() != 0:
                power_col = col
                break
        if not power_col:
            raise ValueError("No power column found in CSV")
        print(f"  Using raw format: {power_col}")
    
    # Check if power is in kW (typical for this data)
    power_max = df[power_col].max()
    is_kw = power_max < 100  # Assume kW if max < 100
    print(f"  Power max: {power_max:.2f} {'kW' if is_kw else 'W'}")
    
    for idx, row in df.iterrows():
        # Parse timestamp
        if pd.notna(row['time']):
            ts = pd.Timestamp(row['time']).timestamp()
        else:
            ts = time.time()
        
        # Get power (convert kW to W if needed)
        power_w = row[power_col] * 1000 if is_kw else row[power_col]
        
        yield {
            'timestamp': ts,
            'power_total': round(float(power_w), 1),
            'voltage': 230.0,
            'current': round(float(power_w) / 230, 2),
        }
        
        # Progress every 10000 rows
        if idx > 0 and idx % 10000 == 0:
            print(f"  Processed {idx:,} / {len(df):,} rows...")


def read_parquet_file(file_path: str, speed_factor: float = 1.0) -> Generator[Dict, None, None]:
    """
    Read power data from Parquet file (production format).
    
    Expected format:
        - Time: datetime column
        - Aggregate: power in kW
    
    Args:
        file_path: Path to parquet file
        speed_factor: Speedup factor (1.0 = realtime, 5.0 = 5x faster)
    
    Yields:
        Dict with timestamp, power_total (in WATTS), voltage, current
    """
    import pandas as pd
    
    print(f"Loading parquet: {file_path}")
    df = pd.read_parquet(file_path)
    print(f"  Loaded {len(df):,} rows")
    
    # Handle different column names
    time_col = 'Time' if 'Time' in df.columns else 'timestamp'
    power_col = 'Aggregate' if 'Aggregate' in df.columns else 'power_total'
    
    # Check if power is in kW (max < 100) or W (max > 100)
    power_max = df[power_col].max()
    is_kw = power_max < 100  # Assume kW if max < 100
    
    if is_kw:
        print(f"  Power in kW (max={power_max:.2f} kW), converting to W")
    else:
        print(f"  Power in W (max={power_max:.0f} W)")
    
    for idx, row in df.iterrows():
        # Get timestamp
        if hasattr(row[time_col], 'timestamp'):
            ts = row[time_col].timestamp()
        else:
            ts = float(row[time_col])
        
        # Get power (convert kW to W if needed)
        power_w = row[power_col] * 1000 if is_kw else row[power_col]
        
        yield {
            'timestamp': ts,
            'power_total': round(float(power_w), 1),
            'voltage': 230.0,
            'current': round(float(power_w) / 230, 2),
        }
        
        # Progress every 10000 rows
        if idx > 0 and idx % 10000 == 0:
            print(f"  Processed {idx:,} / {len(df):,} rows...")


def main():
    parser = argparse.ArgumentParser(description='NILM Data Producer')
    parser.add_argument('--redis-host', type=str, 
                        default=os.environ.get('REDIS_HOST', 'localhost'))
    parser.add_argument('--redis-port', type=int,
                        default=int(os.environ.get('REDIS_PORT', 6379)))
    parser.add_argument('--building-id', type=str, default='building_1')
    parser.add_argument('--source', type=str, choices=['simulation', 'csv', 'parquet'],
                        default='csv')
    parser.add_argument('--file', type=str, 
                        default='../../data/raw/1sec_new/household_2025_01_clean.csv',
                        help='Data file path (parquet or csv)')
    parser.add_argument('--interval', type=float, default=1.0,
                        help='Seconds between samples (default: 1.0 for realtime)')
    parser.add_argument('--skip', type=int, default=0,
                        help='Skip first N rows (to start from different point)')
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit to N rows (0 = all)')
    parser.add_argument('--fast', action='store_true',
                        help='No delay between samples (for testing buffer fill)')
    
    args = parser.parse_args()
    
    # Connect to Redis
    r = redis.Redis(host=args.redis_host, port=args.redis_port)
    
    # Wait for Redis
    while True:
        try:
            r.ping()
            break
        except redis.ConnectionError:
            print("Waiting for Redis...")
            time.sleep(1)
    
    print(f"Connected to Redis at {args.redis_host}:{args.redis_port}")
    
    channel = f"nilm:{args.building_id}:input"
    print(f"Publishing to channel: {channel}")
    
    # Select data source
    if args.source == 'simulation':
        data_gen = simulate_building_power()
    elif args.source == 'csv':
        if not args.file:
            print("ERROR: --file required for csv source")
            sys.exit(1)
        data_gen = read_csv_file(args.file)
    elif args.source == 'parquet':
        # Default to production parquet in same directory
        import os
        file_path = args.file
        if not os.path.isabs(file_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(script_dir, file_path)
        data_gen = read_parquet_file(file_path)
    
    # Main loop
    sample_count = 0
    interval = 0 if args.fast else args.interval
    
    print(f"\n🚀 Starting data stream (interval={interval}s, skip={args.skip}, limit={args.limit or 'all'})")
    print("-" * 60)

    # Use monotonic time for drift-free scheduling
    next_tick = time.monotonic()

    try:
        for data in data_gen:
            # Skip rows if requested
            if args.skip > 0 and sample_count < args.skip:
                sample_count += 1
                continue

            # Publish to Redis
            r.publish(channel, json.dumps(data))

            sample_count += 1
            effective_count = sample_count - args.skip

            if effective_count % 100 == 0:
                print(f"Sent {effective_count:,} samples | Power: {data['power_total']:.0f}W")

            # Check limit
            if args.limit > 0 and effective_count >= args.limit:
                print(f"\n✅ Reached limit of {args.limit} samples")
                break

            # Monotonic scheduling to prevent drift
            if interval > 0:
                next_tick += interval
                sleep_time = next_tick - time.monotonic()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                elif sleep_time < -interval:
                    # Fell behind by more than one interval, reset tick
                    print(f"⚠️ Producer fell behind by {-sleep_time:.2f}s, resetting tick")
                    next_tick = time.monotonic()

    except KeyboardInterrupt:
        print(f"\n⏹️ Stopped. Sent {sample_count - args.skip:,} samples total.")


if __name__ == "__main__":
    main()
