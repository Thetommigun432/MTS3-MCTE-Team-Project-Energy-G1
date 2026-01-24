#!/bin/bash
set -euo pipefail

# Seed Sample Data
# Writes a few sample points to the raw bucket.
# Usage: ./scripts/influx/seed_raw_sample.sh

if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

INFLUX_URL="${INFLUX_URL:-http://localhost:8086}"
INFLUX_ORG="${INFLUX_ORG:-energy-monitor}"
INFLUX_BUCKET_RAW="${INFLUX_BUCKET_RAW:-raw_sensor_data}"
INFLUX_TOKEN="${INFLUX_TOKEN:-}"

if [ -z "$INFLUX_TOKEN" ]; then
    echo "Error: INFLUX_TOKEN not set."
    exit 1
fi

echo "Seeding data to bucket '$INFLUX_BUCKET_RAW'..."

# Generate line protocol for last 1 hour
HOST="bldg_test"
NOW=$(date +%s)

for i in {0..5}; do
    TS=$((NOW - i * 60))
    VAL=$((1000 + RANDOM % 500))
    # Line Protocol: measurement,tags fields timestamp
    echo "energy_reading,building_id=$HOST power=$VAL $TS"
    
    curl -s -X POST "$INFLUX_URL/api/v2/write?org=$INFLUX_ORG&bucket=$INFLUX_BUCKET_RAW&precision=s" \
      -H "Authorization: Token $INFLUX_TOKEN" \
      --data-binary "energy_reading,building_id=$HOST power=$VAL $TS"
done

echo "✅ Seeded 6 points."
