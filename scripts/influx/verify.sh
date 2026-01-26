#!/bin/bash
set -euo pipefail

# InfluxDB Verification Script
# Checks connection, auth, and buckets.
# Usage: ./scripts/influx/verify.sh

# Load .env if present (for local dev convenience)
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

# Defaults
INFLUX_URL="${INFLUX_URL:-http://localhost:8086}"
INFLUX_ORG="${INFLUX_ORG:-energy-monitor}"
INFLUX_BUCKET_PRED="${INFLUX_BUCKET_PRED:-predictions}"
INFLUX_TOKEN="${INFLUX_TOKEN:-}"

echo "========================================"
echo "  InfluxDB Setup Verification"
echo "========================================"
echo "URL:    $INFLUX_URL"
echo "ORG:    $INFLUX_ORG"
echo "Bucket: $INFLUX_BUCKET_PRED"
echo "----------------------------------------"

if [ -z "$INFLUX_TOKEN" ]; then
    echo "❌ ERROR: INFLUX_TOKEN is not set."
    echo "  Please set it in .env or environment."
    exit 1
fi

# 1. Check Connectivity
echo -n "Step 1: Checking connectivity... "
if curl -s -o /dev/null -f "$INFLUX_URL/health"; then
    echo "✅ OK"
else
    echo "❌ FAILED"
    echo "  Could not reach InfluxDB at $INFLUX_URL"
    exit 1
fi

# 2. Check Org
echo -n "Step 2: verifying organization... "
# List orgs and grep for name
if curl -s -f -H "Authorization: Token $INFLUX_TOKEN" "$INFLUX_URL/api/v2/orgs?org=$INFLUX_ORG" | grep -q "\"name\":\"$INFLUX_ORG\""; then
    echo "✅ OK"
else
    echo "❌ FAILED"
    echo "  Organization '$INFLUX_ORG' not found or token invalid."
    exit 1
fi

# 3. Check Buckets
echo "Step 3: Verifying buckets..."

check_bucket() {
    local bname=$1
    echo -n "  - Checking '$bname'... "
    if curl -s -f -H "Authorization: Token $INFLUX_TOKEN" "$INFLUX_URL/api/v2/buckets?org=$INFLUX_ORG&name=$bname" | grep -q "\"name\":\"$bname\""; then
        echo "✅ Found"
    else
        echo "❌ MISSING"
        echo "    Bucket '$bname' does not exist in org '$INFLUX_ORG'."
        return 1
    fi
}

check_bucket "$INFLUX_BUCKET_PRED"

# 4. Minimal Write/Read Test (Predictions)
echo -n "Step 4: Smoke Test (Write -> Read)... "
TEST_VAL="123.456"
# Write
curl -s -f -X POST "$INFLUX_URL/api/v2/write?org=$INFLUX_ORG&bucket=$INFLUX_BUCKET_PRED&precision=s" \
  -H "Authorization: Token $INFLUX_TOKEN" \
  --data-binary "smoke_test,test_id=verify val=$TEST_VAL $(date +%s)"

# Read (Flux)
QUERY="from(bucket:\"$INFLUX_BUCKET_PRED\") |> range(start: -1m) |> filter(fn: (r) => r._measurement == \"smoke_test\") |> last()"
RESPONSE=$(curl -s -f -X POST "$INFLUX_URL/api/v2/query?org=$INFLUX_ORG" \
  -H "Authorization: Token $INFLUX_TOKEN" \
  -H "Content-Type: application/vnd.flux" \
  --data "$QUERY")

if echo "$RESPONSE" | grep -q "$TEST_VAL"; then
    echo "✅ OK"
else
    echo "❌ FAILED"
    echo "  Could not read back value '$TEST_VAL'."
    echo "  Response: $RESPONSE"
    exit 1
fi

echo "========================================"
echo "✅ VERIFICATION SUCCESSFUL"
echo "========================================"
