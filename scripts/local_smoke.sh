#!/bin/bash
# =============================================================================
# NILM Local Smoke Test Script
# Validates that the local Docker Compose stack is running correctly
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

BACKEND_URL="${BACKEND_URL:-http://localhost:8000}"
BUILDING_ID="${BUILDING_ID:-building-1}"

echo "=============================================="
echo "NILM Local Smoke Test"
echo "=============================================="
echo "Backend URL: $BACKEND_URL"
echo "Building ID: $BUILDING_ID"
echo ""

# Function to check endpoint
check_endpoint() {
    local name=$1
    local url=$2
    local expected_status=${3:-200}

    echo -n "Checking $name... "
    response=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null || echo "000")

    if [ "$response" = "$expected_status" ]; then
        echo -e "${GREEN}OK${NC} (HTTP $response)"
        return 0
    else
        echo -e "${RED}FAILED${NC} (HTTP $response, expected $expected_status)"
        return 1
    fi
}

# 1. Check backend health endpoints
echo ""
echo "1. Health Endpoints"
echo "-------------------"
check_endpoint "/live" "$BACKEND_URL/live"
check_endpoint "/ready" "$BACKEND_URL/ready"

# 2. Check API endpoints (no auth required for these)
echo ""
echo "2. API Endpoints"
echo "----------------"
check_endpoint "/api/models" "$BACKEND_URL/api/models"

# 3. Check Redis rolling window
echo ""
echo "3. Redis Rolling Window"
echo "-----------------------"
echo -n "Checking window length... "

# Use docker exec to run redis-cli
window_len=$(docker exec nilm-redis redis-cli LLEN "nilm:$BUILDING_ID:window" 2>/dev/null || echo "error")

if [ "$window_len" = "error" ]; then
    echo -e "${YELLOW}SKIP${NC} (Redis not accessible via docker exec)"
else
    echo -e "${GREEN}$window_len${NC} samples"

    if [ "$window_len" -gt 0 ]; then
        echo -e "  ${GREEN}Window is being populated${NC}"
    else
        echo -e "  ${YELLOW}Window is empty (simulator may still be starting)${NC}"
    fi

    # Check if at or near capacity (3600)
    if [ "$window_len" -ge 3600 ]; then
        echo -e "  ${GREEN}Window at capacity (3600)${NC}"
    elif [ "$window_len" -ge 3500 ]; then
        echo -e "  ${GREEN}Window near capacity ($window_len/3600)${NC}"
    fi
fi

# 4. Check for predictions in InfluxDB
echo ""
echo "4. InfluxDB Predictions"
echo "-----------------------"
echo -n "Checking predictions endpoint... "

# Make a request to predictions endpoint (requires building_id, start, end)
pred_response=$(curl -s -o /dev/null -w "%{http_code}" \
    "$BACKEND_URL/api/analytics/predictions?building_id=$BUILDING_ID&start=-5m&end=now()" 2>/dev/null || echo "000")

if [ "$pred_response" = "200" ]; then
    echo -e "${GREEN}OK${NC} (HTTP 200)"

    # Try to get actual data
    pred_data=$(curl -s "$BACKEND_URL/api/analytics/predictions?building_id=$BUILDING_ID&start=-5m&end=now()" 2>/dev/null || echo "{}")
    pred_count=$(echo "$pred_data" | grep -o '"count":[0-9]*' | grep -o '[0-9]*' || echo "0")

    if [ "$pred_count" -gt 0 ]; then
        echo -e "  ${GREEN}Found $pred_count predictions${NC}"
    else
        echo -e "  ${YELLOW}No predictions yet (inference may still be warming up)${NC}"
    fi
elif [ "$pred_response" = "401" ] || [ "$pred_response" = "403" ]; then
    echo -e "${YELLOW}AUTH REQUIRED${NC} (HTTP $pred_response) - endpoint is protected"
else
    echo -e "${RED}FAILED${NC} (HTTP $pred_response)"
fi

# 5. Docker Compose services status
echo ""
echo "5. Docker Compose Services"
echo "--------------------------"
docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || echo "Could not get docker compose status"

# Summary
echo ""
echo "=============================================="
echo "Smoke Test Complete"
echo "=============================================="
echo ""
echo "Expected success signals:"
echo "  - /live and /ready return 200"
echo "  - /api/models returns 200 with model list"
echo "  - Redis window length grows and caps at 3600"
echo "  - Predictions endpoint accessible (may require auth)"
echo ""
echo "If predictions are not appearing:"
echo "  1. Wait for simulator to send enough data (window warmup)"
echo "  2. Check worker logs: docker compose logs worker"
echo "  3. Check simulator logs: docker compose logs simulator"
echo ""