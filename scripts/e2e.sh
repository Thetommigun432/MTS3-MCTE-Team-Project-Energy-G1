
#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Starting E2E Test Suite...${NC}"

# 1. Setup Env
export INFLUX_TOKEN=admin-token # Dummy token for test
export INFLUX_ORG=energy-monitor
export INFLUX_BUCKET_RAW=raw_sensor_data
export INFLUX_BUCKET_PRED=predictions

# 2. Generate Fixtures
echo "Generating test fixtures..."
# Run local script using docker python to avoid host dependency issues
docker run --rm -v "${PWD}:/app" -w /app python:3.11-slim sh -c "pip install pandas pyarrow numpy && python scripts/generate_fixtures.py"

# 3. Start Stack
echo "Starting Docker Stack (e2e profile)..."
docker compose -f compose.e2e.yaml up -d --build --remove-orphans

# Function to clean up on exit
cleanup() {
    echo "Stopping Docker Stack..."
    docker compose -f compose.e2e.yaml down -v
}
trap cleanup EXIT

# 4. Wait for healthy
echo "Waiting for services to be healthy..."
# We wait for influxdb specifically; others depend on it
timeout=60
while [ $timeout -gt 0 ]; do
    if docker compose -f compose.e2e.yaml ps influxdb | grep "healthy" > /dev/null; then
        echo -e "${GREEN}InfluxDB is healthy.${NC}"
        break
    fi
    sleep 2
    timeout=$((timeout - 2))
done

if [ $timeout -le 0 ]; then
    echo -e "${RED}Timeout waiting for InfluxDB.${NC}"
    exit 1
fi

sleep 10 # Extra buffer for initialization

# 5. Run Backend Integration Tests (inside docker)
echo "Running Internal Verification Tests..."
docker compose -f compose.e2e.yaml run --rm e2e-tests

# 6. Run Playwright Tests (Frontend)
echo "Running Frontend Tests..."
cd tests/e2e && npm install && npx playwright test

echo -e "${GREEN}E2E Tests Passed!${NC}"
