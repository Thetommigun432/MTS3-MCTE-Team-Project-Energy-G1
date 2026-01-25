
# E2E Testing Guide

This repository contains a full End-to-End (E2E) testing harness to verify the data flow from ingestion to visualization.

## Architecture Tested

1.  **Ingestion**: `data_producer.py` generates synthetic power readings.
2.  **Transport**: Redis Streams / PubSub.
3.  **Inference**: `nilm-inference` service (simulated dummy model).
4.  **Persistence**: `nilm-persister` writes predictions to InfluxDB.
5.  **Visualization**: Frontend (Vite/React) displays data.

## Prerequisites

- Docker & Docker Compose
- Python 3.11+ (for local scripts)
- Node.js (for Playwright)

## Running Tests

### One-Shot Runner
The easiest way to run everything is:
```bash
./scripts/e2e.sh
```
This script will:
1. Generate test fixtures.
2. Spin up the specific E2E docker stack (`compose.e2e.yaml`).
3. Wait for InfluxDB health.
4. Run Backend Integration Tests (inside Docker).
5. (Optional) Run Playwright Frontend Tests.

### Manual Running

**1. Start Stack:**
```bash
docker compose -f compose.e2e.yaml up -d --build
```

**2. Run Backend Tests:**
```bash
docker compose -f compose.e2e.yaml run --rm e2e-tests
```

**3. Run Frontend Tests:**
```bash
cd tests/e2e
npm install
npx playwright test
```

## Troubleshooting

- **Redis Connection**: Ensure port 6379 is free.
- **InfluxDB Setup**: The stack uses `influxdb-init` to creating buckets. Check logs if buckets are missing:
  ```bash
  docker compose -f compose.e2e.yaml logs influxdb-init
  ```
- **Pipelines**: Check `nilm-inference` and `nilm-persister` logs for errors in data flow.
