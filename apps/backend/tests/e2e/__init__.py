"""End-to-end tests for the NILM pipeline.

These tests require a running Docker stack with all services.
Run with: pytest -m e2e -v

Environment variables:
- INFLUX_URL: InfluxDB URL (default: http://localhost:8086)
- INFLUX_TOKEN: InfluxDB token
- INFLUX_ORG: InfluxDB organization
- INFLUX_BUCKET_PRED: Predictions bucket name
- BACKEND_URL: Backend API URL (default: http://localhost:8000)
- E2E_RUN_ID: Unique run ID for test isolation in CI
"""
