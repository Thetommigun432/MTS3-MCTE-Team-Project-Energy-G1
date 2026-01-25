# Testing Guide - NILM Energy Monitor

This document covers the test architecture, running tests, and writing new tests for the NILM Energy Monitor project.

---

## Test Architecture

```
apps/backend/tests/
├── __init__.py
├── conftest.py              # Shared fixtures (env vars, mock clients)
├── fixtures/                # Test data (parquet files, numpy samples)
│   ├── *.parquet           # Sample building data (~21MB)
│   └── *.npy               # Sample model inputs
├── unit/                    # Pure unit tests (no I/O, no mocks)
│   ├── test_validation.py
│   ├── test_flux_builder.py
│   ├── test_data_ingestion.py
│   ├── test_model_registry.py
│   ├── test_producer_schema.py
│   ├── test_influx_writer.py
│   └── test_railway_config.py
├── component/               # API tests with mocked dependencies
│   ├── test_health_api.py
│   ├── test_auth_jwks.py
│   ├── test_inference_pipeline.py
│   ├── test_models_api.py
│   ├── test_analytics_api.py
│   └── test_ingest_api.py
└── e2e/                     # Full stack tests (Docker required)
    └── test_e2e_flow.py
```

### Test Markers

| Marker | Description | Dependencies | Speed |
|--------|-------------|--------------|-------|
| `unit` | Pure unit tests | None | Fast (~5s) |
| `component` | API tests with mocks | FastAPI TestClient | Fast (~10s) |
| `e2e` | Full pipeline tests | Docker stack | Slow (~2-3min) |
| `railway` | Deployed service smoke tests | Railway URL | Fast (~10s) |

---

## Running Tests

### Prerequisites

```bash
cd apps/backend
pip install -e .
pip install pytest pytest-asyncio pytest-cov
```

### Unit Tests (No Dependencies)

```bash
# Run all unit tests
pytest tests/unit -v -m unit

# Run specific test file
pytest tests/unit/test_model_registry.py -v

# Run with coverage
pytest tests/unit -v -m unit --cov=app --cov-report=term-missing
```

### Component Tests (Mocked Dependencies)

```bash
# Run all component tests
pytest tests/component -v -m component

# Run specific API tests
pytest tests/component/test_models_api.py -v
```

### All Fast Tests (CI Default)

```bash
# Run unit + component tests (excludes e2e and railway)
pytest tests/ -v -m "not e2e and not railway" --tb=short
```

### E2E Tests (Docker Stack Required)

```bash
# 1. Start the full stack
docker compose -f compose.yaml -f compose.e2e.yaml up -d --build

# 2. Wait for services to be ready
timeout 120s bash -c 'until curl -sf http://localhost:8000/live; do sleep 2; done'

# 3. Run E2E tests
pytest tests/e2e -v -m e2e

# 4. Cleanup
docker compose down -v
```

Or use the one-shot script:

```bash
./scripts/e2e.sh
```

### Railway Smoke Tests (Deployed Service)

```bash
# Set the Railway URL
export RAILWAY_BACKEND_URL="https://your-service.railway.app"

# Run smoke tests
pytest tests/ -v -m railway
```

---

## Frontend Tests

### Structure

```
apps/web/src/
├── test/
│   ├── setup.ts        # Vitest setup + MSW lifecycle
│   ├── utils.tsx       # renderWithProviders helper
│   └── mocks/
│       ├── handlers.ts # MSW request handlers
│       └── server.ts   # MSW server setup
├── lib/*.test.ts       # Utility tests
├── hooks/*.test.ts     # Hook tests
└── services/*.test.ts  # Service tests
```

### Running Frontend Tests

```bash
cd apps/web

# Run all tests
npm test

# Watch mode
npm run test:watch

# With coverage
npm run test:coverage
```

### MSW Mocking

Tests use [MSW](https://mswjs.io/) to mock API requests:

```tsx
import { server } from '@/test/mocks/server';
import { http, HttpResponse } from 'msw';

it('handles API error', async () => {
  // Override handler for this test
  server.use(
    http.get('http://localhost:8000/live', () => {
      return HttpResponse.json({ error: 'fail' }, { status: 500 });
    })
  );
  
  // ... test error handling
});
```

### Test Utilities

```tsx
import { renderWithProviders } from '@/test/utils';

it('renders with router', () => {
  renderWithProviders(<MyComponent />, { route: '/dashboard' });
});
```

---


## CI Workflows

| Workflow | Trigger | Tests Run | Duration |
|----------|---------|-----------|----------|
| `ci.yml` | PR, push to main/integration | Unit + Component | ~2 min |
| `e2e.yml` | Push to integration, nightly | Full E2E pipeline | ~10 min |
| `railway-smoke.yml` | Daily 6 AM UTC, manual | Health checks | ~1 min |

### Workflow Details

**ci.yml** - Fast feedback on every PR:
- Python 3.12 setup with pip caching
- Runs `pytest -m "not e2e and not railway"`
- Uploads coverage to Codecov
- Also runs frontend lint/typecheck/build

**e2e.yml** - Full pipeline validation:
- Builds and starts Docker stack
- Runs producer → Redis → inference → InfluxDB → API flow
- Uses `E2E_RUN_ID` for test isolation in CI
- Only runs on push (not PRs) to protect secrets

**railway-smoke.yml** - Production health monitoring:
- Tests `/live`, `/ready`, `/models`, `/openapi.json`
- Includes retry logic with exponential backoff
- Reports response times and warns if > 2 seconds

---

## Writing New Tests

### Unit Test Pattern

```python
"""Unit tests for [module name]."""
import pytest

class TestFeatureName:
    """Tests for specific feature."""

    @pytest.mark.unit
    def test_specific_behavior(self):
        """Description of what's being tested."""
        # Arrange
        input_data = {...}

        # Act
        result = function_under_test(input_data)

        # Assert
        assert result == expected_value
```

### Component Test Pattern (API Testing)

```python
"""Component tests for [endpoint]."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock

@pytest.fixture
def mock_dependency():
    """Mock external dependency."""
    client = MagicMock()
    client.method = AsyncMock(return_value=expected)
    return client

@pytest.fixture
def test_client(mock_dependency):
    """Create test client with mocked dependencies."""
    with patch("app.module.get_dependency", return_value=mock_dependency):
        from app.main import app
        with TestClient(app) as client:
            yield client

class TestEndpointName:
    """Tests for specific endpoint."""

    @pytest.mark.component
    def test_returns_200(self, test_client):
        """Endpoint returns 200 OK."""
        response = test_client.get("/endpoint")
        assert response.status_code == 200

    @pytest.mark.component
    def test_response_shape(self, test_client):
        """Response has correct schema."""
        response = test_client.get("/endpoint")
        data = response.json()
        assert "expected_field" in data
```

### E2E Test Pattern

```python
"""E2E tests for [flow name]."""
import pytest
import os

# Environment configuration
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
E2E_RUN_ID = os.environ.get("E2E_RUN_ID", "local")

class TestPipelineFlow:
    """Tests for end-to-end data flow."""

    @pytest.mark.e2e
    def test_data_flows_through_pipeline(self):
        """Data flows from ingestion to API."""
        # Use E2E_RUN_ID for test isolation
        # Query with run_id filter to avoid cross-test pollution
```

---

## Debugging Failed Tests

### Unit Test Failures

```bash
# Run with verbose output and full tracebacks
pytest tests/unit/test_file.py -v --tb=long -s

# Run specific test
pytest tests/unit/test_file.py::TestClass::test_method -v
```

### Component Test Failures

```bash
# Check mock setup
pytest tests/component/test_file.py -v --tb=long

# Add print statements in fixtures to debug
```

### E2E Test Failures

```bash
# Check service logs
docker compose -f compose.yaml -f compose.e2e.yaml logs backend
docker compose -f compose.yaml -f compose.e2e.yaml logs nilm-inference
docker compose -f compose.yaml -f compose.e2e.yaml logs nilm-persister

# Check InfluxDB data
docker compose exec influxdb influx query 'from(bucket:"predictions") |> range(start: -1h)'

# Check Redis streams
docker compose exec redis redis-cli XLEN nilm:readings
```

### CI-Specific Issues

- **E2E_RUN_ID**: CI sets unique run IDs to isolate test data
- **Timing**: E2E tests wait for services; increase timeout if needed
- **Secrets**: E2E doesn't run on PRs from forks (no secrets access)

---

## Best Practices

1. **Use appropriate markers**: Always decorate tests with `@pytest.mark.unit`, `@pytest.mark.component`, or `@pytest.mark.e2e`

2. **Mock at boundaries**: Component tests mock external services (InfluxDB, Redis), not internal logic

3. **Test one thing**: Each test should verify a single behavior

4. **Descriptive names**: Test names should describe the expected behavior

5. **Fast feedback**: Prefer unit tests over component tests, component over E2E

6. **Isolate E2E tests**: Use `E2E_RUN_ID` to filter data and avoid cross-test pollution

7. **Don't test implementation**: Test behavior and contracts, not internal details

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -e .` in apps/backend |
| Redis connection refused | Start Docker stack or mock Redis |
| InfluxDB bucket not found | Check `influxdb-init` logs |
| E2E test hangs | Increase timeout, check service health |
| Flaky E2E test | Add `E2E_RUN_ID` filter to Flux queries |

### Service Health Checks

```bash
# Backend
curl http://localhost:8000/live
curl http://localhost:8000/ready

# InfluxDB
curl http://localhost:8086/health

# Redis
docker compose exec redis redis-cli ping
```

---

## Railway Deployment Verification

To ensure the project is ready for Railway deployment:

```bash
./scripts/verify_railway.sh
```

This script verifies:
1. **Build Context**: `apps/backend/Dockerfile` builds from root context
2. **Frontend**: Standalone frontend build succeeds
3. **Health Endpoints**: `/live` and `/ready` respond correctly

### Deployment Checklist

- [x] `railway.json` present with correct healthcheckPath
- [x] Backend Dockerfile uses root-relative paths
- [x] Backend Docker build succeeds from root
- [x] Health endpoints return correct format
