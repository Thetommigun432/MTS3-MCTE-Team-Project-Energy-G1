# NILM Energy Monitor - Unified Python Backend

Production-grade FastAPI backend for energy monitoring and NILM (Non-Intrusive Load Monitoring) inference.

## Features

- **Unified Service**: Replaces split Node.js + Python backend with a single Python monolith
- **JWT Authentication**: Supabase-compatible HS256/RS256 verification
- **RBAC Authorization**: Building and appliance-level access control with caching
- **Predict & Persist**: Strategy A - predictions are persisted before returning success
- **Safe Queries**: Template-based Flux queries with strict input validation
- **Observability**: Prometheus metrics, structured JSON logging, request tracing
- **Docker & Railway Ready**: Production Dockerfile with Railway PORT binding

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.12+ (for local development)
- InfluxDB token and Supabase credentials

### Local Development with Docker

```bash
# Clone and navigate to repo root
cd MTS3-MCTE-Team-Project-Energy-G1

# Create .env from example
cp apps/backend/.env.example .env

# Edit .env with your credentials
# Required: INFLUX_TOKEN, SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_JWT_SECRET

# Start all services
docker compose up -d

# Check backend health
curl http://localhost:8000/live
curl http://localhost:8000/ready
```

### Local Development without Docker

```bash
cd apps/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\Activate   # Windows

# Install dependencies
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your values

# Run server
uvicorn app.main:app --reload --port 8000
```

## API Reference

### Health Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /live` | Liveness probe (process up) |
| `GET /ready` | Readiness probe (dependencies available) |
| `GET /health` | Rich health info (dev only) |
| `GET /metrics` | Prometheus metrics |

### Inference Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /infer` | Run inference and persist prediction |
| `GET /models` | List available models |

### Analytics Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /analytics/readings` | Query sensor readings |
| `GET /analytics/predictions` | Query predictions |

### Admin Endpoints (requires admin role)

| Endpoint | Description |
|----------|-------------|
| `POST /admin/reload-models` | Reload model registry |
| `POST /admin/cache/invalidate` | Invalidate authz cache |
| `GET /admin/cache/stats` | Cache statistics |

## POST /infer Example

```bash
curl -X POST http://localhost:8000/infer \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: unique-request-id" \
  -d '{
    "building_id": "bldg_123",
    "appliance_id": "heatpump",
    "window": [1.0, 1.1, 1.2, ...],
    "timestamp": "2024-01-15T12:00:00Z"
  }'

# Response (200 OK)
{
  "predicted_kw": 1.25,
  "confidence": 0.85,
  "model_version": "v1.0.0",
  "request_id": "req_abc123",
  "persisted": true
}
```

## Railway Deployment

### Environment Variables

Set these in your Railway service:

| Variable | Required | Description |
|----------|----------|-------------|
| `ENV` | Yes | `prod` |
| `PORT` | Auto | Railway sets automatically |
| `INFLUX_URL` | Yes | Hosted InfluxDB URL (https) |
| `INFLUX_TOKEN` | Yes | InfluxDB admin token |
| `INFLUX_ORG` | Yes | InfluxDB organization |
| `INFLUX_BUCKET_RAW` | Yes | Raw data bucket |
| `INFLUX_BUCKET_PRED` | Yes | Predictions bucket |
| `SUPABASE_URL` | Yes | Supabase project URL |
| `SUPABASE_ANON_KEY` | Yes | Supabase anon key |
| `SUPABASE_JWT_SECRET` | Yes | JWT secret (HS256) |
| `CORS_ORIGINS` | Yes | Frontend origin(s) |
| `ADMIN_TOKEN` | Rec | Admin API token |

### Deploy Steps

1. Create new Railway service
2. Set root directory: `apps/backend`
3. Railway auto-detects Dockerfile
4. Set all required environment variables
5. Deploy

### Health Check

Railway considers the service healthy when `/live` returns 200.

## Model Registry

Models are defined in `models/registry.json`:

```json
{
  "models": [
    {
      "model_id": "heatpump-cnntr-v1",
      "model_version": "v1.0.0",
      "appliance_id": "heatpump",
      "architecture": "CNNTransformer",
      "architecture_params": {
        "input_channels": 1,
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2
      },
      "artifact_path": "heatpump_cnntr.safetensors",
      "artifact_sha256": "...",
      "input_window_size": 1000,
      "preprocessing": {
        "type": "standard",
        "mean": 2.5,
        "std": 1.5
      },
      "is_active": true
    }
  ]
}
```

### Adding a New Model

1. Train model and export to safetensors format
2. Generate SHA256: `sha256sum model.safetensors`
3. Add entry to `models/registry.json`
4. Copy artifact to `models/` directory
5. Call `POST /admin/reload-models`

## Architecture

```
apps/backend/
├── app/
│   ├── main.py              # FastAPI app with lifespan
│   ├── core/                # Config, logging, errors, security
│   ├── api/                 # Routes and middleware
│   │   ├── deps.py          # Dependency injection
│   │   ├── middleware.py    # Rate limit, request ID
│   │   └── routers/         # Endpoint handlers
│   ├── domain/              # Business logic
│   │   ├── inference/       # Model loading & inference
│   │   └── authz/           # Authorization policy
│   ├── infra/               # External services
│   │   ├── influx/          # InfluxDB client
│   │   └── supabase/        # Supabase client
│   └── schemas/             # Pydantic models
├── models/                  # Model artifacts
├── Dockerfile
└── requirements.txt
```

## Testing

```bash
cd apps/backend

# Run unit tests
pytest app/tests/unit/ -v

# Run with coverage
pytest app/tests/ -v --cov=app
```

## Supabase Table Mapping

The backend uses these Supabase tables for authorization:

| Table | Purpose |
|-------|---------|
| `buildings` | User owns buildings via `user_id` |
| `building_appliances` | Links buildings to org_appliances |
| `org_appliances` | Appliance templates |
| `profiles` | User profiles (optional role field) |

## Security

- JWT verification: HS256 (default) or RS256 (via JWKS)
- Rate limiting: 60/min per user, 120/min per IP
- Request size limit: 256KB default
- Safetensors-only model loading (no pickle)
- Template-based Flux queries (no injection)
