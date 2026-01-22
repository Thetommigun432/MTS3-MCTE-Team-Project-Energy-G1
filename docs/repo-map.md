# Repository Map

**Last Updated:** 2026-01-22
**Branch:** frontend (synced with backend)

## Architecture Overview

The NILM Energy Monitor is a monorepo containing:

```
┌─────────────────────────────────────────────────────────────────┐
│                       Production Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│   ┌─────────────────┐          ┌─────────────────┐               │
│   │  Cloudflare     │          │    Railway      │               │
│   │  Pages          │ ◄─────── │    Backend      │               │
│   │  (React SPA)    │  HTTPS   │    (FastAPI)    │               │
│   └────────┬────────┘          └────────┬────────┘               │
│            │                            │                         │
│            │                            │                         │
│            ▼                            ▼                         │
│   ┌─────────────────┐          ┌─────────────────┐               │
│   │    Supabase     │          │    InfluxDB     │               │
│   │    (Auth/DB)    │          │  (Time-Series)  │               │
│   └─────────────────┘          └─────────────────┘               │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

| Component | Location | Deployment | Port |
|-----------|----------|------------|------|
| Frontend (React/Vite) | `apps/web` | Cloudflare Pages | 8080 (dev) |
| Backend (FastAPI) | `apps/backend` | Railway | 8000 |
| InfluxDB | Docker (local) / External (prod) | Railway add-on or external | 8086 |
| Supabase | Cloud | supabase.co | N/A |

---

## Repository Structure

```
MTS3-MCTE-Team-Project-Energy-G1/
│
├── apps/                           # Application code
│   ├── backend/                    # FastAPI Python backend
│   │   ├── app/
│   │   │   ├── api/               # HTTP layer
│   │   │   │   ├── deps.py        # Dependency injection
│   │   │   │   ├── middleware.py  # Rate limiting, metrics, request ID
│   │   │   │   └── routers/       # Endpoint definitions
│   │   │   │       ├── admin.py   # Admin endpoints (/admin/*)
│   │   │   │       ├── analytics.py # Data queries (/analytics/*)
│   │   │   │       ├── health.py  # Health checks (/live, /ready)
│   │   │   │       └── inference.py # ML inference (/infer, /models)
│   │   │   ├── core/              # Core utilities
│   │   │   │   ├── config.py      # Settings (pydantic-settings)
│   │   │   │   ├── errors.py      # Error codes and handlers
│   │   │   │   ├── logging.py     # Structured JSON logging
│   │   │   │   ├── security.py    # JWT verification (RS256/HS256)
│   │   │   │   └── telemetry.py   # Prometheus metrics
│   │   │   ├── domain/            # Business logic
│   │   │   │   ├── authz/         # Authorization service
│   │   │   │   └── inference/     # ML model registry & engine
│   │   │   ├── infra/             # External services
│   │   │   │   ├── influx/        # InfluxDB client
│   │   │   │   ├── redis/         # Redis cache (optional)
│   │   │   │   └── supabase/      # Supabase client
│   │   │   ├── schemas/           # Pydantic models (request/response)
│   │   │   └── main.py            # App entry point
│   │   ├── models/                # ML model artifacts (.safetensors)
│   │   ├── tests/                 # Backend tests
│   │   ├── Dockerfile             # Railway deployment
│   │   ├── requirements.txt       # Python dependencies
│   │   └── .env.example           # Environment template
│   │
│   └── web/                        # React/Vite frontend
│       ├── public/                # Static assets
│       │   ├── _redirects         # SPA routing for Cloudflare
│       │   ├── _headers           # Security headers
│       │   └── data/              # Demo CSV data
│       ├── src/
│       │   ├── components/        # React components
│       │   │   ├── auth/          # Auth guards
│       │   │   ├── layout/        # App shell, sidebar, navbar
│       │   │   ├── nilm/          # NILM-specific visualizations
│       │   │   └── ui/            # shadcn/ui components
│       │   ├── contexts/          # React contexts
│       │   │   ├── AuthContext.tsx    # Supabase auth
│       │   │   ├── EnergyContext.tsx  # Energy data state
│       │   │   └── ThemeContext.tsx   # Dark/light mode
│       │   ├── hooks/             # Custom hooks
│       │   ├── lib/               # Utilities
│       │   │   └── env.ts         # Environment config
│       │   ├── pages/             # Route components
│       │   │   ├── app/           # Protected app pages
│       │   │   └── auth/          # Auth pages (login, signup)
│       │   ├── services/          # API clients
│       │   │   ├── api.ts         # Generic HTTP client
│       │   │   └── energy.ts      # Energy-specific endpoints
│       │   ├── App.tsx            # Router and providers
│       │   ├── main.tsx           # Entry point
│       │   └── config.ts          # Re-exports env config
│       ├── package.json
│       ├── vite.config.ts         # Vite + dev proxy config
│       ├── wrangler.toml          # Cloudflare Pages config
│       └── .env.example           # Environment template
│
├── docs/                           # Documentation
│   ├── README.md                  # Documentation index
│   ├── getting-started.md         # Local dev setup
│   ├── backend.md                 # Backend architecture
│   ├── frontend.md                # Frontend architecture
│   ├── influx.md                  # InfluxDB schema
│   ├── supabase.md                # Supabase auth setup
│   ├── deployment/
│   │   ├── railway.md             # Railway (backend) deployment
│   │   └── cloudflare.md          # Cloudflare Pages deployment
│   └── integration-audit.md       # Integration status
│
├── scripts/                        # Data seeding utilities
│   ├── write-to-influx.ts         # Seed InfluxDB with predictions
│   └── verify-influx.ts           # Verify InfluxDB setup
│
├── training/                       # ML training code
│   └── README.md                  # Training instructions
│
├── supabase/                       # Supabase config (migrations, seeds)
│
├── data/                           # Data files (gitignored)
│   └── processed/                 # Pre-processed datasets
│
├── compose.yaml                    # Docker Compose (dev)
├── railway.json                    # Railway deployment config
├── package.json                    # Root package (workspaces)
├── .env.example                    # Root env vars (InfluxDB, Supabase)
└── .env.local.example              # Local dev env template
```

---

## Key Files Explained

### Backend (`apps/backend`)

| File | Purpose |
|------|---------|
| `app/main.py` | FastAPI app creation, lifespan management, middleware stack |
| `app/core/config.py` | All settings from environment variables (pydantic-settings) |
| `app/core/security.py` | JWT verification supporting RS256 (JWKS) and HS256 |
| `app/api/routers/health.py` | `/live` (liveness) and `/ready` (readiness with dependency checks) |
| `app/api/routers/inference.py` | `/infer` (run prediction), `/models` (list models) |
| `app/api/routers/analytics.py` | `/analytics/readings`, `/analytics/predictions` |
| `app/infra/influx/client.py` | Async InfluxDB client with retry logic |
| `app/domain/inference/registry.py` | ML model registry (loads from JSON manifest) |
| `Dockerfile` | Multi-stage Docker build for Railway |

### Frontend (`apps/web`)

| File | Purpose |
|------|---------|
| `src/main.tsx` | React entry point |
| `src/App.tsx` | Router setup, lazy-loaded routes, providers |
| `src/lib/env.ts` | Environment variable parsing with backward compatibility |
| `src/services/api.ts` | Generic HTTP client with auth token injection |
| `src/services/energy.ts` | Typed API for energy endpoints |
| `src/contexts/AuthContext.tsx` | Supabase authentication state |
| `src/contexts/EnergyContext.tsx` | Energy data state management |
| `vite.config.ts` | Dev server proxy to backend |
| `public/_redirects` | SPA routing for Cloudflare Pages |

### Root Configuration

| File | Purpose |
|------|---------|
| `compose.yaml` | Docker Compose for local dev (InfluxDB + backend) |
| `railway.json` | Railway deployment config (Dockerfile path, health checks) |
| `package.json` | npm workspaces root |
| `.env.example` | Root env vars (shared by compose.yaml) |

---

## Environment Variables

### Frontend (`apps/web/.env`)

| Variable | Required | Description |
|----------|----------|-------------|
| `VITE_SUPABASE_URL` | Yes* | Supabase project URL |
| `VITE_SUPABASE_ANON_KEY` | Yes* | Supabase anonymous/public key |
| `VITE_BACKEND_URL` | No | Backend URL (empty for local dev with proxy) |
| `VITE_DEMO_MODE` | No | Enable demo login UI (default: false) |
| `VITE_LOCAL_MODE` | No | Use local InfluxDB (default: false) |

*Not required if `VITE_DEMO_MODE=true` or `VITE_LOCAL_MODE=true`

### Backend (`apps/backend/.env`)

| Variable | Required | Description |
|----------|----------|-------------|
| `ENV` | No | Environment: dev, test, prod (default: dev) |
| `PORT` | No | Server port (default: 8000, Railway injects this) |
| `CORS_ORIGINS` | Yes | Comma-separated allowed origins |
| `INFLUX_URL` | Yes | InfluxDB URL |
| `INFLUX_TOKEN` | Yes | InfluxDB admin token |
| `INFLUX_ORG` | No | InfluxDB organization (default: energy-monitor) |
| `INFLUX_BUCKET_RAW` | No | Raw data bucket (default: raw_sensor_data) |
| `INFLUX_BUCKET_PRED` | No | Predictions bucket (default: predictions) |
| `SUPABASE_URL` | Yes (prod) | Supabase project URL |
| `SUPABASE_ANON_KEY` | Yes (prod) | Supabase public key |
| `SUPABASE_JWT_SECRET` | No | Legacy HS256 secret (prefer JWKS) |
| `ADMIN_TOKEN` | Yes (prod) | Token for admin endpoints |

### Root (`.env.local`)

Used by `compose.yaml`:

| Variable | Required | Description |
|----------|----------|-------------|
| `INFLUX_TOKEN` | Yes | Shared by InfluxDB and backend container |
| `SUPABASE_URL` | No | Passed to backend container |
| `SUPABASE_ANON_KEY` | No | Passed to backend container |

---

## API Contract

### Health Endpoints (No Auth)

| Endpoint | Method | Response |
|----------|--------|----------|
| `GET /live` | No auth | `{"status": "ok"}` |
| `GET /ready` | No auth | `{"status": "ok", "checks": {...}}` |
| `GET /health` | No auth | Environment details (limited in prod) |
| `GET /metrics` | No auth | Prometheus metrics |

### Analytics Endpoints (JWT Required)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /analytics/readings` | JWT | Query raw sensor data |
| `GET /analytics/predictions` | JWT | Query predictions |

Query Parameters:
- `building_id` (required): Building identifier
- `start` (required): Start time (ISO8601 or relative like "-7d")
- `end` (required): End time (ISO8601 or relative like "now()")
- `resolution` (optional): "1s", "1m", "15m"
- `appliance_id` (optional): Filter by appliance

### Inference Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `GET /models` | No auth | List available ML models |
| `POST /infer` | JWT | Run inference and persist result |

### Admin Endpoints (JWT + Admin Token)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /admin/reload-models` | JWT + X-Admin-Token | Reload model registry |
| `POST /admin/cache/invalidate` | JWT + X-Admin-Token | Clear cache |
| `GET /admin/cache/stats` | JWT + X-Admin-Token | Cache statistics |

---

## Where to Start

### Frontend Development

```bash
# 1. Install dependencies
cd apps/web
npm install

# 2. Set up environment
cp .env.example .env
# Edit .env with your Supabase credentials OR set VITE_DEMO_MODE=true

# 3. Start dev server (will proxy /api to localhost:8000)
npm run dev
```

Frontend is available at `http://localhost:8080`

### Backend Development (Docker)

```bash
# 1. Set up root environment
cp .env.local.example .env.local
# Edit .env.local - at minimum set INFLUX_TOKEN

# 2. Start services
docker compose up -d

# 3. Verify
curl http://localhost:8000/live
```

### Backend Development (Local Python)

```bash
cd apps/backend

# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

# 2. Install dependencies
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 3. Set up environment
cp .env.example .env

# 4. Start InfluxDB (required)
docker compose up -d influxdb influxdb-init

# 5. Run server
uvicorn app.main:app --reload --port 8000
```

### End-to-End Local Test

```bash
# 1. Start backend stack
docker compose up -d

# 2. Verify backend health
curl http://localhost:8000/ready

# 3. Seed test data (optional)
npm run predictions:seed

# 4. Start frontend
cd apps/web && npm run dev

# 5. Open http://localhost:8080
```

### Production Deployment

**Backend (Railway):**
- Connect GitHub repo to Railway
- Set root directory to repo root
- Railway uses `railway.json` which points to `apps/backend/Dockerfile`
- Set required environment variables in Railway dashboard

**Frontend (Cloudflare Pages):**
- Connect GitHub repo to Cloudflare Pages
- Build command: `npm run build --workspace=apps/web`
- Build output directory: `apps/web/dist`
- Set environment variables for Supabase and backend URL

---

## Branch Conventions

| Branch | Purpose |
|--------|---------|
| `main` | ML training code and data preprocessing |
| `frontend` | Full-stack development (frontend + backend) |
| `backend` | Full-stack development (synced with frontend) |

The `frontend` and `backend` branches contain the same code and should be kept in sync. The `main` branch has different content focused on ML model training.

---

## Testing

### Frontend

```bash
cd apps/web

# Run tests
npm run test

# Type check
npm run typecheck

# Lint
npm run lint

# Build
npm run build
```

### Backend

```bash
cd apps/backend

# Run tests (requires Docker for InfluxDB or mocking)
pytest

# Or run in Docker
docker run --rm nilm-backend pytest
```

---

## Common Issues

1. **"supabaseKey is required" in tests**: Tests import modules that initialize Supabase client. Set `VITE_DEMO_MODE=true` in test setup or mock the client.

2. **CORS errors in browser**: Ensure `CORS_ORIGINS` in backend includes your frontend origin.

3. **InfluxDB connection failed**: Ensure InfluxDB is running and `INFLUX_TOKEN` matches between compose.yaml and .env.local.

4. **JWT verification failed**: Check `SUPABASE_URL` is set correctly. For RS256, ensure JWKS endpoint is accessible.

5. **Build fails on Railway**: Ensure Dockerfile paths use `apps/backend/` prefix as Railway builds from repo root.
