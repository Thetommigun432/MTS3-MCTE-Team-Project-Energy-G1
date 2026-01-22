# CLAUDE.MD - NILM Energy Monitor

## Project Overview

Non-Intrusive Load Monitoring (NILM) web application with deep learning-based energy disaggregation. Predicts individual appliance power consumption from aggregate household meter readings.

**Stack:**
- Frontend: React 19 + TypeScript + Vite 7 + Tailwind CSS + shadcn/ui
- Backend: FastAPI (Python 3.12) + PyTorch
- Database: InfluxDB 2.8 (time-series), Supabase (auth + metadata)
- Deployment: Cloudflare Pages (frontend) + Railway (backend)

## Repository Structure

```
apps/
├── backend/          # FastAPI Python backend
│   ├── app/          # Source code
│   ├── models/       # ML model artifacts (.safetensors)
│   ├── Dockerfile    # Railway deployment
│   └── requirements.txt
└── web/              # React/Vite frontend
    ├── src/          # Source code
    ├── public/       # Static assets + _redirects for SPA
    └── package.json

docs/                 # Documentation
scripts/              # Data seeding utilities
compose.yaml          # Local development (Docker)
railway.json          # Railway config
```

## Branch Conventions

| Branch | Purpose |
|--------|---------|
| `main` | ML training code, preprocessing scripts |
| `frontend` | Full-stack app (frontend + backend) |
| `backend` | Full-stack app (synced with frontend) |

**Important:** `frontend` and `backend` branches have identical content. The `main` branch is different (ML training focus).

## Local Development

### Quick Start (Docker)

```bash
# 1. Set up environment
cp .env.local.example .env.local
# Edit .env.local: set INFLUX_TOKEN (min 32 chars)

# 2. Start backend + InfluxDB
docker compose up -d

# 3. Verify backend
curl http://localhost:8000/live    # Should return {"status": "ok"}
curl http://localhost:8000/ready   # Should return all checks passing

# 4. Start frontend
cd apps/web
cp .env.example .env
# Edit .env: set VITE_DEMO_MODE=true OR add Supabase credentials
npm install
npm run dev
```

### Endpoints

| Service | URL |
|---------|-----|
| Frontend | http://localhost:8080 |
| Backend API | http://localhost:8000 |
| InfluxDB UI | http://localhost:8086 |

## Environment Variables

### Frontend (`apps/web/.env`)

```env
VITE_SUPABASE_URL=https://xxx.supabase.co
VITE_SUPABASE_ANON_KEY=eyJ...
VITE_BACKEND_URL=                    # Empty for local dev (uses proxy)
VITE_DEMO_MODE=true                  # Set true to skip Supabase requirement
```

### Backend (`apps/backend/.env`)

```env
ENV=dev
PORT=8000
CORS_ORIGINS=http://localhost:8080,http://localhost:5173
INFLUX_URL=http://localhost:8086
INFLUX_TOKEN=your-token-here
INFLUX_ORG=energy-monitor
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_ANON_KEY=eyJ...
```

### Root (`.env.local`)

Used by docker-compose:

```env
INFLUX_TOKEN=your-token-here        # Must match backend
```

## Quality Gates

### Frontend

```bash
cd apps/web
npm run lint        # ESLint
npm run typecheck   # TypeScript
npm run build       # Production build
npm run test        # Vitest
```

### Backend

```bash
# In Docker
docker build -f apps/backend/Dockerfile -t backend-test .
docker run --rm backend-test pytest

# Or locally
cd apps/backend
pip install -r requirements.txt
pytest
```

## API Endpoints

### Health (No Auth)

- `GET /live` - Liveness probe
- `GET /ready` - Readiness with dependency checks
- `GET /metrics` - Prometheus metrics

### Analytics (JWT Required)

- `GET /analytics/readings?building_id=X&start=-7d&end=now()&resolution=15m`
- `GET /analytics/predictions?building_id=X&start=-7d&end=now()`

### Inference (JWT Required)

- `GET /models` - List available ML models
- `POST /infer` - Run prediction

### Admin (JWT + Admin Token)

- `POST /admin/reload-models`
- `GET /admin/cache/stats`

## Common Commands

```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f backend

# Stop services
docker compose down

# Seed InfluxDB with test data
npm run predictions:seed

# Frontend dev
cd apps/web && npm run dev

# Build frontend
npm run build

# Type check frontend
npm run typecheck
```

## Code Style

### TypeScript/React

- Functional components with hooks
- shadcn/ui component library
- Path alias: `@/` maps to `src/`
- Lazy loading for route components
- Context providers for global state

### Python/FastAPI

- pydantic-settings for configuration
- Async handlers throughout
- Structured JSON logging
- Prometheus metrics on all endpoints
- JWT verification (RS256 preferred, HS256 fallback)

## Deployment

### Railway (Backend)

- Uses `railway.json` which points to `apps/backend/Dockerfile`
- Health check: `/live`
- Set all env vars in Railway dashboard
- Port is auto-injected via `PORT` env var

### Cloudflare Pages (Frontend)

- Build command: `npm run build --workspace=apps/web`
- Output directory: `apps/web/dist`
- SPA routing handled by `public/_redirects`
- Set `VITE_*` env vars in Cloudflare dashboard

## Known Issues

1. **Test Failure**: `useNilmCsvData.test.ts` fails due to Supabase client initialization. Workaround: mock Supabase or set env vars.

2. **Stub Methods**: `energyApi.getBuildings()` and `energyApi.getAppliances()` return empty arrays - they should query Supabase directly.

3. **ESLint Warnings**: 9 unused variable warnings in frontend (non-blocking).

## Making Changes Safely

1. **Before changes:**
   - `git status` to verify clean working directory
   - Ensure you're on `frontend` or `backend` branch

2. **After changes:**
   - Run `npm run lint` and `npm run typecheck` in `apps/web`
   - Run `npm run build` to verify production build
   - Test locally with `docker compose up -d`

3. **Commits:**
   - Keep commits small and focused
   - Never commit `.env` files (only `.env.example`)
   - Never commit model weights unless intended

4. **Environment variables:**
   - Never hardcode secrets
   - Use `.env.example` files as templates
   - Production secrets go in deployment platform

## Documentation

- `docs/README.md` - Documentation index
- `docs/repo-map.md` - Complete repository structure
- `docs/integration-audit.md` - Integration status and issues
- `docs/getting-started.md` - Local development guide
- `docs/deployment/railway.md` - Backend deployment
- `docs/deployment/cloudflare.md` - Frontend deployment
