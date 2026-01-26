# Repository Map

This document provides an overview of the NILM Energy Monitor monorepo structure.

## Top-Level Structure

```
/
├── apps/                    # Application code
│   ├── backend/             # FastAPI backend (Python 3.12)
│   └── web/                 # React frontend (Vite + TypeScript)
├── checkpoints/             # ML training checkpoints (not deployed)
├── data/                    # Local data files (gitignored)
├── docs/                    # Documentation
├── production/              # DEPRECATED - legacy files
├── scripts/                 # Utility scripts
├── supabase/                # Supabase configuration and migrations
├── tests/                   # Integration tests
└── training/                # ML training code
```

## /apps/backend (FastAPI)

The canonical NILM inference backend:

```
apps/backend/
├── src/app/
│   ├── main.py              # FastAPI application entry
│   ├── worker_main.py       # Redis inference worker entry
│   ├── api/                 # REST API routes
│   ├── domain/              # Business logic
│   │   └── inference/       # Model registry + engine
│   ├── infra/               # External services (Redis, Influx, Supabase)
│   ├── pipeline/            # Streaming pipeline workers
│   └── schemas/             # Pydantic models
├── models/                  # Model artifacts (.safetensors)
├── tests/                   # Unit tests
├── Dockerfile               # Production image
├── railway.api.toml         # Railway API service config
└── railway.worker.toml      # Railway worker service config
```

## /apps/web (Vite React)

Frontend dashboard:

```
apps/web/
├── src/
│   ├── components/          # UI components
│   ├── pages/               # Route pages
│   ├── services/            # API clients
│   └── lib/                 # Utilities
├── public/                  # Static assets
└── Dockerfile               # Production nginx image
```

## Key Entry Points

| Purpose | File | Start Command |
|---------|------|---------------|
| API Server | `apps/backend/src/app/main.py` | `uvicorn app.main:app` |
| Inference Worker | `apps/backend/src/app/worker_main.py` | `python -m app.worker_main` |
| Frontend Dev | `apps/web/` | `npm run dev` |

## Deprecated Folders

| Folder | Status | Notes |
|--------|--------|-------|
| `/production/` | DEPRECATED | Legacy inference scripts, now merged into `/apps/backend/` |

## Configuration Files

| File | Purpose |
|------|---------|
| `railway.json` | Root Railway config (API service) |
| `compose.yaml` | Local Docker Compose stack |
| `compose.e2e.yaml` | E2E test stack |
