# CLAUDE.md — NILM Energy Monitor (Monorepo)

This file gives Claude persistent, repo-specific context: where things live, how to run them, how to test, and what constraints to follow.

## Hard rules
- Work **only** on branch: `integration` (no new branches).
- **Never commit secrets**. No real tokens/keys in code, docs, workflows, or logs.
- Prefer **small, verifiable** changes. After each change: run the narrowest relevant checks.
- Don’t invent new services. Fix and harden what already exists.
- **Node version is pinned** by `.nvmrc` and `package.json#engines`. CI/local must match.

---

## Repo map (high level)
- `apps/web/` — Frontend (React 19 + Vite)
- `apps/backend/` — Backend + pipeline workers (FastAPI + workers)
- `apps/backend/models/` — Model registry + artifacts (must be available to inference workers)
- `simulation_data.parquet` — 1 month simulation dataset (1 row = 1 second)
- `compose.yaml` — Local stack (canonical for dev + local demo)
- `compose.e2e.yaml` — E2E overrides (used via multi-file compose merge)
- `railway.json` — Railway config-as-code (CURRENTLY for **backend API service only**)
- `.nvmrc` — Node version source of truth
- `package-lock.json` (repo root) — npm workspace lockfile source of truth
- `.github/workflows/` — CI/E2E + smoke tests
- `docs/` — Architecture, operations, deployment notes

---

## Source of truth (do not “guess”)
- **Node version**: `.nvmrc`
- **Frontend deps lockfile**: root `package-lock.json` (npm workspaces)
- **Backend deps**: `apps/backend/pyproject.toml` (requirements.txt is legacy/duplicate if still present)
- **Vite build output**: `apps/web/dist`
- **Pipeline behavior contract**: “Realtime pipeline contract” section below

---

## Quick commands

### Frontend (Vite/React) — from repo root
Prereq: Node from `.nvmrc` (use `nvm use` if available)

- Install: `npm ci`
- Dev: `npm --workspace apps/web run dev`
- Build: `npm run build:web`
- Lint: `npm run lint:web`
- Typecheck: `npm run typecheck:web`
- Tests: `npm run test:web`

Notes:
- `npm ci` must be run from repo root (workspaces + root lockfile).
- Frontend build produces `apps/web/dist/` and copies `dist/index.html` to `dist/404.html` (see `apps/web/package.json`).

### Backend (FastAPI) — from `apps/backend`
Preferred (pyproject-based):
- Create venv: `python -m venv .venv`
- Activate (Linux/macOS): `source .venv/bin/activate`
- Activate (Windows): `.venv\Scripts\activate`
- Install (editable): `python -m pip install -e .`
- Run (dev): `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`
- Tests (fast): `pytest -m "not e2e" -q`
- Tests (all): `pytest -q`

---

## Docker / Compose (local)

### Start/stop stack (canonical)
- Start: `docker compose -f compose.yaml up -d --build`
- Logs: `docker compose -f compose.yaml logs -f --tail=200`
- Stop (wipe volumes): `docker compose -f compose.yaml down -v`

### E2E stack (multi-file merge; no `include:`)
- Start E2E stack:
  `docker compose -f compose.yaml -f compose.e2e.yaml up -d --build --remove-orphans`
- Run E2E tests (wherever pytest is available):
  `pytest -m e2e -v`
- Teardown:
  `docker compose -f compose.yaml -f compose.e2e.yaml down -v --remove-orphans`

Compose merge rule of thumb:
- First file = base services (`compose.yaml`)
- Second file = overrides for E2E (`compose.e2e.yaml`)
- Order matters.

---

## Realtime pipeline contract (MUST MATCH)

### Dataset + rate
- `simulation_data.parquet` represents ~1 month of sensor data.
- **1 row = 1 second** (1 Hz).
- Producer must publish **exactly 1 sample per second** in wall-clock time (no bursts).

### Redis input buffer (strict rolling 1 hour)
- Redis must hold **exactly the last 3600 samples** (1 hour * 1 Hz).
- Once full: for every new sample added, **exactly one oldest** sample is removed (ring-buffer semantics).
- Buffer behavior must be deterministic and observable (periodic logs/metrics of buffer length).

### Inference cadence
- Inference must produce **exactly 1 prediction per second** (1 Hz), aligned with the input sample’s timestamp.
- If inference needs a window (warmup), define one consistent behavior:
  - either “skip until warmup complete” (persister must tolerate missing predictions)
  - OR “emit warmup predictions” with an explicit flag/low-confidence field

### Influx write cadence + frontend realtime
- Predictions must be written to InfluxDB at **1 Hz** (one point per second).
- Frontend visualizes near-real-time:
  - backend endpoints must support fetching last N seconds quickly (typical: 60–300 seconds)
  - avoid forced aggregation that destroys 1-second resolution

### Conceptual architecture (do not change without explicit requirement)
Producer → Redis → Inference Worker → Redis → Persister → InfluxDB → FastAPI → Frontend

### Pipeline implementation notes

**Canonical implementation**: `apps/backend/src/app/pipeline/`
- Producer: `producer.py` (publishes to Redis Pub/Sub)
- Inference: `inference_worker.py` (Redis LIST buffer + LTRIM)
- Persister: `persister.py` (writes to InfluxDB)

**Legacy/experimental**: `apps/backend/src/app/domain/pipeline/redis_inference_worker.py`
- Uses Redis Streams (not Pub/Sub)
- Used by `worker_main.py` on Railway
- Currently non-functional on Railway (no producer)

**Railway deployment**: API-only. Full pipeline requires local Docker Compose or dedicated infrastructure.

---

## Backend API contract (ops + realtime)
- `GET /live` — liveness (fast, no dependency checks)
- `GET /ready` — readiness (bounded dependency checks with timeouts)
- Analytics endpoints must support **1-second resolution** queries for recent data
- If Redis holds raw readings:
  - analytics endpoint(s) should be able to return recent raw readings from Redis (last 60–3600 seconds)

---

## Environment variables (do not leak secrets)

### Frontend (Vite)
- Vite environment variables are injected at **build time** into the client bundle.
- Anything in the client bundle must NOT be secret.
- Client-intended env vars must be prefixed with `VITE_`.

Common:
- `VITE_BACKEND_URL` (must include protocol in production: `https://...`)
- `VITE_SUPABASE_URL`
- `VITE_SUPABASE_ANON_KEY` (anon/public key is OK to ship)

Rules:
- Do not commit `.env.production` with credentials.
- Keep an `.env.example` template; inject actual values via Cloudflare Pages build environment.

### Backend
Common:
- `CORS_ORIGINS` (explicit allowed origins)
- `INFLUX_URL`, `INFLUX_TOKEN`, `INFLUX_ORG`, `INFLUX_BUCKET_*`
- `REDIS_URL`
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY` (secret)
- `PORT` (in Railway or other platforms)

---

## Deployment notes

### Local Docker is canonical for development + correctness
- `compose.yaml` is the default “truth” for running the complete loop locally.
- E2E must validate that the pipeline writes to Influx and the backend can query it deterministically.

### Railway (current state; optional)
- `railway.json` at repo root is written for the **backend API** Dockerfile (`apps/backend/Dockerfile`).
- Do NOT attach this backend Dockerfile to the InfluxDB service (or any other service) on Railway.
- If Railway runs the pipeline, each service must point to the correct image/Dockerfile and must have clear ownership of volumes/artifacts.

---

## CI / Workflows (expected invariants)
- CI must use the Node version from `.nvmrc`.
- Frontend CI must run `npm ci` from repo root (workspaces + root lockfile), then run workspace scripts.
- E2E workflow must:
  - start Docker stack using multi-file merge (`-f compose.yaml -f compose.e2e.yaml`)
  - dump compose logs on failure
  - avoid relying on pre-existing Influx data; prefer tagging writes with an `E2E_RUN_ID`

---

## Model artifacts (common failure mode)
Inference workers may fail to preload models if artifacts aren’t present inside the container (e.g., missing `model.safetensors` at expected paths).
Rules:
- Worker images must either:
  - COPY model artifacts into the image, OR
  - mount them via volumes in compose
- In CI/E2E, missing artifacts should not crash the whole stack unless the test explicitly requires real models.
  - If models are required, CI must ensure they are present.

---

## Troubleshooting (common failures)

### E2E fails with “Failed to load safetensors: No such file or directory”
Likely causes:
- Model artifacts not copied into the worker image
- Wrong container path vs registry path
- Missing volume mount in compose
Fix approach:
- Verify expected on-container model paths
- Ensure Dockerfile COPY and/or compose volume mounts match those paths

### Frontend shows “API unreachable”
Check in order:
1) `VITE_BACKEND_URL` was set at build time
2) backend reachable publicly (if deployed) or reachable in compose network (if local)
3) CORS allows the frontend origin
4) auth requirements match frontend token usage

### “No predictions” in dashboards
Usually:
- persister never wrote to Influx
- wrong bucket/measurement/field schema
- query window too broad/narrow
- services not ready (add readiness waits/healthchecks)

---

## Railway E2E Testing

### Overview
Railway E2E tests validate the deployed pipeline on Railway via secure probe endpoints.
These tests run against the live Railway deployment (not local Docker Compose).

### E2E Probe Endpoints
Protected endpoints for testing (require `E2E_PROBES_ENABLED=true` + `X-E2E-Token` header):
- `POST /e2e/preprocess` — Test preprocessing (returns 7-element feature vector)
- `POST /e2e/inject` — Inject sample into real pipeline via Redis
- `GET /e2e/redis-buffer` — Check Redis buffer status
- `GET /e2e/influx-status?run_id=xxx` — Query InfluxDB for predictions with run_id

### Enabling E2E Probes on Railway
Set these environment variables in Railway dashboard for the backend service:
```
E2E_PROBES_ENABLED=true
E2E_TOKEN=<strong-random-token>
```

Generate token: `openssl rand -hex 32`

### Running Railway E2E Tests Locally
```bash
# Set environment variables
export RAILWAY_BACKEND_URL="https://energy-monitor.up.railway.app"
export E2E_TOKEN="your-e2e-token"
export E2E_RUN_ID="local-$(date +%s)"

# Run tests
cd apps/backend
pytest tests/e2e/test_railway_e2e.py -v
```

### GitHub Actions Workflow
The `railway-e2e.yml` workflow runs automatically:
- On push to `integration` branch (after Railway deploy)
- Daily at 7 AM UTC
- Manual trigger via workflow_dispatch

Required GitHub secrets:
- `RAILWAY_BACKEND_URL`: Railway backend URL
- `E2E_TOKEN`: Same token as Railway env var

### Token Rotation
1. Generate new token: `openssl rand -hex 32`
2. Update Railway environment: `E2E_TOKEN=<new-token>`
3. Update GitHub secret: `E2E_TOKEN`
4. Redeploy Railway backend

### RUN_ID Correlation
- Each test run generates a unique `E2E_RUN_ID`
- The run_id flows through: inject → Redis → inference → persister → InfluxDB
- Tests query InfluxDB filtering by run_id for isolation

---

## Change discipline (what Claude should do)
When making changes:
1) Read relevant files first (don't guess).
2) Change one coherent thing at a time.
3) Run the smallest verification (unit/component → build → e2e).
4) If touching deploy/config/tests, update `docs/` accordingly.
5) Summarize changes file-by-file and include copy/paste verification commands.
