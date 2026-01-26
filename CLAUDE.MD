# CLAUDE.md — NILM Energy Monitor (Monorepo)

This file gives Claude persistent, repo-specific context: where things live, how to run them, how to test, and what constraints to follow.

## Hard rules
- Work **only** on branch: `integration` (no new branches).
- **Never commit secrets**. No real tokens/keys in code, docs, workflows, or logs.
- Prefer **small, verifiable** changes. After each change: run the narrowest relevant checks.
- Don’t invent new services. Fix and harden what already exists.

---

## Repo map (high level)
- `apps/web/` — Frontend (React 19 + Vite)
- `apps/backend/` — Backend (FastAPI)
- `compose.yaml` — Local stack (services)
- `compose.e2e.yaml` — E2E overrides (should be used via multi-file merge)
- `railway.json` — Railway config-as-code (build/deploy settings + healthcheck path)
- `docs/` — Architecture, operations, deployment notes

---

## Quick commands

### Frontend (Vite/React)
From repo root:
- Install: `npm ci`
- Dev: `npm --workspace apps/web run dev`
- Build: `npm run build:web`
- Lint: `npm run lint:web`
- Typecheck: `npm run typecheck:web`
- Tests: `npm run test:web`

### Backend (FastAPI)
From `apps/backend`:
- Install: `python -m pip install -r requirements.txt`
- Run (dev): `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`
- Tests: `pytest -q`
  - If markers exist:  
    - fast: `pytest -m "not e2e" -q`  
    - e2e: `pytest -m e2e -v`

### Docker / Compose (local)
- Start stack: `docker compose -f compose.yaml up -d --build`
- Stop stack: `docker compose -f compose.yaml down -v`

### Docker / Compose (E2E)
**Do not use `include:` to override services.** Use multi-file compose merging:
- Start E2E stack:  
  `docker compose -f compose.yaml -f compose.e2e.yaml up -d --build --remove-orphans`
- Run E2E tests (backend container or local):  
  `pytest -m e2e -v`
- Teardown:  
  `docker compose -f compose.yaml -f compose.e2e.yaml down -v --remove-orphans`

---

## Architecture (contract-level)

### Frontend
- React 19 + Vite.
- Auth via Supabase.
- Two modes:
  - **Demo mode**: uses local/demo data.
  - **API mode**: must reach backend and display:
    - Buildings (user-created)
    - Appliances (catalog from models; user assigns to buildings)
    - Models page (models + basic metrics)

### Backend
- FastAPI serves:
  - `GET /live` — liveness (fast, no dependency checks)
  - `GET /ready` — readiness (bounded checks for external deps)
  - Model registry endpoints (models + heads + metrics)
  - Analytics endpoints (reads predictions from Influx)
- Time-series: InfluxDB
- Streaming/cache: Redis
- Inference: current model approach (transformer-based model is deprecated)

### Data flow (expected)
preprocess → Redis ingest → inference per appliance/model → aggregate → write predictions to Influx → backend serves → frontend visualizes.

---

## Environment variables (do not leak secrets)

### Frontend (Vite)
- Vite replaces `import.meta.env.*` at **build time**.
- **Anything exposed to the client bundle must NOT be secret.**
- Variables intended for client use must be prefixed with `VITE_`.

Common:
- `VITE_BACKEND_URL` (API base URL)
- `VITE_SUPABASE_URL`
- `VITE_SUPABASE_ANON_KEY` (public anon key is OK to ship)

### Backend
Common:
- `PORT` (Railway injects this; do not hardcode 8000 in production)
- `CORS_ORIGINS` (explicit list of allowed origins; must include deployed frontend domain)
- `INFLUX_URL`, `INFLUX_TOKEN`, `INFLUX_ORG`, `INFLUX_BUCKET_*`
- `REDIS_URL`
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY` (backend-only, secret)
- `SUPABASE_JWKS_URL` (if used)

---

## Deployment

### Railway (backend)
- Railway uses a provided `PORT`. Backend must bind to `0.0.0.0:$PORT`.
- `railway.json` is config-as-code and overrides dashboard settings.
- Healthcheck path in `railway.json` must match an actual route (prefer `/live`).
- Expect `/ready` to check dependencies with timeouts (no hanging).

### Cloudflare Pages (frontend)
- Ensure build-time env vars are set for the production build (not runtime).
- API mode must show actionable errors:
  - missing VITE_BACKEND_URL vs network/CORS vs 401/403 vs 5xx.

---

## Testing strategy (must stay thorough)

### Layers
- **Unit**: pure logic, no network/docker.
- **Component/API**: FastAPI TestClient + mocked deps.
- **E2E**: docker stack; validates pipeline writes to Influx and backend can query it.
- **Railway smoke**: checks deployed endpoints (`/live`, `/ready`) from GitHub Actions.

### E2E determinism rule
- E2E must not rely on “random existing data”.
- Prefer tagging Influx writes with an `E2E_RUN_ID` so queries can filter and be deterministic.

---

## CI / Workflows (expected)
- `ci.yml`: runs on PR + push; fast checks:
  - backend unit/component tests (not e2e)
  - frontend lint/typecheck/build/tests
- `e2e.yml`: runs on push to `integration` + scheduled:
  - boots docker stack (multi-file compose)
  - runs pytest e2e tests
  - dumps compose logs on failure
- `railway-smoke.yml`: runs on push to `integration` + scheduled:
  - curls deployed `/live` and `/ready`
  - uses `RAILWAY_BACKEND_URL` secret
  - must NOT run with secrets on fork PRs

---

## Troubleshooting (common failures)

### Frontend shows “API unreachable”
Check in order:
1) `VITE_BACKEND_URL` was set at build time (Cloudflare).
2) Backend is reachable publicly (Railway URL).
3) CORS allows the frontend origin.
4) Backend auth requirements match frontend token usage.

### E2E fails “no predictions”
Usually:
- pipeline never wrote to Influx
- wrong measurement/bucket
- query too broad (no run_id filter)
- services not ready (add healthchecks + waits)

### Compose errors about “conflicts with imported resource”
Stop using `include:` overrides. Use:
`docker compose -f compose.yaml -f compose.e2e.yaml ...`

---

## Change discipline (what Claude should do)
When making changes:
1) Read relevant files first (don’t guess).
2) Change one coherent thing at a time.
3) Run the smallest verification (unit/component → build → e2e).
4) If touching deploy/config/tests, update `docs/` accordingly.
5) Always summarize changes file-by-file and include copy/paste verification commands.
