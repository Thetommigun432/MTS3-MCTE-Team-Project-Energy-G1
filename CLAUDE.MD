# CLAUDE.MD - NILM Energy Monitor

## Commands
- **Frontend**: `cd apps/web` -> `npm install`, `npm run dev`, `npm run build`, `npm run typecheck`
- **Backend (Local)**: `cd apps/backend` -> `uvicorn app.main:app --reload --port 8000`
- **Backend (Docker)**: `docker compose up -d`
- **Tests**: `npm test` (web), `pytest` (backend)

## Architecture
- **Frontend**: React 19 + Vite. Auth via Supabase. State via Context.
- **Backend**: FastAPI. InfluxDB for time-series. PyTorch for inference.
- **Deploy**: Cloudflare Pages (Web), Railway (Backend).

## Documentation
- See `docs/PROJECT.md` for Architecture & Repo Map.
- See `docs/OPERATIONS.md` for Setup & Deployment.
- See `docs/API.md` for Endpoints.

## Guidelines
- **Branches**: Work on `frontend` or `backend` (synced). `main` is for ML training.
- **Secrets**: Never commit `.env` or secrets. Use `.env.example`.
- **Docs**: Update `docs/*.md` when changing architecture or deployment steps.
