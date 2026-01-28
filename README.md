# NILM Energy Monitor

Real-time Non-Intrusive Load Monitoring (NILM) web application with deep learning-based energy disaggregation.

> **Status**: Active Development
> **Canonical Branch**: `backend` (merged with `frontend`)

## 📚 Documentation

The full documentation is located in the `docs/` directory:

- [**Documentation Index**](docs/README.md) - Start here for all guides.
- [**Project Overview**](docs/PROJECT.md) - Goals, architecture, and repo structure.
- [**Local Development**](docs/LOCAL_DEV.md) - Docker Compose + npm workspace setup.
- [**Deployment**](docs/DEPLOYMENT.md) - Railway (API + Worker + Redis + Influx) and Cloudflare Pages.
- [**API Reference**](docs/API.md) - Backend API endpoints and authentication.

## 🚀 Quick Start (Local)

Follow the canonical guide in [docs/LOCAL_DEV.md](docs/LOCAL_DEV.md). It covers:

- Docker Compose setup (`.env.local`)
- Frontend dev server (`npm run dev:web`)
- Local ports and verification steps

### Two Modes Available

| Mode | Command | Data Source |
|------|---------|-------------|
| **Simulator** | `docker compose up -d` | Local simulated data |
| **MQTT Realtime** | `docker compose -f compose.realtime.yaml up -d` | Howest Energy Lab broker |

**Public Dashboard**: http://localhost:8080/live (no login required)

## Tech Stack
- **Frontend**: React 19, Vite 7, TypeScript
- **Backend**: FastAPI, Python 3.12, PyTorch
- **Data**: InfluxDB 2.8, Supabase
