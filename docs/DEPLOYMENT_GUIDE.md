# Deployment Guide

This guide details how to deploy the NILM Energy Monitor application. The architecture consists of:
1.  **Backend API**: FastAPI service hosted on **Railway**.
2.  **Worker**: Background Python worker hosted on **Railway**.
3.  **Frontend**: React (Vite) application hosted on **Cloudflare Pages**.
4.  **Databases**: InfluxDB and Redis hosted on **Railway**.

---

## 1. Railway Deployment (Backend)

We use Railway's Config as Code (toml) to define our services.

### Prerequisites
*   Railway CLI installed (`npm i -g @railway/cli`)
*   Logged in (`railway login`)
*   A Railway Project created.

### Services Configuration

#### A. Databases
Ensure your Railway project has the following databases mapped or provisioned:
*   **Redis**: Standard provision.
*   **InfluxDB**: Custom Image (`influxdb:2.7`) or persistent service. Ensure volumes are attached.

#### B. Backend API Service
*   **Source**: Monorepo root.
*   **Config File**: `apps/backend/railway.api.toml`
*   **Networking**: Public Networking **Enabled**.
*   **Healthcheck**: `/live` (Timeout: 300s).

#### C. Worker Service
*   **Source**: Monorepo root.
*   **Config File**: `apps/backend/railway.worker.toml`
*   **Networking**: Public Networking **Disabled** (Internal only).
*   **Restart Policy**: Always.

### Environment Variable Matrix (Backend)

Apply these variables to the **Shared Environment** in Railway so both API and Worker can access them.

| Variable | Required | Description | Example Value |
| :--- | :--- | :--- | :--- |
| `PORT` | Yes (API) | Provided by Railway. | `8000` |
| `ENV` | Yes | Environment mode. | `production` |
| `LOG_LEVEL` | No | Logging verbosity. | `INFO` |
| `INFLUX_URL` | Yes | Internal InfluxDB URL. | `http://influxdb:8086` |
| `INFLUX_TOKEN` | Yes | Admin token. | `secure-production-token` |
| `INFLUX_ORG` | Yes | Organization name. | `mcte-energy` |
| `INFLUX_BUCKET_RAW` | Yes | Bucket for raw readings. | `raw_sensor_data` |
| `INFLUX_BUCKET_PRED`| Yes | Bucket for predictions. | `predictions` |
| `REDIS_URL` | Yes | Internal Redis URL. | `redis://default:pass@redis:6379` |
| `SUPABASE_URL` | Yes | Supabase Project URL. | `https://xyz.supabase.co` |
| `SUPABASE_SERVICE_ROLE_KEY` | Yes | **Secret** Service Role Key. | `ey...` |
| `SUPABASE_JWKS_URL` | Yes | URL for JWT verification. | `https://xyz.supabase.co/auth/v1/.well-known/jwks.json` |
| `CORS_ORIGINS` | Yes | Allowed Frontend Origins. | `https://your-frontend.pages.dev` |
| `PIPELINE_ENABLED` | Yes | Enable worker pipeline logic. | `true` (Worker), `false` (API) |

> **Note**: Set `PIPELINE_ENABLED=true` specifically for the **Worker** service variable override, and `false` for the API if possible, or handle via shared env logic (Defaults to false in code if missing, but Worker needs it true).

---

## 2. Cloudflare Pages Deployment (Frontend)

Run the frontend as a static site.

### Build Configuration
*   **Framework Preset**: Vite
*   **Build Command**: `npm run -w apps/web build`
*   **Root Directory**: `/` (Repository Root)
*   **Build Output Directory**: `apps/web/dist`

### Environment Variable Matrix (Frontend)

Set these in **Cloudflare Pages > Settings > Environment Variables** (Production).

| Variable | Required | Description | Example Value |
| :--- | :--- | :--- | :--- |
| `VITE_BACKEND_URL` | Yes | Public URL of Railway API. | `https://backend-api-production.up.railway.app` |
| `VITE_SUPABASE_URL` | Yes | Supabase Project URL. | `https://xyz.supabase.co` |
| `VITE_SUPABASE_PUBLISHABLE_KEY`| Yes | Public Anon Key. | `ey...` |

> **Important**: `VITE_SUPABASE_ANON_KEY` is deprecated but supported as fallback. Use `VITE_SUPABASE_PUBLISHABLE_KEY`.

### Verification
1.  **Live Check**: Visit `https://your-frontend.pages.dev/api/live`. It should proxy or directly call backend (if configured). Actually, frontend calls backend via CORS.
2.  **Network Tab**: Verify XHR requests go to `VITE_BACKEND_URL/api/...`.

---

## 3. Local Development (Docker Compose)
To run the full stack locally:

```bash
# Start all services (Backend, Worker, Influx, Redis)
docker compose up -d --build

# Run frontend dev server
npm run -w apps/web dev
```
