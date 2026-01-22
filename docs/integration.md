# Frontend-Backend Integration Guide

This document describes how to run and deploy the NILM Energy Monitor with proper frontend-backend integration.

## Architecture Overview

```
┌─────────────────┐     HTTPS     ┌──────────────────┐
│ Cloudflare Pages│ ◄────────────►│  Railway Backend │
│    (Frontend)   │   REST API    │    (FastAPI)     │
└─────────────────┘               └──────────────────┘
         │                                 │
         │                                 │
         ▼                                 ▼
┌─────────────────┐               ┌──────────────────┐
│    Supabase     │               │    InfluxDB      │
│  (Auth + DB)    │               │  (Time Series)   │
└─────────────────┘               └──────────────────┘
```

## Local Development

### Prerequisites
- Node.js 20+
- Python 3.11+
- Docker + Docker Compose (for InfluxDB/Redis)

### 1. Start Infrastructure
```bash
# From repo root
docker compose up -d
```

### 2. Start Backend
```bash
cd apps/backend
cp .env.example .env
# Edit .env with your Supabase credentials
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 3. Start Frontend
```bash
cd apps/web
cp .env.example .env
# Edit .env with your Supabase credentials
npm install
npm run dev
```

Frontend runs on `http://localhost:8080` and proxies `/api/*` to backend on port 8000.

### Environment Variables

#### Frontend (.env)
```env
# Required: Supabase Auth
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=eyJ...

# Required for production, optional for local (defaults to localhost:8000)
VITE_BACKEND_URL=

# Optional: Demo mode
VITE_DEMO_MODE=false
```

#### Backend (.env)
```env
# Server
ENV=dev
PORT=8000

# CORS - comma-separated origins
CORS_ORIGINS=http://localhost:5173,http://localhost:8080

# InfluxDB
INFLUX_URL=http://localhost:8086
INFLUX_TOKEN=my-admin-token
INFLUX_ORG=docs
INFLUX_BUCKET_RAW=raw_sensor_data
INFLUX_BUCKET_PRED=predictions

# Supabase (for JWT verification)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=eyJ...
SUPABASE_JWT_SECRET=your-jwt-secret
```

## Production Deployment

### Cloudflare Pages (Frontend)

Set these environment variables in Cloudflare Pages dashboard:

| Variable | Description | Example |
|----------|-------------|---------|
| `VITE_SUPABASE_URL` | Supabase project URL | `https://xxx.supabase.co` |
| `VITE_SUPABASE_ANON_KEY` | Supabase anon key | `eyJ...` |
| `VITE_BACKEND_URL` | Railway backend URL | `https://your-app.railway.app` |

Build settings:
- Build command: `npm run build`
- Build output directory: `apps/web/dist`
- Root directory: `/`

### Railway (Backend)

Set these environment variables in Railway:

| Variable | Description | Example |
|----------|-------------|---------|
| `ENV` | Environment | `prod` |
| `PORT` | Server port | `8000` |
| `CORS_ORIGINS` | Allowed origins | `https://your-app.pages.dev` |
| `INFLUX_URL` | InfluxDB URL | `http://influxdb.railway.internal:8086` |
| `INFLUX_TOKEN` | InfluxDB admin token | `your-token` |
| `INFLUX_ORG` | InfluxDB org | `docs` |
| `INFLUX_BUCKET_RAW` | Raw data bucket | `raw_sensor_data` |
| `INFLUX_BUCKET_PRED` | Predictions bucket | `predictions` |
| `SUPABASE_URL` | Supabase project URL | `https://xxx.supabase.co` |
| `SUPABASE_ANON_KEY` | Supabase anon key | `eyJ...` |
| `SUPABASE_JWT_SECRET` | JWT signing secret | `your-secret` |
| `REDIS_URL` | Redis connection | `redis://...railway.internal:6379` |
| `ADMIN_TOKEN` | Admin API token | `secure-random-token` |

## API Integration Details

### Authentication Flow
1. User logs in via Supabase Auth on frontend
2. Frontend stores JWT access token in localStorage
3. API client (`apps/web/src/services/api.ts`) automatically attaches `Authorization: Bearer <token>` header
4. Backend validates JWT via Supabase JWKS endpoint

### Config Module
The canonical config is in `apps/web/src/lib/env.ts`:
- `getEnv()` - Returns all environment config
- `hasBackendUrl()` - Check if backend URL is configured
- `isSupabaseEnabled()` - Check if Supabase is configured

Re-exported from `apps/web/src/config.ts` for convenience.

## Troubleshooting

### CORS Errors
- Ensure `CORS_ORIGINS` on backend includes your frontend domain
- Check for trailing slashes (don't include them)
- Verify preflight OPTIONS requests work

### 401 Unauthorized
- Check Supabase JWT is valid and not expired
- Verify `SUPABASE_JWT_SECRET` matches on backend
- Try refreshing the auth session

### Network Errors
- Confirm backend is running and accessible
- Check `VITE_BACKEND_URL` is correct
- Verify no firewall/proxy issues

### Backend Not Starting
- Run `python -c "from app.main import app"` to check imports
- Verify all required env vars are set
- Check InfluxDB connection
