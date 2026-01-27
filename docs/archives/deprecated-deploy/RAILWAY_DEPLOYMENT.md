# Deprecated: archived deployment doc

> This document is archived and no longer maintained.
> Use [docs/DEPLOYMENT.md](../../DEPLOYMENT.md) instead.

# Railway Deployment Guide

This guide details how to deploy the NILM system (API + Worker) on Railway.

## Architecture

*   **Repository**: Monorepo
*   **Services**:
    1.  **Backend API** (Public): `apps/backend/Dockerfile`
    2.  **Inference Worker** (Private): `apps/backend/Dockerfile.worker`
    3.  **InfluxDB** (External or Railway Service)
    4.  **Redis** (External or Railway Service)

## Deployment Checklist

### Service 1: Backend API (Public)
1.  **Create Service**: "GitHub Repo" -> Select Repo.
2.  **Settings > Service > Config File Path**: Set to `apps/backend/railway.api.toml`.
    *   *Note: This file handles Build (Dockerfile) and Deploy (Healthcheck) settings.*
3.  **Networking**: Enable **Public Networking**.
4.  **Variables**: Configure variables per the [Variable Matrix](#variable-matrix).

### Service 2: Inference Worker (Private)
1.  **Create Service**: "GitHub Repo" -> Select Repo -> "Add Service" (to existing project).
2.  **Settings > Service > Config File Path**: Set to `apps/backend/railway.worker.toml`.
    *   *Note: This file handles Build (Dockerfile.worker) and Start Command settings.*
3.  **Networking**: Disable Public Networking (Service should be private).
4.  **Variables**: Configure variables per the [Variable Matrix](#variable-matrix).

## Variable Matrix

| Variable | Scope | Value / Reference | Notes |
| :--- | :--- | :--- | :--- |
| `ENV` | Shared | `prod` | |
| `INFLUX_URL` | Shared | `${{InfluxDB.INFLUX_URL}}` | Or internal DNS: `http://influxdb.railway.internal:8086` |
| `INFLUX_ORG` | Shared | `${{InfluxDB.INFLUX_ORG}}` | |
| `INFLUX_BUCKET_PRED` | Shared | `predictions` | |
| `INFLUX_TOKEN` | **Sealed** | *(Secret Value)* | Required for both API and Worker |
| `REDIS_URL` | Shared | `${{Redis.REDIS_URL}}` | Or internal DNS: `redis://redis.railway.internal:6379` |
| `SUPABASE_URL` | Shared | `https://<project>.supabase.co` | |
| `SUPABASE_ANON_KEY` | Shared | *(Public Key)* | |
| `SUPABASE_JWT_SECRET` | **Sealed** | *(Secret Value)* | For legacy HS256 auth (optional) |
| `ADMIN_TOKEN` | **Sealed** | *(Secret Value)* | Protects /admin/* endpoints |
| `PIPELINE_ENABLED` | **API** | `false` | API should NOT run the worker loop |
| `PIPELINE_ENABLED` | **Worker**| `true` | Worker MUST run the worker loop |
| `CORS_ORIGINS` | API | `https://<frontend>.pages.dev` | Comma-separated allowed origins |

## Verification

### 1. API Verification
*   **Health Check**: Visit `https://<api-url>.up.railway.app/live`. Should return `200 OK`.
*   **Logs**: Check Deploy Logs for "Application startup complete".

### 2. Worker Verification
*   **Logs**: Check Deploy Logs. Look for `INFO: [Main] Starting NILM Inference Worker...`.
*   **Redis Connection**: Look for `INFO: [Redis] Connected to Redis`.
*   **InfluxDB Connection**: Look for `INFO: [Persister] Connected to InfluxDB`.

### 3. End-to-End Test
1.  Trigger a new reading (via frontend or curl).
2.  Check Worker logs for `Processing reading for house...`.
3.  Check InfluxDB Data Explorer for new point in `predictions` bucket.
