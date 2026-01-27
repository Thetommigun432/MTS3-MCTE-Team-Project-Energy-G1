# Railway Deployment Guide

## Overview

The NILM backend deploys to Railway as **two separate services**:

| Service | Type | Purpose |
|---------|------|---------|
| `backend-api` | PUBLIC | FastAPI REST endpoints |
| `backend-worker` | PRIVATE | Redis inference pipeline |

Plus supporting services:
- **Redis** (Railway plugin)
- **InfluxDB** (external managed, or self-hosted)

---

## Service Configuration

### ⚠️ Critical: Config File Paths

Railway's config file path **does NOT** automatically follow the "Root Directory" setting.

You **must** explicitly set the config file path for each service:

| Service | Config File Path |
|---------|------------------|
| backend-api | `apps/backend/railway.api.toml` |
| backend-worker | `apps/backend/railway.worker.toml` |

Set this in: **Service Settings → Source → Config File Path**

---

## backend-api Service

**Config**: `apps/backend/railway.api.toml`

### Settings
- Root Directory: `/` (or leave default)
- Config File Path: `apps/backend/railway.api.toml`
- Generate Domain: ✅ Yes (public endpoint)

### Environment Variables

```env
# Supabase Auth
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_PUBLISHABLE_KEY=sb_publishable_xxx
SUPABASE_JWT_SECRET=your-jwt-secret

# InfluxDB
INFLUX_URL=https://your-influx-instance.com
INFLUX_TOKEN=your-influx-token
INFLUX_ORG=energy-monitor
INFLUX_BUCKET_RAW=raw_sensor_data
INFLUX_BUCKET_PRED=predictions

# Redis (Railway reference variable)
REDIS_URL=${{Redis.REDIS_URL}}

# CORS (production frontend domain)
CORS_ORIGINS=https://your-frontend.pages.dev

# Pipeline disabled for API service
PIPELINE_ENABLED=false
```

### Healthcheck

Uses `/live` endpoint (not `/ready`) to avoid deploy deadlocks if InfluxDB is slow during initial connection.

---

## backend-worker Service

**Config**: `apps/backend/railway.worker.toml`

### Settings
- Root Directory: `/` (or leave default)
- Config File Path: `apps/backend/railway.worker.toml`
- Generate Domain: ❌ No (private service)

### Environment Variables

```env
# InfluxDB
INFLUX_URL=https://your-influx-instance.com
INFLUX_TOKEN=your-influx-token
INFLUX_ORG=energy-monitor
INFLUX_BUCKET_RAW=raw_sensor_data
INFLUX_BUCKET_PRED=predictions

# Redis (Railway reference variable)
REDIS_URL=${{Redis.REDIS_URL}}
REDIS_STREAM_KEY=readings-stream
REDIS_CONSUMER_GROUP=inference-workers

# Pipeline enabled for worker
PIPELINE_ENABLED=true
PIPELINE_STRIDE=60
PIPELINE_MAX_BUFFER=2000
```

---

## Redis Service

Use Railway's **Redis plugin** (Add → Database → Redis).

Reference in other services using: `${{Redis.REDIS_URL}}`

---

## InfluxDB

**Recommended**: Use a managed InfluxDB Cloud instance.

If self-hosting on Railway:
- Attach a volume to `/var/lib/influxdb2`
- Be aware of limitations: no replicas, redeploy causes brief downtime
- Keep it private (no public domain)

---

## Deployment Steps

1. **Create Railway Project**

2. **Add Redis Plugin**
   - Add → Database → Redis

3. **Create backend-api Service**
   - New Service → GitHub Repo
   - Set Config File Path: `apps/backend/railway.api.toml`
   - Add environment variables (see above)
   - Generate public domain

4. **Create backend-worker Service**
   - New Service → GitHub Repo (same repo)
   - Set Config File Path: `apps/backend/railway.worker.toml`
   - Add environment variables (see above)
   - Do NOT generate domain

5. **Verify Deployment**
   ```bash
   # Check liveness
   curl https://your-api.railway.app/live
   
   # Check readiness
   curl https://your-api.railway.app/ready
   
   # List models
   curl https://your-api.railway.app/models
   ```

---

## Frontend Configuration

Set in Cloudflare Pages (or your frontend host):

```env
VITE_BACKEND_URL=https://your-backend-api.railway.app
```

**Important**: Include the `https://` scheme. Do not use relative URLs.

---

## Troubleshooting

### "Application failed to respond"
- Check that uvicorn binds to `0.0.0.0:$PORT` (not localhost)
- Verify healthcheckPath is `/live` not `/ready`

### Worker not processing messages
- Verify `REDIS_URL` is set correctly
- Check worker logs for connection errors
- Ensure `PIPELINE_ENABLED=true`

### CORS errors in frontend
- Add frontend origin to `CORS_ORIGINS` (no trailing slash)
- Redeploy backend after changing CORS settings
