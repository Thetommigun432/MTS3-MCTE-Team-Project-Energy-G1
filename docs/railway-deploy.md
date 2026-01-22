# Railway Deployment Guide

Deploy the NILM backend service to Railway with Redis caching and InfluxDB persistence.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Railway Project                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────────┐    ┌──────────────────┐          │
│  │   NILM Backend   │───▶│      Redis       │          │
│  │   (FastAPI)      │    │   (Template)     │          │
│  │   PUBLIC         │    │   PRIVATE        │          │
│  └────────┬─────────┘    └──────────────────┘          │
│           │                                             │
│           ▼                                             │
│  ┌──────────────────┐                                  │
│  │    InfluxDB      │                                  │
│  │   (Docker)       │                                  │
│  │   PRIVATE        │                                  │
│  └──────────────────┘                                  │
│                                                          │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
              ┌──────────────────┐
              │     Supabase     │
              │   (External)     │
              └──────────────────┘
```

## Prerequisites

- Railway CLI installed: `npm install -g @railway/cli`
- Railway account with active project
- Supabase project configured

## Service Setup

### 1. Backend Service (Public)

The backend uses the Dockerfile in `apps/backend/`. Railway will auto-detect `railway.json`.

```bash
# Link to Railway project
cd apps/backend
railway link

# Deploy
railway up
```

Set environment variables in Railway dashboard or CLI:

```bash
# Required
railway variables set ENV=prod
railway variables set CORS_ORIGINS=https://your-app.pages.dev
railway variables set SUPABASE_URL=https://your-project.supabase.co
railway variables set SUPABASE_ANON_KEY=eyJ...
railway variables set ADMIN_TOKEN=your-admin-token

# InfluxDB (private network)
railway variables set INFLUX_URL=http://influxdb.railway.internal:8086
railway variables set INFLUX_TOKEN=your-influx-token
railway variables set INFLUX_ORG=energy-monitor
railway variables set INFLUX_BUCKET_RAW=raw_sensor_data
railway variables set INFLUX_BUCKET_PRED=predictions

# Redis (optional, falls back to in-memory)
railway variables set REDIS_URL=redis://default:PASSWORD@redis.railway.internal:6379
```

### 2. Redis Service (Private)

1. Railway Dashboard → **Add Service** → **Redis (Template)**
2. Keep private network only (no public domain)
3. Copy connection URL from service variables

### 3. InfluxDB Service (Private)

1. Railway Dashboard → **Add Service** → **Docker Image**
2. Image: `influxdb:2.7-alpine`
3. Add volume mount: `/var/lib/influxdb2`
4. Set environment variables:

```
DOCKER_INFLUXDB_INIT_MODE=setup
DOCKER_INFLUXDB_INIT_USERNAME=admin
DOCKER_INFLUXDB_INIT_PASSWORD=<secure-password>
DOCKER_INFLUXDB_INIT_ORG=energy-monitor
DOCKER_INFLUXDB_INIT_BUCKET=raw_sensor_data
DOCKER_INFLUXDB_INIT_ADMIN_TOKEN=<your-token>
```

5. Keep private (no public domain)

## Verification

### Health Checks

```bash
# Liveness probe (always 200)
curl https://your-backend.railway.app/live

# Readiness probe (checks InfluxDB + registry)
curl https://your-backend.railway.app/ready
```

Expected `/ready` response:
```json
{
  "status": "ok",
  "checks": {
    "influxdb_connected": true,
    "influx_bucket_raw": true,
    "influx_bucket_pred": true,
    "registry_loaded": true,
    "models_count": 1,
    "redis_available": true
  }
}
```

### Inference Test

```bash
curl -X POST https://your-backend.railway.app/infer \
  -H "Authorization: Bearer $JWT" \
  -H "Content-Type: application/json" \
  -d '{
    "building_id": "test-building",
    "window": [0.1, 0.2, ...],
    "model_id": "my-model"
  }'
```

Expected response (multi-head):
```json
{
  "predicted_kw": {"fridge": 0.05, "oven": 0.0},
  "confidence": {"fridge": 0.85, "oven": 0.85},
  "model_version": "1.0.0",
  "request_id": "abc123",
  "persisted": true
}
```

## Troubleshooting

### InfluxDB Connection Errors

```bash
railway logs -f
```

Check:
- `INFLUX_URL` uses Railway private networking format
- Token has write permissions
- Buckets exist (check InfluxDB logs)

### Redis Fallback

If `/ready` shows `redis_available: false`:
- Check `REDIS_URL` is correct
- Verify Redis service is healthy
- Backend gracefully falls back to in-memory cache

### Model Not Found

- Ensure `models/registry.json` exists
- Verify `.safetensors` files are in `models/`
- Check `is_active: true` for at least one model
