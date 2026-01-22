# NILM Backend

Production-grade FastAPI backend for the Energy Monitor.

**Full Documentation**:
- [Backend Documentation](../../docs/backend.md)
- [Railway Deployment Guide](../../docs/deployment/railway.md)

## Railway Deployment

This service is deployed on Railway with Redis and InfluxDB.

### Quick Start (CLI)

```bash
# 1. Link to project
railway link

# 2. Deploy Backend
# (Run from repo root to capture build context)
railway up --service backend
```

### Configuration

- **Config as Code**: `railway.json` (at repo root) defines build context and Dockerfile.
- **Port**: Auto-detected (8000).
- **Health Checks**:
  - `/live` (Liveness)
  - `/ready` (Readiness - checks InfluxDB/Redis)

### Services Setup

1. **Backend** (Public): `MTS3-MCTE-Team-Project-Energy-G1`
   - Env Vars: `ENV, INFLUX_*, REDIS_URL, SUPABASE_*, ADMIN_TOKEN`
2. **Redis** (Private): `Redis`
   - Used for headers caching & idempotency.
3. **InfluxDB** (Private): `influxdb`
   - **Important**: Must attach volume to `/var/lib/influxdb2` manually in UI if using Docker image.
   - Init vars: `DOCKER_INFLUXDB_INIT_*`

### Troubleshooting

- **InfluxDB Connection**: Ensure `INFLUX_URL` uses internal DNS (`http://influxdb.railway.internal:8086`).
- **Volume Permissions**: If InfluxDB fails to write, set `RAILWAY_RUN_UID=0` in InfluxDB variables.
- **Predictions Bucket**: Backend attempts to create it on startup if missing (`ensure_predictions_bucket`).
