# Railway Deployment Guide

This guide covers deploying the NILM Energy Monitor to Railway.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Railway Project                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────────┐   │
│  │  API        │   │  Worker     │   │  Simulator          │   │
│  │  (Public)   │   │  (Private)  │   │  (Private)          │   │
│  │             │   │             │   │                     │   │
│  │  FastAPI    │   │  Redis      │   │  Reads from         │   │
│  │  REST API   │   │  Consumer   │   │  InfluxDB raw       │   │
│  │             │   │             │   │  bucket and posts   │   │
│  │  Port 8000  │   │  No port    │   │  to API             │   │
│  └──────┬──────┘   └──────┬──────┘   └──────────┬──────────┘   │
│         │                 │                     │               │
│         └────────┬────────┴─────────────────────┘               │
│                  │                                              │
│         ┌───────┴────────┐                                      │
│         │     Redis      │                                      │
│         │    (Plugin)    │                                      │
│         └───────┬────────┘                                      │
│                 │                                               │
│         ┌───────┴────────┐                                      │
│         │   InfluxDB     │                                      │
│         │   (Service)    │                                      │
│         │                │                                      │
│         │  Buckets:      │                                      │
│         │  - predictions │                                      │
│         │  - raw_readings│                                      │
│         └────────────────┘                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Services

| Service | Config File | Type | Purpose |
|---------|------------|------|---------|
| API | `apps/backend/railway.api.toml` | Public | REST API for frontend |
| Worker | `apps/backend/railway.worker.toml` | Private | Redis stream consumer, runs inference |
| Simulator | `apps/backend/railway.simulator.toml` | Private | Data producer for demos |
| Redis | Railway Plugin | Private | Message queue + buffer |
| InfluxDB | Railway Service | Private | Time series database |

## Prerequisites

1. Railway CLI installed: `npm install -g @railway/cli`
2. Railway account with a project created
3. Local copy of the repository

## Deployment Steps

### 1. Login and Link Project

```bash
# Login to Railway
railway login

# Link to your project
railway link
```

### 2. Add Redis Plugin

```bash
# Add Redis plugin from Railway dashboard or CLI
railway add redis
```

### 3. Deploy InfluxDB Service

Create an InfluxDB service via Railway dashboard:

1. Go to your project in Railway dashboard
2. Click "New Service" > "Docker Image"
3. Use image: `influxdb:2.8`
4. Set environment variables:
   ```
   DOCKER_INFLUXDB_INIT_MODE=setup
   DOCKER_INFLUXDB_INIT_USERNAME=admin
   DOCKER_INFLUXDB_INIT_PASSWORD=<strong-password>
   DOCKER_INFLUXDB_INIT_ORG=energy-monitor
   DOCKER_INFLUXDB_INIT_BUCKET=predictions
   DOCKER_INFLUXDB_INIT_ADMIN_TOKEN=<generated-token>
   ```
5. Add a persistent volume mounted at `/var/lib/influxdb2`

### 4. Create InfluxDB Buckets

After InfluxDB is running, create the raw_readings bucket:

**Option A: Via InfluxDB UI**
1. Access InfluxDB UI at the service's public URL
2. Go to Data > Buckets > Create Bucket
3. Create bucket named `raw_readings` with infinite retention

**Option B: Via InfluxDB CLI (connect to container)**
```bash
# Connect to InfluxDB service
railway run --service influxdb sh

# Create raw_readings bucket
influx bucket create \
  --name raw_readings \
  --org energy-monitor \
  --retention 0
```

### 5. Deploy API Service

```bash
# Create API service
railway service create api

# Set config file path
railway service update api --config-path apps/backend/railway.api.toml

# Set environment variables (Railway dashboard recommended for secrets)
# Or via CLI:
railway variables set \
  ENV=prod \
  INFLUX_URL='${{InfluxDB.RAILWAY_PRIVATE_DOMAIN}}:8086' \
  INFLUX_TOKEN=<your-token> \
  INFLUX_ORG=energy-monitor \
  INFLUX_BUCKET_PRED=predictions \
  INFLUX_BUCKET_RAW=raw_readings \
  REDIS_URL='${{Redis.REDIS_URL}}' \
  CORS_ORIGINS=https://your-frontend.pages.dev \
  SUPABASE_URL=https://your-project.supabase.co \
  SUPABASE_PUBLISHABLE_KEY=<key> \
  PIPELINE_ENQUEUE_ENABLED=true \
  PIPELINE_WORKER_IN_API_ENABLED=false

# Deploy
railway up
```

### 6. Deploy Worker Service

```bash
# Create Worker service
railway service create worker

# Set config file path
railway service update worker --config-path apps/backend/railway.worker.toml

# Set environment variables
railway variables set \
  ENV=prod \
  INFLUX_URL='${{InfluxDB.RAILWAY_PRIVATE_DOMAIN}}:8086' \
  INFLUX_TOKEN=<your-token> \
  INFLUX_ORG=energy-monitor \
  INFLUX_BUCKET_PRED=predictions \
  INFLUX_BUCKET_RAW=raw_readings \
  REDIS_URL='${{Redis.REDIS_URL}}' \
  REDIS_STREAM_KEY=nilm:readings \
  REDIS_CONSUMER_GROUP=nilm-infer

# Deploy
railway up
```

### 7. Seed Raw Data (One-Time)

Before deploying the simulator, you need to populate the raw_readings bucket with your parquet data. Run this from your local machine:

```bash
cd apps/backend

# Install dependencies if needed
pip install -e .

# Set environment variables pointing to Railway InfluxDB
export INFLUX_URL=https://your-influxdb.railway.app
export INFLUX_TOKEN=your-token
export INFLUX_ORG=energy-monitor
export INFLUX_BUCKET_RAW=raw_readings
export PARQUET_PATH=data/simulation_data.parquet
export BUILDING_ID=building-1

# Run ingestion tool
python -m app.tools.ingest_raw_to_influx
```

This will upload the parquet data to InfluxDB. The process takes a few minutes for the default 1-month dataset.

### 8. Deploy Simulator Service

```bash
# Create Simulator service
railway service create simulator

# Set config file path
railway service update simulator --config-path apps/backend/railway.simulator.toml

# Set environment variables
railway variables set \
  BACKEND_URL='http://api.railway.internal:8000' \
  RAW_DATA_SOURCE=influx \
  INFLUX_URL='${{InfluxDB.RAILWAY_PRIVATE_DOMAIN}}:8086' \
  INFLUX_TOKEN=<your-token> \
  INFLUX_ORG=energy-monitor \
  INFLUX_BUCKET_RAW=raw_readings \
  BUILDING_ID=building-1 \
  SIM_SPEEDUP=1 \
  SIM_LOOP=true

# Deploy
railway up
```

## Environment Variable Reference

### Shared Variables (All Services)

| Variable | Required | Description |
|----------|----------|-------------|
| `ENV` | Yes | Environment: `prod` |
| `INFLUX_URL` | Yes | InfluxDB URL (use private domain) |
| `INFLUX_TOKEN` | Yes | InfluxDB admin token |
| `INFLUX_ORG` | Yes | InfluxDB organization |
| `INFLUX_BUCKET_PRED` | Yes | Predictions bucket name |
| `INFLUX_BUCKET_RAW` | Yes | Raw readings bucket name |
| `REDIS_URL` | Yes | Redis connection URL |

### API Service

| Variable | Required | Description |
|----------|----------|-------------|
| `PORT` | No | Injected by Railway |
| `CORS_ORIGINS` | Yes | Comma-separated allowed origins |
| `SUPABASE_URL` | Yes | Supabase project URL |
| `SUPABASE_PUBLISHABLE_KEY` | Yes | Supabase public key |
| `PIPELINE_ENQUEUE_ENABLED` | Yes | Must be `true` |
| `PIPELINE_WORKER_IN_API_ENABLED` | No | Keep `false` (separate worker) |
| `AUTH_VERIFY_AUD` | Yes | Must be `true` in prod |

### Worker Service

| Variable | Required | Description |
|----------|----------|-------------|
| `REDIS_STREAM_KEY` | No | Default: `nilm:readings` |
| `REDIS_CONSUMER_GROUP` | No | Default: `nilm-infer` |
| `MODEL_ARTIFACT_BASE_URL` | Yes | URL for model downloads |

### Simulator Service

| Variable | Required | Description |
|----------|----------|-------------|
| `BACKEND_URL` | Yes | API internal URL |
| `RAW_DATA_SOURCE` | Yes | Must be `influx` for Railway |
| `BUILDING_ID` | No | Default: `building-1` |
| `SIM_SPEEDUP` | No | Playback speed (default: 1) |
| `SIM_LOOP` | No | Loop forever (default: true) |

## InfluxDB Persistence

Railway services have ephemeral storage by default. For InfluxDB, you MUST add a persistent volume:

1. Go to InfluxDB service settings
2. Add Volume
3. Mount path: `/var/lib/influxdb2`
4. Size: Depends on data volume (10GB recommended for demo)

Without this volume, all data will be lost on redeployment.

## Verification

### Check API Health

```bash
curl https://your-api.railway.app/live
curl https://your-api.railway.app/ready
```

### Check Logs

```bash
# API logs
railway logs --service api

# Worker logs
railway logs --service worker

# Simulator logs
railway logs --service simulator
```

### Verify Pipeline Flow

1. Simulator should show "Loading data from InfluxDB" in logs
2. Worker should show "Processing message" entries
3. InfluxDB predictions bucket should have new data

## Troubleshooting

### Simulator: "No raw readings found"

The raw_readings bucket is empty. Run the ingestion tool:
```bash
python -m app.tools.ingest_raw_to_influx
```

### Worker: "Failed to load model"

Ensure `MODEL_ARTIFACT_BASE_URL` is set and models are accessible.

### API: "InfluxDB connection failed"

- Check INFLUX_URL uses the private domain
- Verify INFLUX_TOKEN is correct
- Ensure InfluxDB service is running

### CORS Errors

Add your frontend domain to `CORS_ORIGINS` in the API service.
