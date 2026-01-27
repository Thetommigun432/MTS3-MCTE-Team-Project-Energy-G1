# Railway Deployment: Inference Worker

This guide explains how to deploy the NILM Inference Worker as a separate service on Railway using the **NILM Stream Pipeline**.

## Architecture

- **API Service**: Handles ingestion (`POST /ingest/readings`) and writes to Redis Streams. Disables internal worker threads.
- **Worker Service**: Consumes streams via `RedisInferenceWorker`, runs inference, and writes to InfluxDB.
- **Redis**: Acts as the message broker (Streams) and state store (Buffers/Semaphores).

## Deployment Steps

### 1. Prerequisites
- A Railway Project with:
  - Redis (Plugin)
  - InfluxDB (Custom Service or Plugin)
  - The `backend` API service deployed.

### 2. Create the Worker Service
1.  **New Service** -> **GitHub Repo** -> Select this repo.
2.  **Settings** -> **Configuration File Path**:
    Enter: `apps/backend/railway.worker.toml`
    *This tells Railway to use the worker configuration (Dockerfile.worker, start command, etc.)*
3.  **Variables** -> Add the following:
    - `ENV`: `prod`
    - `REDIS_URL`: Reference internal Redis (e.g., `${{Redis.REDIS_URL}}`)
    - `INFLUX_URL`: Internal InfluxDB URL (e.g., `http://influxdb:8086`)
    - `INFLUX_TOKEN`: Same as API
    - `INFLUX_ORG`: Same as API
    - `INFLUX_BUCKET_PRED`: Same as API (`predictions`)
    - `PIPELINE_ENABLED`: `true`
    - `MODEL_ARTIFACT_BASE_URL`: (See [RAILWAY_MODELS.md](./RAILWAY_MODELS.md))

4.  **Networking**:
    - Do **NOT** generate a domain. This service is internal only.

### 3. Verify
- Check Deploy Logs: Ensure it says "Starting Redis Inference Worker" and "Subscribed to nilm:readings".
- Check Persistence: Send data to API `/ingest/readings` and check `influxdb` for new predictions.

## Why Separate Services?
- **Scaling**: You can scale the Worker service replicas independently of the API.
- **Reliability**: Worker crashes (OOM, etc.) do not bring down the API.
- **Concurrency**: Redis Streams Consumer Groups allow multiple workers to process shards of buildings in parallel.
