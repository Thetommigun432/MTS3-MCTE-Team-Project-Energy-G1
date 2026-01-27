# Operations & Troubleshooting

This document is a runbook for operating and troubleshooting the system. It does not replace setup or deployment guides.

**Primary guides:**
- Local dev: [LOCAL_DEV.md](./LOCAL_DEV.md)
- Deployment: [DEPLOYMENT.md](./DEPLOYMENT.md)
- Integration dataflow: [integration.md](./integration.md)

## Health Checks

### API
- Liveness: `GET /live`
- Readiness: `GET /ready`

### Expected responses
- `/live` returns 200 when the API process is running.
- `/ready` returns 200 when dependencies (InfluxDB/Redis) are reachable.

## Common Incidents

### 1) Frontend shows "API unreachable"
Checklist:
- Verify the backend URL used at build time (`VITE_BACKEND_URL`).
- Check that the backend responds at `/live`.
- Ensure `CORS_ORIGINS` includes your frontend domain.

### 2) No predictions visible
Checklist:
- Verify the worker is consuming Redis streams.
- Confirm the predictions bucket exists (`INFLUX_BUCKET_PRED`).
- Check InfluxDB for recent points in the `predictions` bucket.

### 3) Redis stream empty
Checklist:
- Ensure `PIPELINE_ENQUEUE_ENABLED=true` on the API service.
- Confirm ingestion endpoint is receiving data (`POST /ingest/readings`).

## Logs

### Docker Compose (local)
```bash
docker compose logs -f backend
docker compose logs -f worker
docker compose logs -f simulator
```

### Railway (production)
Use Railway service logs for:
- API service: request handling, CORS issues, readiness
- Worker service: stream consumption, inference errors

## Dataflow Verification (quick)

1) **API health**
```bash
curl https://<api-domain>/live
```

2) **Redis stream length** (local)
```bash
docker exec nilm-redis redis-cli XLEN nilm:readings
```

3) **Influx predictions** (local)
```bash
docker exec -it $(docker compose ps -q influxdb) influx query \
  'from(bucket: "predictions") |> range(start: -5m) |> limit(n: 5)' \
  --org energy-monitor --token $INFLUX_TOKEN
```
