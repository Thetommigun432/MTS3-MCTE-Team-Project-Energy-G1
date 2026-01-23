# API Reference

**Base URL**: 
- Local: `http://localhost:8000`
- Production: *(Defined in `VITE_BACKEND_URL`)*

## Authentication

All protected endpoints require a Bearer Token (JWT) from Supabase Auth.

**Header**: `Authorization: Bearer <access_token>`

## Health & Metrics (Public)

| Endpoint | Method | Description | Response |
|----------|--------|-------------|----------|
| `/live` | GET | Liveness probe | `{"status": "ok"}` |
| `/ready` | GET | Readiness probe | `{"status": "ok", "checks": {...}}` |
| `/metrics` | GET | Prometheus metrics | Plain text |

## Analytics Endpoints

### Get Readings
Query High-frequency sensor data or aggregated readings.

`GET /analytics/readings`

**Parameters**:
- `building_id` (string, required): Building UUID
- `start` (string, required): ISO8601 or relative (e.g., `-7d`)
- `end` (string, required): ISO8601 or relative (e.g., `now()`)
- `resolution` (string, optional): `1s` (raw), `1m`, `15m` (downsampled)

### Get Predictions
Query historical inference results.

`GET /analytics/predictions`

**Parameters**: Same as readings.

## Inference Endpoints

### List Models
`GET /models`

Returns available models in the registry.

### Run Inference
`POST /infer`

Run a prediction on a window of power data.

**Request Body**:
```json
{
  "building_id": "string",
  "appliance_id": "string",
  "window": [float, float, ...], // Array of power readings
  "timestamp": "ISO8601 key" // Optional
}
```

**Response**:
```json
{
  "predicted_kw": float,
  "confidence": float,
  "model_version": "string",
  "persisted": boolean
}
```

## Admin Endpoints
Requires `X-Admin-Token` header.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/admin/reload-models` | POST | Hot-reload model registry |
| `/admin/cache/invalidate` | POST | Clear authz cache |
| `/admin/cache/stats` | GET | View cache hit rates |
