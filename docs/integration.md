# Integration Guide

## Overview

This document describes the data flow and contracts between the frontend, backend, and external services (InfluxDB, Redis, Supabase).

---

## Building Identifiers

### Problem: UUID vs Stream Key

| Component | Identifier Type | Example |
|-----------|-----------------|---------|
| Supabase `buildings` table | UUID | `550e8400-e29b-41d4-a716-446655440000` |
| Redis streams | Stream Key | `building_1` |
| InfluxDB tags | Either | Depends on persister configuration |

### Solution: `stream_key` Column

Buildings table now has an optional `stream_key` column:

```sql
ALTER TABLE buildings ADD COLUMN stream_key text UNIQUE;
```

**Usage:**
- If `stream_key` is set, it maps legacy stream identifiers to the Supabase UUID
- Backend resolves building queries using this mapping
- New buildings use UUID directly (no stream_key needed)

---

## InfluxDB Schema

### Canonical Schema (Wide Format)

```
Bucket: predictions
Measurement: prediction
Tags:
  - building_id (UUID or stream_key)
  - model_version
  - stream_key (optional, for mapping)
Fields:
  - predicted_kw_{appliance} (float)
  - confidence_{appliance} (float)
  - is_on_{appliance} (bool, optional)
  - latency_ms
  - request_id
  - user_id
```

**Example Point:**
```
prediction,building_id=550e8400...,model_version=1.0.0 
  predicted_kw_HeatPump=2.5,confidence_HeatPump=0.92,
  predicted_kw_Dishwasher=0.0,confidence_Dishwasher=0.65 1706200800000000000
```

### Legacy Schema (Narrow Format)

```
Bucket: predictions
Measurement: nilm_predictions
Tags:
  - building_id
  - appliance
  - model_version
Fields:
  - power_watts (float)
  - confidence (float)
  - probability (float)
  - is_on (bool)
```

**Note:** Backend supports both formats with fallback queries.

---

## Redis Channels

### Prediction Ingestion

Channel: `nilm:{stream_key}:predictions`

Message format:
```json
{
  "timestamp": 1706200800.123,
  "predictions": [
    {
      "appliance": "HeatPump",
      "power_watts": 2500.0,
      "confidence": 0.92,
      "probability": 0.92,
      "is_on": true,
      "model_version": "1.0.0"
    }
  ]
}
```

---

## API Endpoints

### GET /analytics/appliances

Returns unique appliances for a building.

**How it works:**
1. Queries `schema.measurementFieldKeys()` to list all fields in `prediction` measurement
2. Parses field keys matching `predicted_kw_*` pattern
3. Falls back to `appliance_id` tag query for legacy narrow format

### GET /models

Returns available models with heads and metrics.

**Response includes:**
- `heads[]`: Array of `{appliance_id, field_key}` for multi-head models
- `metrics`: Optional performance metrics (MAE, RMSE, F1)

---

## Frontend Mode

The frontend operates in two modes:

| Mode | Description | Data Source |
|------|-------------|-------------|
| `demo` | Simulated data | Mock generators |
| `api` | Real backend | Backend API calls |

**API mode requirements:**
- `VITE_BACKEND_URL` must be set at build time
- User must be authenticated
- Backend must be reachable
