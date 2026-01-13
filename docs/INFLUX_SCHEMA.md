# InfluxDB Schema Documentation

Complete reference for the InfluxDB data model used in the NILM Energy Monitor project.

## Table of Contents

- [Overview](#overview)
- [Database Configuration](#database-configuration)
- [Data Model](#data-model)
- [Query Examples](#query-examples)
- [API Endpoints](#api-endpoints)
- [Data Retention](#data-retention)

---

## Overview

The NILM Energy Monitor uses **InfluxDB 2.7** as a time-series database to store appliance power predictions. This document describes the schema, tags, fields, and query patterns.

### Why InfluxDB?

- **Time-Series Optimized**: Designed for timestamp-indexed data
- **High Write Throughput**: Handles 100k+ points/sec
- **Flux Query Language**: Powerful data transformation and aggregation
- **Retention Policies**: Automatic data expiration
- **Built-in Downsampling**: Reduce storage for old data

---

## Database Configuration

### Connection Details

| Property | Value |
|----------|-------|
| **URL** | `http://localhost:8086` |
| **Organization** | `energy-monitor` |
| **Bucket** | `predictions` |
| **Token** | Set in `.env.local` (see `.env.local.example`) |

### Authentication

All API requests require an authentication token:

```bash
curl -H "Authorization: Token YOUR_INFLUX_TOKEN" \
  http://localhost:8086/api/v2/query
```

The token is stored in `.env.local` and should **never** be committed to version control.

---

## Data Model

### Measurement

**Name:** `appliance_prediction`

This is the primary measurement that stores all NILM prediction data.

### Tags (Indexed)

Tags are indexed and used for filtering queries. They should have low cardinality.

| Tag Name | Type | Description | Example Values |
|----------|------|-------------|----------------|
| `building_id` | string | Unique identifier for the building | `"local"`, `"building-123"` |
| `appliance_name` | string | Type of appliance | `"Dryer"`, `"HeatPump"`, `"Stove"` |

**Tag Cardinality:**
- `building_id`: Low (1-100 buildings)
- `appliance_name`: Low (11 appliances)
- **Total combinations**: ~1,100 series

### Fields (Not Indexed)

Fields store the actual numeric values.

| Field Name | Type | Description | Range | Units |
|------------|------|-------------|-------|-------|
| `predicted_kw` | float | Predicted power consumption | 0.0 - 10.0 | kilowatts (kW) |
| `confidence` | float | Prediction confidence score | 0.5 - 0.95 | ratio (0.0 = 0%, 1.0 = 100%) |

### Timestamp

- **Precision**: Milliseconds
- **Format**: Unix timestamp (milliseconds since epoch)
- **Example**: `1672531200000` = 2023-01-01 00:00:00 UTC

---

## Data Point Structure

### Example Point (JSON)

```json
{
  "_measurement": "appliance_prediction",
  "building_id": "local",
  "appliance_name": "Dryer",
  "predicted_kw": 2.345,
  "confidence": 0.87,
  "_time": "2023-06-15T14:30:00.000Z"
}
```

### Line Protocol Format

InfluxDB uses Line Protocol for writing data:

```
appliance_prediction,building_id=local,appliance_name=Dryer predicted_kw=2.345,confidence=0.87 1686838200000
```

**Format:**
```
<measurement>,<tag_key>=<tag_value>,... <field_key>=<field_value>,... <timestamp>
```

---

## Query Examples

### Basic Queries

#### 1. Get all predictions for a specific appliance

```flux
from(bucket: "predictions")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "appliance_prediction")
  |> filter(fn: (r) => r.appliance_name == "Dryer")
  |> filter(fn: (r) => r._field == "predicted_kw")
```

#### 2. Get predictions for all appliances in a building

```flux
from(bucket: "predictions")
  |> range(start: -7d, stop: now())
  |> filter(fn: (r) => r._measurement == "appliance_prediction")
  |> filter(fn: (r) => r.building_id == "local")
```

#### 3. Get latest prediction for each appliance

```flux
from(bucket: "predictions")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "appliance_prediction")
  |> filter(fn: (r) => r.building_id == "local")
  |> last()
```

### Aggregation Queries

#### 4. Average power consumption per appliance (last 24h)

```flux
from(bucket: "predictions")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "appliance_prediction")
  |> filter(fn: (r) => r._field == "predicted_kw")
  |> group(columns: ["appliance_name"])
  |> mean()
```

#### 5. Total energy consumption per appliance (kWh)

```flux
from(bucket: "predictions")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "appliance_prediction")
  |> filter(fn: (r) => r._field == "predicted_kw")
  |> group(columns: ["appliance_name"])
  |> integral(unit: 1h)  // Convert to kWh
```

#### 6. Downsampled hourly averages

```flux
from(bucket: "predictions")
  |> range(start: -7d)
  |> filter(fn: (r) => r._measurement == "appliance_prediction")
  |> filter(fn: (r) => r._field == "predicted_kw")
  |> aggregateWindow(every: 1h, fn: mean)
```

### Pivot Queries (Wide Format)

#### 7. Pivot fields into columns

```flux
from(bucket: "predictions")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "appliance_prediction")
  |> filter(fn: (r) => r.building_id == "local")
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
```

**Result:**
```
_time                   appliance_name  predicted_kw  confidence
2023-06-15T14:30:00Z    Dryer           2.345         0.87
2023-06-15T14:30:00Z    HeatPump        3.120         0.92
```

### Advanced Queries

#### 8. Find appliances with power > threshold

```flux
from(bucket: "predictions")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "appliance_prediction")
  |> filter(fn: (r) => r._field == "predicted_kw")
  |> filter(fn: (r) => r._value > 1.0)  // More than 1 kW
```

#### 9. Calculate aggregate power (sum of all appliances)

```flux
from(bucket: "predictions")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "appliance_prediction")
  |> filter(fn: (r) => r._field == "predicted_kw")
  |> group(columns: ["_time"])
  |> sum()
```

#### 10. Confidence-weighted average

```flux
import "join"

power = from(bucket: "predictions")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "appliance_prediction")
  |> filter(fn: (r) => r._field == "predicted_kw")

confidence = from(bucket: "predictions")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "appliance_prediction")
  |> filter(fn: (r) => r._field == "confidence")

join.inner(
  left: power,
  right: confidence,
  on: (l, r) => l._time == r._time and l.appliance_name == r.appliance_name,
  as: (l, r) => ({
    _time: l._time,
    appliance_name: l.appliance_name,
    weighted: l._value * r._value
  })
)
```

---

## API Endpoints

### Local API Server

The backend server exposes a simplified API for querying predictions.

#### GET /api/local/predictions

**Query Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `buildingId` | string | No | `"local"` | Building identifier |
| `start` | string | No | `-7d` | Start time (Flux duration or ISO8601) |
| `end` | string | No | `now()` | End time (Flux duration or ISO8601) |

**Example Request:**

```bash
curl "http://localhost:3001/api/local/predictions?buildingId=local&start=-24h&end=now()"
```

**Example Response:**

```json
{
  "success": true,
  "data": [
    {
      "_time": "2023-06-15T14:30:00.000Z",
      "building_id": "local",
      "appliance_name": "Dryer",
      "predicted_kw": 2.345,
      "confidence": 0.87
    },
    {
      "_time": "2023-06-15T14:30:00.000Z",
      "building_id": "local",
      "appliance_name": "HeatPump",
      "predicted_kw": 3.120,
      "confidence": 0.92
    }
  ],
  "count": 2,
  "buildingId": "local",
  "timeRange": {
    "start": "-24h",
    "end": "now()"
  }
}
```

**Error Response:**

```json
{
  "success": false,
  "error": "Query failed: bucket not found"
}
```

---

## Data Retention

### Default Retention Policy

By default, InfluxDB keeps all data indefinitely. For production, configure retention policies to manage storage.

### Create Retention Policy

```flux
// Keep full-resolution data for 30 days
buckets.create(
  name: "predictions_30d",
  orgID: orgID,
  retentionRules: [{
    type: "expire",
    everySeconds: 2592000  // 30 days
  }]
)
```

### Downsampling Task

Downsample old data to reduce storage:

```flux
// Task: Downsample to hourly averages after 7 days
option task = {name: "downsample_predictions", every: 1d}

from(bucket: "predictions")
  |> range(start: -8d, stop: -7d)
  |> filter(fn: (r) => r._measurement == "appliance_prediction")
  |> aggregateWindow(every: 1h, fn: mean)
  |> to(bucket: "predictions_downsampled")
```

---

## Performance Tips

### Write Performance

1. **Batch Writes**: Write multiple points at once (100-5000 points per batch)
2. **Use Line Protocol**: More efficient than JSON
3. **Avoid High Cardinality Tags**: Keep unique tag combinations < 100k

### Query Performance

1. **Use Time Bounds**: Always specify `range(start:, stop:)`
2. **Filter Early**: Apply filters before aggregations
3. **Limit Fields**: Only query fields you need
4. **Use Appropriate Precision**: Milliseconds is sufficient for most use cases

### Example: Optimized Query

```flux
// ✅ GOOD: Early filtering, limited range
from(bucket: "predictions")
  |> range(start: -1h)
  |> filter(fn: (r) => r._measurement == "appliance_prediction")
  |> filter(fn: (r) => r.appliance_name == "Dryer")
  |> filter(fn: (r) => r._field == "predicted_kw")

// ❌ BAD: No time bounds, late filtering
from(bucket: "predictions")
  |> range(start: 0)  // Queries ALL data!
  |> filter(fn: (r) => r._measurement == "appliance_prediction")
  |> mean()
  |> filter(fn: (r) => r.appliance_name == "Dryer")
```

---

## Debugging Queries

### InfluxDB UI Data Explorer

1. Open: http://localhost:8086
2. Navigate to: **Data Explorer**
3. Select bucket: `predictions`
4. Use query builder or write Flux manually
5. Click **Submit** to run

### Check Query Execution Time

```flux
import "profiler"

option profiler.enabledProfilers = ["query", "operator"]

from(bucket: "predictions")
  |> range(start: -24h)
  // ... your query ...
```

### Validate Schema

```flux
// List all measurements
import "influxdata/influxdb/schema"

schema.measurements(bucket: "predictions")

// List all tag keys
schema.tagKeys(bucket: "predictions")

// List all field keys
schema.fieldKeys(bucket: "predictions", measurement: "appliance_prediction")
```

---

## Migration Guide

### Exporting Data

```bash
# Export to CSV
influx query 'from(bucket:"predictions") |> range(start: -30d)' \
  --format csv > export.csv

# Export to JSON
influx query 'from(bucket:"predictions") |> range(start: -30d)' \
  --format json > export.json
```

### Importing Data

```bash
# Import from Line Protocol file
influx write -b predictions -f data.lp

# Import from CSV
influx write -b predictions \
  --format csv \
  --header "#constant measurement,appliance_prediction" \
  --file data.csv
```

---

## Schema Version History

### v1.0 (Current)

- Initial schema with `appliance_prediction` measurement
- Tags: `building_id`, `appliance_name`
- Fields: `predicted_kw`, `confidence`

### Future Enhancements

- Add `model_version` tag for ML model tracking
- Add `actual_kw` field for ground truth comparison
- Add `error_pct` calculated field for accuracy metrics

---

## See Also

- [LOCAL_DEVELOPMENT.md](./LOCAL_DEVELOPMENT.md) - Setup guide
- [InfluxDB Flux Documentation](https://docs.influxdata.com/flux/v0.x/)
- [InfluxDB Line Protocol](https://docs.influxdata.com/influxdb/v2.0/reference/syntax/line-protocol/)
