# InfluxDB Setup & Guide

This project uses **InfluxDB 2.x** for storing time-series data (appliance energy predictions and raw sensor readings).

## 1. Quick Start (Local Dev)

The easiest way to run InfluxDB is via Docker Compose.

### Prerequisites
- Docker & Docker Compose
- A Valid `.env` file (copied from `.env.example`) with `INFLUX_TOKEN` set.

### Steps
1.  **Configure Environment**:
    Ensure your `.env` has:
    ```bash
    INFLUX_URL=http://localhost:8086
    INFLUX_TOKEN=my-super-secret-admin-token  # Set this to something secure!
    INFLUX_ORG=energy-monitor
    INFLUX_BUCKET_RAW=raw_sensor_data
    INFLUX_BUCKET_PRED=predictions
    ```

2.  **Start Services**:
    ```bash
    docker compose up -d
    ```
    This starts `influxdb` and a `influxdb-init` helper that specifically ensures both buckets exist.

3.  **Verify Setup**:
    Run the verification script to check connectivity and buckets:
    ```bash
    ./scripts/influx/verify.sh
    ```
    Or check the backend readiness:
    ```bash
    curl http://localhost:8000/ready
    ```
    (Should return `{"status": "ok", ...}`)

4.  **Access UI**:
    - URL: [http://localhost:8086](http://localhost:8086)
    - Username: `admin`
    - Password: `admin12345` (default in compose) or checks `DOCKER_INFLUXDB_INIT_PASSWORD`

## 2. Data Management

### Buckets
- **`raw_sensor_data`**: Stores incoming energy readings from sensors.
- **`predictions`**: Stores ML model inference results.

### Resetting Data
To **completely wipe** all InfluxDB data (buckets, dashboards, users):

> [!WARNING]
> This destroys all local data permanently.

```bash
docker compose down -v
```

To re-initialize:
```bash
docker compose up -d
```

### Seeding Sample Data
To write a few test points to the raw bucket:
```bash
./scripts/influx/seed_raw_sample.sh
```

## 3. Production / Hosted Setup

For production (e.g., Railway, InfluxDB Cloud), this repo does **not** host InfluxDB. You must provide an external instance.

1.  **Create InfluxDB Instance** (Cloud or self-hosted VPS).
2.  **Create Organization** (e.g., `energy-monitor`).
3.  **Create Buckets**: `raw_sensor_data` and `predictions`.
4.  **Create Token**:
    - **Recommended**: Create an "All Access" token for Admin tasks.
    - **Better**: Create a Read/Write token scoped *only* to the two buckets for the Backend service.
5.  **Configure Service Variables**:
    Set these env vars in your deployment (Railway/AWS/etc):
    - `INFLUX_URL`: `https://us-east-1-1.aws.cloud2.influxdata.com` (example)
    - `INFLUX_ORG`: `energy-monitor`
    - `INFLUX_TOKEN`: `your-scoped-token`
    - `INFLUX_BUCKET_RAW`: `raw_sensor_data`
    - `INFLUX_BUCKET_PRED`: `predictions`

## 4. Troubleshooting

**Backend says "Service Unavailable" (503)**:
- Check `docker compose logs backend`.
- Ensure InfluxDB is running (`docker compose ps`).
- Ensure `INFLUX_TOKEN` matches in both `compose.yaml` (or `.env`) for Influx and Backend.

**Buckets missing**:
- Run `docker compose up -d influxdb-init` to re-run the initialization logic.
- Check logs: `docker compose logs influxdb-init`.
