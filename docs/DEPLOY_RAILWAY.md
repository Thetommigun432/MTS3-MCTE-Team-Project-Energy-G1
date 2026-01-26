# Deploying to Railway

The backend is a FastAPI application running in a Docker container.

## Environment Variables

Configure these in the Railway service settings.

### Critical Variables

| Variable | Value (Example) | Description |
|----------|-----------------|-------------|
| `ENV` | `prod` | Enables production validation & security. |
| `PORT` | `8000` | Railway injects this, app listens on it. |
| `CORS_ORIGINS` | `https://mts3-mcte-team-project-energy-g1.pages.dev` | **Comma-separated** list of allowed frontend domains. **No spaces**. |
| `HOST` | `0.0.0.0` | Required for Docker networking. |

### Integrations

| Variable | Description |
|----------|-------------|
| `SUPABASE_URL` | Connects to Supabase (Project Settings > API). |
| `SUPABASE_PUBLISHABLE_KEY` | Public key (Project Settings > API). |
| `INFLUX_URL` | **Private** networking URL (e.g. `http://influxdb:8086`). |
| `INFLUX_TOKEN` | **SECRET**. Admin token or R/W token. |
| `REDIS_URL` | Redis connection string (optional, recommended). |

---

## InfluxDB Configuration

> [!IMPORTANT]
> Properly configuring InfluxDB is critical for the application to function. Follow this matrix exactly.

### Variable Matrix

| Backend Variable | Influx Service Variable | Description |
|------------------|-------------------------|-------------|
| `INFLUX_URL`     | `INFLUXDB_HTTP_PORT` (default 8086) | Use Private Networking URL (e.g. `http://influxdb.railway.internal:8086`) |
| `INFLUX_TOKEN`   | `DOCKER_INFLUXDB_INIT_ADMIN_TOKEN` | **SECRET**. Connection token (admin or read/write) |
| `INFLUX_ORG`     | `DOCKER_INFLUXDB_INIT_ORG` | Organization name (default `energy-monitor`) |
| `INFLUX_BUCKET_RAW` | `DOCKER_INFLUXDB_INIT_BUCKET` | Primary bucket created on init (default `raw_sensor_data`) |
| `INFLUX_BUCKET_PRED` | N/A (Created by Backend) | Secondary bucket for predictions (default `predictions`) |

### Private vs Public Networking

1.  **Backend -> InfluxDB**: MUST use **Private Networking**.
    - Enabled in Railway Service Settings > Networking.
    - URL format: `http://[SERVICE_NAME].railway.internal:[PORT]`
    - Example: `http://influxdb:8086` (if service name is `influxdb`) or custom private domain.
    - **Why?** Latency is lower, traffic is free, and it's secure (not exposed to internet).

2.  **Local Dev -> InfluxDB** (Optional): Use **Public Networking**.
    - Expose a domain in Railway (e.g. `my-influx.up.railway.app`).
    - Use this URL in your local `.env`.
    - **Warning**: Ensure strong authentication is enabled.

### Persistence

- **Volume is Mandatory**: Ensure a volume is attached to `/var/lib/influxdb2`.
- **Initialization**: `DOCKER_INFLUXDB_INIT_*` variables only run on the **first boot** of an empty volume. If you change them later, they will NOT auto-update the database user/org. You must use the Influx CLI or UI to make changes after initialization.

## Health Checks

Railway needs to know if your app is ready.

- **Healthcheck Path**: `/live`
- **Timeout**: `5` seconds
- **Restart Policy**: On Failure

### Verifying Deployment

From your local machine:
```bash
curl https://energy-monitor.up.railway.app/live
# Expected: {"status":"ok","version":"..."}
```

If `/live` returns 200, the backend is up.
If `/ready` returns 200, all dependencies (Influx, Redis) are connected.

## Troubleshooting

- **CORS Errors**: Check `CORS_ORIGINS`. Must match the frontend URL exactly (protocol + domain).
- **AttributeError**: Ensure you are using the latest Docker image (fixes applied in `integration` branch).
