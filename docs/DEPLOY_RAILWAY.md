# Deploying to Railway

The backend is a FastAPI application running in a Docker container.

## Environment Variables

Configure these in the Railway service settings.

### Critical Variables

| Variable | Value (Example) | Description |
|----------|-----------------|-------------|
| `ENV` | `prod` | Enables production validation & security. |
| `PORT` | `8000` | Railway injects this, app listens on it. |
| `CORS_ORIGINS` | `https://myapp.pages.dev,https://energy-monitor.com` | **Comma-separated** list of allowed frontend domains. **No spaces**. |
| `HOST` | `0.0.0.0` | Required for Docker networking. |

### Integrations

| Variable | Description |
|----------|-------------|
| `SUPABASE_URL` | Connects to Supabase. |
| `SUPABASE_PUBLISHABLE_KEY` | Public key. |
| `INFLUX_URL` | Private networking URL (e.g. `http://influxdb:8086`). |
| `INFLUX_TOKEN` | Admin token. |
| `REDIS_URL` | Redis connection string (optional, recommended). |

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
