# InfluxDB Configuration Audit

## Discovery Findings

### 1. Docker Compose Configuration
- **Local (`compose.yaml`)**:
  - Uses `influxdb:2.8` with `setup` mode.
  - Automatically initializes `raw_sensor_data` bucket.
  - Has `influxdb-init` helper service to create the second `predictions` bucket.
  - Healthcheck uses `influx ping`.
  - Uses safe defaults for local development (e.g., `influx-admin-token-2026-secure`).

- **E2E (`compose.e2e.yaml`)**:
  - Overrides `nilm-persister`, `e2e-tests` to inject test-specific configuration.
  - **Issue**: Uses `${INFLUX_TOKEN}` without defaults in several places. This causes `docker compose down` to fail locally if the variable isn't set in the shell, as variable substitution happens before the command runs.

### 2. Backend Configuration
- Uses `pydantic-settings` in `src/app/core/config.py`.
- Defaults `influx_url` to `http://localhost:8086`.
- correctly validates `INFLUX_TOKEN` presence in `prod` environment.

### 3. Backend Client Implementation
- Uses `influxdb-client` (v2) async client.
- Has `ensure_buckets()` logic that duplicates `influxdb-init` but adds robustness.
- **Issue**: `connect()` logs a warning but proceeds if `INFLUX_TOKEN` is missing, potentially leading to runtime errors later.

### 4. CI/CD Workflows
- `e2e.yml` correctly generates a `.env.e2e` file with all required variables `INFLUX_TOKEN`, `INFLUX_ORG`, etc.
- Passes `--env-file .env.e2e` to both `up` and `down` commands.
- Captures logs on failure.

---

## Remediation Plan

### 1. Fix Compose Files
- Update `compose.e2e.yaml` to use the same safe defaults as `compose.yaml` (e.g., `${INFLUX_TOKEN:-influx-admin-token-2026-secure}`). This ensures local `up`/`down` commands are stable without requiring explicit env vars.

### 2. Harden Backend Client
- Update `InfluxClient.connect()` to raise `InfluxError` immediately if `influx_token` is missing, ensuring fail-fast behavior.

### 3. Verify Data Model
- Confirm `persister` writes include all required tags (`building_id`, `appliance_id`, `model_version`).
- Confirm E2E isolation uses `E2E_RUN_ID`.

## Final Configuration Summary

| Environment | Influx URL | Token Source | Bucket Init |
|-------------|------------|--------------|-------------|
| **Local** | `http://influxdb:8086` | `.env` or Default | `DOCKER_INFLUXDB_INIT_*` + `influxdb-init` |
| **E2E (CI)** | `http://influxdb:8086` | Generated in Workflow | `DOCKER_INFLUXDB_INIT_*` + `influxdb-init` |
| **Prod (Railway)** | Set via Env Var | Railway Config | `ensure_buckets()` (Backend) |
