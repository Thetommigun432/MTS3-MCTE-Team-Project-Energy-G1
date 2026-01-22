# InfluxDB Setup Report

## 1. Inventory

### Files
- **`compose.yaml`**: Orchestrates InfluxDB service and initialization.
    - Service: `influxdb` (image: `influxdb:2.8`)
    - Service: `influxdb-init` (creates `predictions` bucket)
- **`apps/backend/app/infra/influx/client.py`**: Python Async InfluxDB client.
    - Handles connection, querying readings/predictions, and writing predictions.
- **`docs/INFLUX_SCHEMA.md`**: detailed schema documentation.
- **`scripts/verify-influx.ts`**: TypeScript verification script (incomplete coverage).
- **`apps/backend/README.md`**: Backend documentation mentioning Influx.

### Current Environment Variables
| Variable | Default (Code/Compose) | Usage |
| bound | bound | bound |
| `INFLUX_URL` | `http://influxdb:8086` | Backend connection URL |
| `INFLUX_TOKEN` | required | Admin token |
| `INFLUX_ORG` | `energy-monitor` | Organization name |
| `INFLUX_BUCKET_RAW` | `raw_sensor_data` | Raw sensor readings bucket |
| `INFLUX_BUCKET_PRED` | `predictions` | ML predictions bucket |

### Current Buckets
- `raw_sensor_data`: Created by `influxdb` service via `DOCKER_INFLUXDB_INIT_BUCKET`.
- `predictions`: Created by `influxdb-init` service.

### Current Healthchecks
- `influxdb`: `influx ping`
- `backend`: Checks `/live` (which just returns OK), readiness likely deeper.

## 2. Decisions & Standardization

### Env Vars
We will enforce the following canonical set across the repo:
- `INFLUX_URL`
- `INFLUX_TOKEN`
- `INFLUX_ORG`
- `INFLUX_BUCKET_RAW`
- `INFLUX_BUCKET_PRED`

### Docker Compose
- Maintain `influxdb` + `influxdb-init` pattern as it is already somewhat robust.
- Harden `influxdb-init` to be strictly one-shot and fail-closed.
- Ensure `backend` waits for `influxdb-init`.

### Backend Integration
- Backend will use `INFLUX_TOKEN` (admin) for local dev simplicity.
- Docs will mention creating a restricted token for production.
- Improve `/ready` to strictly check Influx connectivity and *both* buckets.

### Scripts
- Create `scripts/influx/verify.sh` for a lightweight, dependency-free verification (curl/cli based).
- Create `scripts/influx/seed_raw_sample.sh` for dev convenience.

## 3. Plan

1.  **Standardize Config**: Update `.env.example` files.
2.  **Harden Docker Compose**: Ensure `influxdb-init` is robust.
3.  **Backend Updates**: Improve `readiness` check.
4.  **Verification Scripts**: Add shell scripts.
5.  **Documentation**: consolidate into `docs/influx.md`.

## 4. Summary of Changes

### Configurations
- **`compose.yaml`**: Hardened `influxdb-init` to fail closed if buckets cannot be verified. Added strict checks for both `raw_sensor_data` and `predictions`.
- **`.env.example`** (Web & Backend): Added standardized `INFLUX_*` variables.

### Backend
- **`client.py`**: Added `verify_setup()` method that performs a deep check (ping + bucket existence).
- **`routers/health.py`**: Updated `/ready` endpoint to return **503 Service Unavailable** if InfluxDB is down or buckets are missing.

### Scripts & Docs
- **`scripts/influx/verify.sh`**: New comprehensive verification script (checks connection, auth, buckets, write/read).
- **`scripts/influx/seed_raw_sample.sh`**: New helper to seed sample data.
- **`docs/influx.md`**: New centralized InfluxDB setup guide.
