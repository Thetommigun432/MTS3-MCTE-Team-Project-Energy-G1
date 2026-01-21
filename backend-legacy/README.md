# Backend Legacy (DEPRECATED)

> ⚠️ **DEPRECATED**: This folder contains a legacy Flask/Redis backend implementation that has been superseded by the canonical backend at `apps/backend`.

## Status

This folder is preserved for historical reference only. **Do not use or build this code.**

## Canonical Backend

The current production backend is located at:
- **Path**: `apps/backend`
- **Technology**: FastAPI (Python 3.12)
- **Port**: 8000
- **Endpoints**: `/live`, `/ready`, `/infer`, `/models`, `/analytics/*`

## Why Deprecated?

The unified backend at `apps/backend` consolidates:
- This Flask API (`backend/`)
- The PyTorch inference service (`apps/inference-service`)
- The Node.js InfluxDB proxy (`apps/local-server`)

Into a single Python FastAPI monolith.
