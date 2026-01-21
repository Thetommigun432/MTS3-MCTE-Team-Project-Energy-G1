# Inference Service (DEPRECATED)

> ⚠️ **DEPRECATED**: This service has been superseded by the unified backend at `apps/backend`.

## Status

This folder is preserved for reference during migration. **Do not build or deploy this service.**

## Canonical Backend

The current production inference service is located at:
- **Path**: `apps/backend`
- **Technology**: FastAPI (Python 3.12) with CNNTransformer, safetensors
- **Port**: 8000
- **Inference Endpoint**: `POST /infer`
- **Models Endpoint**: `GET /models`

## Why Deprecated?

The unified backend at `apps/backend` includes:
- All model architectures (CNNTransformer, CNNSeq2Seq, UNet1D)
- Safetensors-only loading (more secure than pickle-based .pth)
- Integrated InfluxDB persistence
- JWT authentication and authorization

See `apps/backend/README.md` for full documentation.
