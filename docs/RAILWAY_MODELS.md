# Production Model Artifacts

The system uses `.safetensors` model weights defined in `models/registry.json`. Since these large files are not stored in Git, they must be fetched at runtime in production.

## Registry & SHA Verification
The `apps/backend/models/registry.json` file is the source of truth. It contains:
- `artifact_path`: Relative path (e.g., `tcn_sa/evcharger/model.safetensors`)
- `artifact_sha256`: **Strictly enforced** checksum.

If a downloaded file does not match the SHA256, it is deleted and the worker refuses to load that model.

## Hosting Models
You must host the model files in a publicly accessible or reachable location (S3, R2, Supabase Storage, GitHub Releases).

### Directory Structure
Your host must mirror the structure expected by `artifact_path`.
Example Base URL: `https://my-bucket.r2.cloudflarestorage.com/models`

Expected URLs:
- `https://.../models/tcn_sa/evcharger/v1-sota/model.safetensors`
- `https://.../models/tcn_sa/heatpump/v1-sota/model.safetensors`

## Configuration
Set the following environment variable on both **API** and **Worker** services in Railway:

```bash
MODEL_ARTIFACT_BASE_URL=https://your-storage-url.com/models
```

### Optional Settings
- `MODEL_ARTIFACT_TIMEOUT_SEC`: Download timeout (default 60s)
- `MODEL_ARTIFACT_MAX_MB`: Max size in MB (default 500)

## Automatic Fetching
When the application starts (or when a model is requested), the system:
1. Checks `/app/models/{artifact_path}`.
2. If missing, downloads from `{MODEL_ARTIFACT_BASE_URL}/{artifact_path}`.
3. Verifies SHA256.
4. Loads the model.

If the file exists but has the wrong SHA (e.g., corrupted or outdated), it is deleted and re-downloaded.
