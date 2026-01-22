# Railway Deployment Report

## 1. Current Status Analysis
- **Repo Structure**: Monorepo with backend in `apps/backend`.
- **Dockerfile**: Located at `apps/backend/Dockerfile`.
    - ✅ Base: `python:3.12-slim`
    - ✅ User: Non-root `appuser`
    - ✅ Port: Binds to `0.0.0.0:$PORT`
    - ✅ Healthcheck: Built-in `curl` to `/live`
    - ✅ Dependencies: Pinned
- **Missing**:
    - No `railway.json` or `railway.toml` (Config as Code).
    - Railway UI settings likely rely on defaults which fail for monorepos (watching root `/` instead of `apps/backend`).

## 2. Deployment Strategy
**Selected Strategy: A (Root Directory = /apps/backend)**

- **Railway Config**: `railway.json` at repo root.
- **Service Settings**:
    - **Root Directory**: Set to `apps/backend`.
    - **Builder**: Dockerfile (Auto-detected in `apps/backend`).
- **Why**: This properly scopes the build context to the backend application, preventing unintentional rebuilds from frontend changes and keeping the Docker build consistent with local `docker build` commands run from `apps/backend`.

## 3. Implementation Plan
1.  **Create `railway.json`**: Define watch patterns and health checks.
2.  **Update Documentation**: Explicitly instruct user to set "Root Directory" in Railway.
3.  **Verify**: Ensure Dockerfile builds locally with the same context.

## 4. Risks & Mitigations
- **Models**: The Dockerfile tries to copy `models/`. usage of `2>/dev/null || true` prevents build failure if missing, but runtime inference will fail if models aren't present.
    - *Mitigation*: User must ensure models are available (either committed, or downloaded). For now, we assume models are handled or the app starts without them (lazy load).
