# Supabase JWT Migration Audit Report

## 1. Inventory

### 1.1 JWT Verification Entrypoint
- **Canonical Entrypoint**: `apps/backend/app/core/security.py` -> `verify_token(token: str)`
- **Dependencies**: `PyJWT`, `httpx` (imported but unused in security.py?), `cryptography` (implied by PyJWT[crypto]).

### 1.2 Environment Variables
| Variable | Current Status | Usage |
|----------|----------------|-------|
| `SUPABASE_URL` | Application Setting | Used to derive issuer and default JWKS URL. |
| `SUPABASE_JWT_SECRET` | Configured (HS256) | Currently used as fallback for HS256 verification. |
| `SUPABASE_JWKS_URL` | Optional Config | Currently used for RS256 if explicitly set. |
| `AUTH_VERIFY_AUD` | Configured | controls audience verification. |

### 1.3 Gaps Identified vs Requirements
1.  **Verification Logic**: Current logic attempts RS256 then falls back to HS256 on error. **Requirement**: Parse header `alg` first ("Auto" mode).
2.  **JWKS URL**: Current logic requires explicit `SUPABASE_JWKS_URL`. **Requirement**: Default to `{SUPABASE_URL}/auth/v1/.well-known/jwks.json`.
3.  **Caching/Refresh**: Current logic relies on simple TTL and `PyJWKClient`. **Requirement**: Explicit "refresh on missing kid" logic with concurrency locking.
4.  **Fail Closed**: Current logic is generally fail-closed, but can be stricter by rejecting unknown algorithms immediately.

## 3. Implementation Status [COMPLETED]

### Phase 1: Configuration Updates
- [x] Update `apps/backend/app/core/config.py`:
    - Add `SUPABASE_PUBLISHABLE_KEY` (preferred).
    - Add `SUPABASE_JWKS_URL` (optional, auto-derived from `SUPABASE_URL`).
    - Mark `SUPABASE_JWT_SECRET` as optional (legacy fallback).
- [x] Update `.env.example` files to reflect new priority.

### Phase 2: Security Module Refactor
- [x] Create `JWKSCache` class in `apps/backend/app/core/security.py`:
    - Use `pyjwt[crypto]` `PyJWKClient`.
    - Implement caching with TTL (default 6h).
    - Implement proper locking (`threading.Lock`) for concurrency safety.
    - Implement "refresh on unknown kid" logic.
- [x] Refactor `verify_token` to "Auto" mode:
    - Parse unverified header for `alg`.
    - If `RS256` -> use `JWKSCache` (Strict JWKS verification).
    - If `HS256` -> check `SUPABASE_JWT_SECRET` (Legacy fallback).
    - If unknown or mismatch -> 401.

### Phase 3: Testing & cleanup
- [x] Add unit tests `apps/backend/tests/test_auth_jwks.py`:
    - Verified RS256 success with real RSA keys.
    - Verified HS256 legacy fallback work.
    - Verified error cases (expired, invalid keys, missing claims).
- [x] Verified local environment integration and Env Var updates.

## 4. Verification Results
- **Unit Tests**: Passed (5/5 tests covering all logic paths).
- **Environment**: Updated `.env` files to include explicit `SUPABASE_JWKS_URL` for clarity and installed `prometheus-client`.
