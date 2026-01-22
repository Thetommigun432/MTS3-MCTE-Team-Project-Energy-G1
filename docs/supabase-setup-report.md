# Supabase Setup & Inventory Report

## 1. Inventory

### Frontend (`apps/web`)
- **Client Configuration**: `src/integrations/supabase/client.ts` initializes the client.
- **Environment Handling**: `src/lib/env.ts` reads `VITE_SUPABASE_URL` and `VITE_SUPABASE_PUBLISHABLE_KEY`.
- **Auth**: Uses `supabase-js` `auth` module with PKCE flow.
- **Types**: generated types in `src/integrations/supabase/types.ts`.

### Backend (`apps/backend`)
- **Client Configuration**: `app/infra/supabase/client.py` wrapper around `supabase-py`.
- **Services**: `client.py` provides `get_user_buildings`, `get_building_appliances`, `get_user_role`.
- **Auth Verification**: `app/core/security.py` handles JWT verification (HS256 fallback, optional RS256).

### AuthZ Schema Analysis
Based on `apps/web/src/integrations/supabase/types.ts`:
- **Buildings**: `buildings` (id, user_id, ...)
- **Appliances**: `appliances` (id, building_id, user_id, ...)
- **Org Appliances**: `org_appliances` (id, slug, ...)
- **Building Appliances**: `building_appliances` (id, building_id, org_appliance_id, ...)
- **Predictions**: `predictions` (id, building_id, org_appliance_id, ...)

**Potential Mismatch**:
- Backend `get_user_role` queries `profiles.role`, but `types.ts` `profiles` table definition does NOT contain a `role` column. The code notes this discrepancy. We should rely on JWT claims for role or fix the types if the column exists in DB.

## 2. Canonical Environment Variables

We are standardizing on the new Supabase Key Model (Publishable/Secret) with legacy fallback.

### Frontend (Browser)
| Variable | Status | Description |
|----------|--------|-------------|
| `VITE_SUPABASE_URL` | **Required** | Project URL |
| `VITE_SUPABASE_PUBLISHABLE_KEY` | **Preferred** | Public/Anon key. Safe for browser. |
| `VITE_SUPABASE_ANON_KEY` | *Deprecated* | Legacy fallback for `PUBLISHABLE_KEY`. |

### Backend (Server)
| Variable | Status | Description |
|----------|--------|-------------|
| `SUPABASE_URL` | **Required** | Project URL |
| `SUPABASE_PUBLISHABLE_KEY` | **Preferred** | Public/Anon key. Used for metadata queries. |
| `SUPABASE_ANON_KEY` | *Deprecated* | Legacy fallback for `PUBLISHABLE_KEY`. |
| `SUPABASE_JWT_SECRET` | **Required** | Access Token Secret (HS256). |
| `SUPABASE_JWKS_URL` | *Optional* | JWKS URL (RS256). |
| `SUPABASE_SECRET_KEY` | *Restricted* | Service Role Key. Implementation MUST avoid unless fully justified. |

## 3. Hardening Plan

1.  **Standardize Env Vars**: updates `.env.example` files and config code.
2.  **Frontend Polish**:
    - Ensure `client.ts` strictly follows the precedence: Publishable > Anon.
    - Verify `env.ts` correctly exposes these.
3.  **Backend Robustness**:
    - Update `security.py` to be fail-closed (already looks good, just double check tests).
    - Update `config.py` to support `SUPABASE_PUBLISHABLE_KEY`.
    - Fix `get_user_role` to prefer JWT claims or handle missing column gracefully.
4.  **Verification**:
    - Add scripts to verify setup.

## 4. Changes Made (Execution Summary)

### Environment Variables
- Standardized to use `PUBLISHABLE_KEY` (modern) with `ANON_KEY` (legacy) fallback.
- Updated `.env.example` files in root, `apps/web`, and `apps/backend`.

### Frontend (`apps/web`)
- Updated `src/lib/env.ts` to strictly prioritize `VITE_SUPABASE_PUBLISHABLE_KEY`.
- Updated `src/integrations/supabase/client.ts` to use standardized keys.
- Confirmed strict auth verification logic in `security.py`.

### Backend (`apps/backend`)
- Updated `app/core/config.py`:
  - Added `supabase_publishable_key` setting.
  - Added validator to auto-fallback to `supabase_anon_key` if publishable is missing.
- Updated `app/infra/supabase/client.py`:
  - Client initializes with the negotiated key (publishable or anon).
  - `get_user_role` is now fail-safe (returns `None` if column missing in DB, allowing fallback to JWT claim).
- Updated `app/domain/authz/policy.py`:
  - Confirmed it prefers JWT role claim over DB lookup, ensuring 1 fewer DB call + schema safety.

## 5. Verification
See `docs/supabase-verification.md` for manual and automated verification instructions.

A new script `apps/backend/scripts/verify_auth_setup.py` has been added to verify backend configuration:
```bash
python -m apps.backend.scripts.verify_auth_setup
```
