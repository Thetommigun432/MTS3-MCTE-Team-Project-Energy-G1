# Supabase Setup Guide for NILM Energy Monitor

This guide covers the complete Supabase setup for the NILM Energy Monitor project.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Frontend Hosting Setup](#frontend-hosting-setup)
3. [Supabase Dashboard Configuration](#supabase-dashboard-configuration)
4. [Database Migrations](#database-migrations)
5. [Storage Bucket Setup](#storage-bucket-setup)
6. [Edge Functions Deployment](#edge-functions-deployment)
7. [GitHub Secrets Setup](#github-secrets-setup)
8. [Verification Checklist](#verification-checklist)

---

## Prerequisites

### Required Tools
- [Supabase CLI](https://supabase.com/docs/guides/cli)
- [Deno](https://deno.land/) (for Edge Functions)
- Node.js 18+ with npm/bun

### Environment Variables

```bash
# Production Site URL (Azure Storage Static Website)
PROD_SITE_URL="https://energymonitorstorage.z1.web.core.windows.net/"

# Local Development URL
LOCAL_SITE_URL="http://localhost:5173"

# Supabase Project (from Dashboard > Settings > API)
SUPABASE_PROJECT_URL="https://bhdcbvruzvhmcogxfkil.supabase.co"
SUPABASE_ANON_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJoZGNidnJ1enZobWNvZ3hma2lsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Njc5NTc4NDgsImV4cCI6MjA4MzUzMzg0OH0.zFAx3Wwz6Gyy9dfCc0xxe9QomaembUI8b4bmM23IrRI"

# ⚠️ NEVER expose service_role key in frontend code or commit to git!
# SUPABASE_SERVICE_ROLE_KEY is only used in Edge Functions (server-side)
```

---

## Frontend Hosting Setup

### Current: Cloudflare Pages

The frontend is hosted on **Cloudflare Pages** with automated deployments from GitHub.

See complete guide: **[frontend/docs/DEPLOY_CLOUDFLARE_PAGES.md](../frontend/docs/DEPLOY_CLOUDFLARE_PAGES.md)**

**Key features:**
- Automatic deployments on git push
- SPA routing via `_redirects` file
- Security headers via `_headers` file
- Free SSL certificates
- Global CDN

**Redirect URLs**: Configure in Supabase Dashboard → Authentication → URL Configuration:
```
https://your-site.pages.dev/**
https://your-site.pages.dev/auth/**
https://your-site.pages.dev/login
https://your-site.pages.dev/verify-email
https://your-site.pages.dev/reset-password
```

### Legacy: Azure Storage (Deprecated)

<details>
<summary>Azure Storage Static Website setup (click to expand)</summary>

#### 1. Enable Static Website Hosting

1. Go to Azure Portal → Storage Account → **Static website**
2. Enable static website hosting
3. Set **Index document**: `index.html`
4. Set **Error document**: `index.html` (critical for SPA routing!)
5. Copy the **Primary endpoint** URL

#### 2. Why Error Document = index.html?

Supabase Auth sends email links to routes like `/auth/confirm` and `/auth/callback`. Without this setting, Azure returns 404 for these routes instead of serving the SPA.

**Note**: This method is deprecated. Use Cloudflare Pages for new deployments.

</details>

---

## Supabase Dashboard Configuration

### 1. Auth URL Configuration

Navigate to: **Authentication → URL Configuration**

| Setting | Value |
|---------|-------|
| Site URL | `https://energymonitorstorage.z1.web.core.windows.net/` |

### 2. Redirect URLs Allowlist

Add ALL of these (Authentication → URL Configuration → Redirect URLs):

```
http://localhost:5173/**
https://energymonitorstorage.z1.web.core.windows.net/**
https://energymonitorstorage.z1.web.core.windows.net/auth/**
https://energymonitorstorage.z1.web.core.windows.net/login
https://energymonitorstorage.z1.web.core.windows.net/verify-email
https://energymonitorstorage.z1.web.core.windows.net/reset-password
```

### 3. Email Templates

Navigate to: **Authentication → Email Templates**

Ensure templates use `{{ .SiteURL }}` and `{{ .RedirectTo }}` correctly:
- Confirmation template redirects to confirmation page
- Password reset template redirects to reset page

### 4. Auth Providers

Navigate to: **Authentication → Providers**
- ✅ Enable **Email** provider
- Configure email confirmation (recommended: enabled)

---

## Database Migrations

### Apply Migrations

```bash
cd frontend

# Link to your Supabase project
supabase link --project-ref bhdcbvruzvhmcogxfkil

# Apply all migrations
supabase db push
```

### Migration Files Created

1. **`20260109150000_complete_schema_organizations.sql`**
   - Organizations and org_members tables
   - login_history table
   - disaggregation_predictions table
   - Helper functions for org membership checks

2. **`20260109150001_rls_policies_complete.sql`**
   - Comprehensive RLS policies for all tables
   - Organization-based access control
   - User ownership policies

3. **`20260109150002_storage_policies.sql`**
   - Avatar storage bucket policies

### Verify Tables Exist

Run in Supabase SQL Editor:

```sql
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'public'
ORDER BY table_name;
```

Expected tables:
- `appliances`, `building_appliances`, `buildings`
- `disaggregation_predictions`
- `inference_runs`, `invitations`
- `login_events`, `login_history`
- `model_versions`, `models`
- `org_appliances`, `org_members`, `organizations`
- `predictions`, `profiles`, `readings`
- `user_settings`

### Verify RLS is Enabled

```sql
SELECT tablename, rowsecurity 
FROM pg_tables 
WHERE schemaname = 'public';
```

All tables should show `rowsecurity = true`.

---

## Storage Bucket Setup

### 1. Create Avatars Bucket

Navigate to: **Storage → New bucket**

| Setting | Value |
|---------|-------|
| Name | `avatars` |
| Public | ✅ Yes (for profile image display) |

### 2. Apply Storage Policies

The migration attempts to set policies, but you may need to configure manually:

Navigate to: **Storage → avatars → Policies**

Create these policies:

#### Policy: Users can upload own avatar
- **Operation**: INSERT
- **Target roles**: authenticated
- **Policy definition**:
```sql
(bucket_id = 'avatars' AND auth.uid()::text = (storage.foldername(name))[1])
```

#### Policy: Users can update own avatar
- **Operation**: UPDATE
- **Target roles**: authenticated
- **Policy definition**:
```sql
(bucket_id = 'avatars' AND auth.uid()::text = (storage.foldername(name))[1])
```

#### Policy: Users can delete own avatar
- **Operation**: DELETE
- **Target roles**: authenticated
- **Policy definition**:
```sql
(bucket_id = 'avatars' AND auth.uid()::text = (storage.foldername(name))[1])
```

#### Policy: Public avatar access
- **Operation**: SELECT
- **Target roles**: public
- **Policy definition**:
```sql
(bucket_id = 'avatars')
```

---

## Edge Functions Deployment

### 1. Deploy All Functions

```bash
cd frontend

# Deploy all edge functions
supabase functions deploy invite-user
supabase functions deploy get-dashboard-data
supabase functions deploy upsert-avatar
supabase functions deploy log-auth-event
supabase functions deploy generate-report
supabase functions deploy delete-account
supabase functions deploy admin-invite
supabase functions deploy accept-invite
supabase functions deploy log-login-event
supabase functions deploy get-readings
supabase functions deploy run-inference
supabase functions deploy register-model
supabase functions deploy create-model-version-upload
supabase functions deploy finalize-model-version
supabase functions deploy set-active-model-version
```

### 2. Set Edge Function Secrets (if needed)

```bash
# Set custom secrets (e.g., for Influx integration)
supabase secrets set INFLUX_URL="your-influx-url"
supabase secrets set INFLUX_TOKEN="your-influx-token"
supabase secrets set INFLUX_ORG="your-org"
supabase secrets set INFLUX_BUCKET="your-bucket"

# Set site URL for email redirects
supabase secrets set SITE_URL="https://energymonitorstorage.z1.web.core.windows.net"
```

### Edge Functions Overview

| Function | Purpose | Uses Service Role? |
|----------|---------|-------------------|
| `invite-user` | Invite user to organization | ✅ Yes |
| `get-dashboard-data` | Fetch real dashboard data | ❌ No (uses RLS) |
| `upsert-avatar` | Upload user avatar | ❌ No (uses RLS) |
| `log-auth-event` | Log authentication events | ✅ Yes |
| `generate-report` | Generate energy report | ❌ No (uses RLS) |
| `delete-account` | Delete user account | ✅ Yes |

---

## GitHub Secrets Setup

Add these secrets to your GitHub repository:

1. Go to: **Repository → Settings → Secrets and variables → Actions**
2. Add these secrets:

| Secret Name | Value |
|-------------|-------|
| `VITE_SUPABASE_URL` | `https://bhdcbvruzvhmcogxfkil.supabase.co` |
| `VITE_SUPABASE_PUBLISHABLE_KEY` | `eyJhbGciOiJIUzI1NiIs...` (anon key) |

⚠️ **NEVER add `SUPABASE_SERVICE_ROLE_KEY` to GitHub Secrets for frontend builds!**

---

## Verification Checklist

### ✅ Auth Flow Tests

1. **Sign Up**
   - [ ] Create new account → receives confirmation email
   - [ ] Click email link → redirects to `PROD_SITE_URL` (not localhost)
   - [ ] Profile is created in `profiles` table

2. **Sign In**
   - [ ] Login with email/password works
   - [ ] Login event logged in `login_events` or `login_history`
   - [ ] Session persists (if remember me is enabled)

3. **Password Reset**
   - [ ] Request reset → receives email
   - [ ] Click link → redirects to `PROD_SITE_URL/reset-password`
   - [ ] Can set new password

4. **Invite Flow**
   - [ ] Admin can invite user via `invite-user` function
   - [ ] Invited user receives email
   - [ ] Clicking invite link works correctly

### ✅ SPA Routing Tests

- [ ] Direct navigation to `/app/dashboard` works (no 404)
- [ ] Direct navigation to `/auth/callback` works
- [ ] Browser refresh on any route works

### ✅ RLS Security Tests

1. **Cross-User Access Prevention**
   ```sql
   -- As user A, try to read user B's profile
   SELECT * FROM profiles WHERE id = '<user-b-id>';
   -- Should return empty
   ```

2. **Org-Based Access**
   ```sql
   -- User can only see buildings in their org
   SELECT * FROM buildings;
   -- Should only return accessible buildings
   ```

### ✅ Storage Tests

- [ ] Upload avatar → success
- [ ] Avatar URL updates in profile
- [ ] Avatar is publicly viewable
- [ ] Cannot upload to another user's folder

### ✅ Edge Function Tests

```bash
# Test get-dashboard-data
curl -X POST \
  'https://bhdcbvruzvhmcogxfkil.supabase.co/functions/v1/get-dashboard-data' \
  -H 'Authorization: Bearer <user-jwt>' \
  -H 'Content-Type: application/json' \
  -d '{"building_id": "<building-uuid>"}'
```

### ✅ API Mode Tests

- [ ] Dashboard fetches real data (not demo CSV)
- [ ] Reports generate from database
- [ ] "What's ON now" shows actual predictions

---

## Troubleshooting

### Email Links Go to Wrong URL

1. Check **Site URL** in Supabase Dashboard
2. Verify `redirectTo` in signup/reset calls matches allowlist
3. Ensure Azure Static Website error document is `index.html`

### RLS Blocks All Access

1. Check user is authenticated (`auth.uid()` is not null)
2. Verify policies exist: `SELECT * FROM pg_policies WHERE tablename = 'your_table'`
3. Test with service role in SQL Editor to bypass RLS

### Edge Function 500 Errors

1. Check function logs: `supabase functions logs <function-name>`
2. Verify secrets are set: `supabase secrets list`
3. Test locally: `supabase functions serve`

### Avatar Upload Fails

1. Check storage bucket exists and is public
2. Verify storage policies are created
3. Check file size (max 5MB)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Azure Storage (Static Website)               │
│                    (index.html for all routes)                   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Vite/React SPA                              │
│                  (uses ANON key only)                            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  Supabase     │   │  Supabase     │   │  Supabase     │
│  Auth         │   │  Postgres     │   │  Edge         │
│  (RLS-based)  │   │  (RLS-based)  │   │  Functions    │
└───────────────┘   └───────────────┘   │ (service_role)│
                                        └───────────────┘
```

---

## Security Reminders

1. **ANON key in frontend**: Safe because RLS protects data
2. **SERVICE_ROLE key**: ONLY in Edge Functions, NEVER in browser
3. **RLS enabled**: On ALL public tables
4. **Storage policies**: Restrict uploads to user's own folder
