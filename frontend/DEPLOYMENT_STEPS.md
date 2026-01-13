# Complete Deployment Steps - NILM Energy Monitor

## Current Status ✅
- ✅ Frontend built and deployed to Azure Storage Static Website
- ✅ Security audit completed (all fixes committed)
- ✅ TypeScript strictness enabled
- ⏳ Database migrations need to be applied
- ⏳ Edge functions need to be deployed

---

## Step 1: Apply Database Migrations

You have 13 migration files that need to be applied to your Supabase database.

### Option A: Using Supabase Dashboard (Recommended for Windows)

1. Go to your Supabase Dashboard: https://supabase.com/dashboard/project/bhdcbvruzvhmcogxfkil
2. Navigate to: **SQL Editor**
3. Apply each migration file in order:

#### Migration 1: `20260107134422_9e64fcba-4517-42d6-a1a1-f1e7262fe295.sql`
```bash
# Copy the SQL content and paste into SQL Editor
cd frontend/supabase/migrations
cat 20260107134422_9e64fcba-4517-42d6-a1a1-f1e7262fe295.sql
```
Click **Run** in the SQL Editor

#### Migration 2-13: Repeat for all migrations in order
Apply each file in chronological order (sorted by timestamp).

### Option B: Using Supabase CLI via npx

If you want to try using npx (without global install):

```bash
cd frontend

# Link to your project (only once)
npx supabase link --project-ref bhdcbvruzvhmcogxfkil

# Apply all migrations
npx supabase db push
```

### Verify Migrations Applied

Run this in **Supabase Dashboard → SQL Editor**:

```sql
-- Check all tables exist
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
ORDER BY table_name;
```

Expected tables:
- `appliances`
- `building_appliances`
- `buildings`
- `disaggregation_predictions`
- `inference_runs`
- `invitations`
- `login_events`
- `login_history`
- `model_versions`
- `models`
- `org_appliances`
- `org_members`
- `organizations`
- `predictions`
- `profiles`
- `readings`
- `user_settings`

---

## Step 2: Configure Supabase Authentication

### 2.1 Set Site URL

1. Go to: **Authentication → URL Configuration**
2. Set **Site URL** to: `https://energymonitorstorage.z1.web.core.windows.net/`

### 2.2 Add Redirect URLs

Go to: **Authentication → URL Configuration → Redirect URLs**

Add these URLs (one per line):

```
http://localhost:5173/**
https://energymonitorstorage.z1.web.core.windows.net/**
https://energymonitorstorage.z1.web.core.windows.net/auth/**
https://energymonitorstorage.z1.web.core.windows.net/login
https://energymonitorstorage.z1.web.core.windows.net/verify-email
https://energymonitorstorage.z1.web.core.windows.net/reset-password
```

### 2.3 Configure Email Provider

Go to: **Authentication → Providers → Email**

- ✅ Enable **Email** provider
- Configure confirmation email (recommended: enable)
- Update email templates to use production URLs

---

## Step 3: Set Up Storage Bucket

### 3.1 Create Avatars Bucket

1. Go to: **Storage → New bucket**
2. Configure:
   - **Name**: `avatars`
   - **Public**: ✅ Yes
3. Click **Create bucket**

### 3.2 Apply Storage Policies

Go to: **Storage → avatars → Policies**

Create these 4 policies:

#### Policy 1: Users can upload own avatar
- **Policy name**: `Users can upload own avatar`
- **Allowed operation**: INSERT
- **Target roles**: `authenticated`
- **Policy definition**:
```sql
(bucket_id = 'avatars'::text) AND ((auth.uid())::text = (storage.foldername(name))[1])
```

#### Policy 2: Users can update own avatar
- **Policy name**: `Users can update own avatar`
- **Allowed operation**: UPDATE
- **Target roles**: `authenticated`
- **Policy definition**:
```sql
(bucket_id = 'avatars'::text) AND ((auth.uid())::text = (storage.foldername(name))[1])
```

#### Policy 3: Users can delete own avatar
- **Policy name**: `Users can delete own avatar`
- **Allowed operation**: DELETE
- **Target roles**: `authenticated`
- **Policy definition**:
```sql
(bucket_id = 'avatars'::text) AND ((auth.uid())::text = (storage.foldername(name))[1])
```

#### Policy 4: Public can view avatars
- **Policy name**: `Public avatar access`
- **Allowed operation**: SELECT
- **Target roles**: `public`
- **Policy definition**:
```sql
bucket_id = 'avatars'::text
```

---

## Step 4: Deploy Edge Functions

You have 16 edge functions that need to be deployed.

### Option A: Using Supabase Dashboard (Manual)

Unfortunately, edge functions cannot be created via the dashboard. You must use the CLI.

### Option B: Using npx supabase (Recommended)

```bash
cd frontend

# Deploy each function (replace <function-name> with actual name)
npx supabase functions deploy log-login-event
npx supabase functions deploy log-auth-event
npx supabase functions deploy invite-user
npx supabase functions deploy admin-invite
npx supabase functions deploy accept-invite
npx supabase functions deploy delete-account
npx supabase functions deploy get-readings
npx supabase functions deploy get-dashboard-data
npx supabase functions deploy generate-report
npx supabase functions deploy upsert-avatar
npx supabase functions deploy run-inference
npx supabase functions deploy register-model
npx supabase functions deploy create-model-version-upload
npx supabase functions deploy finalize-model-version
npx supabase functions deploy set-active-model-version
npx supabase functions deploy parse-nilm-csv
```

### Option C: Deploy All at Once

Create a PowerShell script:

```powershell
# deploy-functions.ps1
cd frontend

$functions = @(
    "log-login-event",
    "log-auth-event",
    "invite-user",
    "admin-invite",
    "accept-invite",
    "delete-account",
    "get-readings",
    "get-dashboard-data",
    "generate-report",
    "upsert-avatar",
    "run-inference",
    "register-model",
    "create-model-version-upload",
    "finalize-model-version",
    "set-active-model-version",
    "parse-nilm-csv"
)

foreach ($func in $functions) {
    Write-Host "Deploying $func..." -ForegroundColor Green
    npx supabase functions deploy $func
}

Write-Host "All functions deployed!" -ForegroundColor Green
```

Run it:
```powershell
.\deploy-functions.ps1
```

---

## Step 5: Configure Environment Secrets

Some edge functions need environment secrets.

### Set Secrets via CLI

```bash
cd frontend

# Set site URL for auth redirects
npx supabase secrets set SITE_URL="https://energymonitorstorage.z1.web.core.windows.net"

# Optional: InfluxDB integration (if using API mode)
npx supabase secrets set INFLUX_URL="your-influx-url"
npx supabase secrets set INFLUX_TOKEN="your-influx-token"
npx supabase secrets set INFLUX_ORG="your-org"
npx supabase secrets set INFLUX_BUCKET="your-bucket"
```

### Verify Secrets

```bash
npx supabase secrets list
```

---

## Step 6: Rebuild and Redeploy Frontend

Now that backend is configured, rebuild with production environment:

```bash
cd frontend

# Create .env.production (already gitignored)
cat > .env.production << EOF
VITE_SUPABASE_URL=https://bhdcbvruzvhmcogxfkil.supabase.co
VITE_SUPABASE_PUBLISHABLE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJoZGNidnJ1enZobWNvZ3hma2lsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Njc5NTc4NDgsImV4cCI6MjA4MzUzMzg0OH0.zFAx3Wwz6Gyy9dfCc0xxe9QomaembUI8b4bmM23IrRI
VITE_DEMO_MODE=false
EOF

# Build for production
npm run build

# Upload to Azure Storage
az storage blob upload-batch -s ./dist -d '$web' --account-name energymonitorstorage --overwrite
```

---

## Step 7: Verification Tests

### 7.1 Test Sign Up Flow

1. Go to: https://energymonitorstorage.z1.web.core.windows.net/signup
2. Create a new account
3. Check email for confirmation link
4. Click link → should redirect to production URL (not localhost)
5. Verify profile created:
   ```sql
   -- In Supabase SQL Editor
   SELECT * FROM profiles ORDER BY created_at DESC LIMIT 1;
   ```

### 7.2 Test Sign In Flow

1. Go to: https://energymonitorstorage.z1.web.core.windows.net/login
2. Log in with created account
3. Should redirect to `/app/dashboard`
4. Verify login event:
   ```sql
   SELECT * FROM login_events ORDER BY logged_at DESC LIMIT 1;
   ```

### 7.3 Test SPA Routing

1. Navigate to: https://energymonitorstorage.z1.web.core.windows.net/app/dashboard
2. Should load (no 404)
3. Refresh page → should still work
4. Navigate to: https://energymonitorstorage.z1.web.core.windows.net/app/settings/profile
5. Should work without 404

### 7.4 Test Avatar Upload

1. Go to: https://energymonitorstorage.z1.web.core.windows.net/app/settings/profile
2. Click avatar or "Change Photo"
3. Upload an image
4. Should succeed and display

### 7.5 Test Admin Route Guard

1. As a non-admin user, try to access:
   https://energymonitorstorage.z1.web.core.windows.net/app/settings/users
2. Should redirect to `/app/dashboard` (not allowed)

---

## Step 8: Monitor and Debug

### View Edge Function Logs

```bash
cd frontend

# View logs for specific function
npx supabase functions logs log-login-event

# Follow logs in real-time
npx supabase functions logs log-login-event --follow
```

### Check RLS Policies

```sql
-- Verify RLS is enabled
SELECT tablename, rowsecurity
FROM pg_tables
WHERE schemaname = 'public';

-- View all policies
SELECT * FROM pg_policies
WHERE schemaname = 'public'
ORDER BY tablename, policyname;
```

### Test Edge Functions

```bash
# Test get-dashboard-data function
curl -X POST \
  'https://bhdcbvruzvhmcogxfkil.supabase.co/functions/v1/get-dashboard-data' \
  -H 'Authorization: Bearer YOUR_USER_JWT_TOKEN' \
  -H 'Content-Type: application/json' \
  -d '{"building_id": "your-building-uuid"}'
```

---

## Troubleshooting

### Issue: Email links go to localhost instead of production

**Solution**:
1. Check **Site URL** in Supabase Dashboard → Authentication → URL Configuration
2. Must be: `https://energymonitorstorage.z1.web.core.windows.net/`
3. Update email templates to use `{{ .SiteURL }}`

### Issue: 404 on refresh or direct navigation

**Solution**:
1. Verify Azure Storage static website error document is set to `index.html`
2. Go to: Azure Portal → Storage Account → Static website
3. Set **Error document path** to: `index.html`

### Issue: RLS blocks all database access

**Solution**:
1. Verify user is authenticated (check JWT token)
2. Check RLS policies exist:
   ```sql
   SELECT * FROM pg_policies WHERE tablename = 'profiles';
   ```
3. Test query as service role (bypasses RLS) to verify data exists

### Issue: Avatar upload fails with 403

**Solution**:
1. Check storage bucket is public
2. Verify storage policies are created
3. Check file path format: `avatars/{user_id}/{filename}`

### Issue: Edge function returns 500 error

**Solution**:
1. Check function logs: `npx supabase functions logs <function-name>`
2. Verify secrets are set: `npx supabase secrets list`
3. Test function locally: `npx supabase functions serve`

---

## Quick Reference

### Important URLs

| Service | URL |
|---------|-----|
| **Production Site** | https://energymonitorstorage.z1.web.core.windows.net/ |
| **Supabase Dashboard** | https://supabase.com/dashboard/project/bhdcbvruzvhmcogxfkil |
| **Azure Portal** | https://portal.azure.com/ |

### Important Commands

```bash
# Link to Supabase project (one-time)
npx supabase link --project-ref bhdcbvruzvhmcogxfkil

# Apply database migrations
npx supabase db push

# Deploy a single edge function
npx supabase functions deploy <function-name>

# Set environment secret
npx supabase secrets set KEY="value"

# View function logs
npx supabase functions logs <function-name>

# Build frontend
npm run build

# Upload to Azure
az storage blob upload-batch -s ./dist -d '$web' --account-name energymonitorstorage --overwrite
```

---

## Security Checklist

Before going to production:

- [ ] All database migrations applied
- [ ] RLS enabled on all tables (verify with SQL query)
- [ ] Storage policies created for avatars bucket
- [ ] Site URL configured in Supabase Auth
- [ ] Redirect URLs added to allowlist
- [ ] Edge functions deployed
- [ ] Environment secrets set (SITE_URL at minimum)
- [ ] `.env.production` exists locally but NOT in git
- [ ] Frontend rebuilt with production environment variables
- [ ] Latest build uploaded to Azure Storage
- [ ] Test full signup/login flow
- [ ] Test password reset flow
- [ ] Test avatar upload
- [ ] Test admin route protection
- [ ] Verify console logs stripped in production bundle

---

## Next Steps After Deployment

1. **Monitor Error Logs**: Set up error tracking (Sentry, LogRocket)
2. **Performance Monitoring**: Use Azure Application Insights
3. **Backup Database**: Schedule regular Supabase backups
4. **Rotate Keys**: If any secrets were exposed, rotate immediately
5. **User Testing**: Have team members test all flows
6. **Documentation**: Update team docs with deployment procedures
7. **CI/CD**: Set up GitHub Actions for automated deployments

---

## Support

If you encounter issues:

1. Check the [SUPABASE_SETUP.md](./docs/SUPABASE_SETUP.md) detailed guide
2. Check the [SECURITY.md](./docs/SECURITY.md) for security best practices
3. Review Supabase function logs for errors
4. Check Azure Storage logs for 404s
5. Verify RLS policies in Supabase Dashboard

Good luck with your deployment! 🚀
