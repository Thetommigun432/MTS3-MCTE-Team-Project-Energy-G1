# Cloudflare Pages Deployment Guide

Deploy the NILM Energy Monitor frontend to Cloudflare Pages with automated builds from GitHub.

## Prerequisites

- **Cloudflare Account**: Free tier works fine
- **GitHub Repository**: Code must be in GitHub
- **Supabase Credentials**: Project URL and anon key

---

## Step 1: Cloudflare Pages Setup

### 1.1 Connect GitHub Repository

1. Log in to [Cloudflare Dashboard](https://dash.cloudflare.com/)
2. Navigate to: **Workers & Pages** → **Create application** → **Pages**
3. Click **Connect to Git**
4. Select **GitHub** and authorize Cloudflare
5. Select repository: `MTS3-MCTE-Team-Project-Energy-G1`
6. Select branch: `frontend`
7. Click **Begin setup**

### 1.2 Configure Build Settings

> **IMPORTANT:** This is a monorepo with npm workspaces. Settings below are critical.

**Cloudflare Pages Settings:**

| Setting | Value |
|---------|-------|
| **Framework preset** | `None` (manual configuration) |
| **Root directory** | *(leave empty - repo root)* |
| **Build command** | `npm run build --workspace=apps/web` |
| **Build output directory** | `apps/web/dist` |

**Advanced settings:**
- **Node.js version**: `20` (or set `NODE_VERSION=20` env var)

Click **Save and Deploy**

### 1.3 Alternative: Using .nvmrc

The repo includes a `.nvmrc` file at the root specifying Node version. Cloudflare should auto-detect this.

---

## Step 2: Environment Variables

### 2.1 Required Variables

Navigate to: **Pages project** → **Settings** → **Environment variables**

**Production variables:**

| Variable | Value | Where to find |
|----------|-------|---------------|
| `VITE_SUPABASE_URL` | `https://xxx.supabase.co` | Supabase Dashboard → Settings → API |
| `VITE_SUPABASE_ANON_KEY` | `eyJhb...` (anon key) | Supabase Dashboard → Settings → API |
| `VITE_BACKEND_URL` | `https://your-backend.railway.app` | Railway Dashboard |

**Optional variables:**

| Variable | Value | Purpose |
|----------|-------|---------|
| `VITE_DEMO_MODE` | `false` | Disable demo login in production |
| `VITE_LOCAL_MODE` | `false` | Disable local InfluxDB mode |
| `NODE_VERSION` | `20` | Explicit Node.js version |

### 2.2 Apply to Environments

- **Production**: Set all variables
- **Preview**: Use same values OR separate test Supabase project

Click **Save** after adding each variable.

---

## Step 3: Verify Build Locally

Before pushing, always verify the build works with a clean install:

```bash
# From repo root
rm -rf node_modules apps/web/node_modules apps/web/dist

# Clean install (mimics Cloudflare)
npm ci --progress=false

# Build the frontend
npm run build --workspace=apps/web

# Verify output exists
ls apps/web/dist/
# Should show: index.html, assets/, etc.
```

---

## Step 4: Deploy and Verify

### 4.1 Trigger Deployment

- **Automatic**: Push to `frontend` branch triggers auto-deploy
- **Manual**: Dashboard → **Deployments** → **Retry deployment**

Wait 2-3 minutes for build to complete.

### 4.2 Get Deployment URL

After successful deployment:
- Production URL: `https://nilm-energy-monitor.pages.dev`
- Or: `https://[project-slug].pages.dev`

### 4.3 SPA Routing Verification

**Test deep links work** (must NOT 404):

1. Open: `https://your-site.pages.dev/app/dashboard`
   - Should load dashboard directly (not 404)

2. Open: `https://your-site.pages.dev/app/reports`
   - Should load reports page

3. **Refresh test**: On any `/app/*` page, press `F5`
   - Should reload same page (not 404)

### 4.4 Security Headers Verification

```bash
curl -I https://your-site.pages.dev/
```

**Expected response headers:**
```
x-content-type-options: nosniff
x-frame-options: DENY
referrer-policy: strict-origin-when-cross-origin
```

---

## Step 5: Update Supabase Redirect URLs

1. Go to: [Supabase Dashboard](https://supabase.com/dashboard)
2. Select your project
3. Navigate to: **Authentication** → **URL Configuration**
4. **Add** to **Redirect URLs**:
   ```
   https://your-site.pages.dev/**
   https://your-site.pages.dev/auth/**
   https://your-site.pages.dev/login
   https://your-site.pages.dev/verify-email
   https://your-site.pages.dev/reset-password
   ```
5. **Site URL**: Set to `https://your-site.pages.dev`
6. Click **Save**

---

## Troubleshooting

### Build Fails: "npm ci can only install packages when your package.json and package-lock.json are in sync"

**Cause**: Lock file out of sync with package.json

**Fix:**
```bash
# From repo root
rm -rf node_modules apps/web/node_modules
npm install
# Commit the updated package-lock.json
git add package-lock.json
git commit -m "fix: sync package-lock.json"
git push
```

### Build Fails: "Module not found"

**Cause**: Cloudflare running from wrong directory

**Fix:**
1. Verify **Root directory** is empty (repo root)
2. Verify **Build command** is `npm run build --workspace=apps/web`
3. Verify **Build output directory** is `apps/web/dist`

### Routes Return 404

**Cause**: Missing `_redirects` file

**Fix:**
1. Verify `apps/web/public/_redirects` exists
2. Content: `/* /index.html 200`
3. Redeploy

### Environment Variables Not Working

**Cause**: Variables not set or misspelled

**Fix:**
1. Check spelling (must start with `VITE_`)
2. Verify set in **Production** environment
3. Redeploy after adding variables

---

## Quick Reference: Cloudflare Pages Settings

Copy-paste settings for Cloudflare Pages:

```
Root directory:           (empty - repo root)
Build command:            npm run build --workspace=apps/web
Build output directory:   apps/web/dist
```

Environment variables:
```
NODE_VERSION=20
VITE_SUPABASE_URL=https://xxx.supabase.co
VITE_SUPABASE_ANON_KEY=eyJ...
VITE_BACKEND_URL=https://your-backend.railway.app
```

---

## Success Checklist

- [ ] Cloudflare Pages project created
- [ ] Build command: `npm run build --workspace=apps/web`
- [ ] Output directory: `apps/web/dist`
- [ ] Root directory: empty (repo root)
- [ ] Environment variables configured
- [ ] Build succeeds (green checkmark)
- [ ] Site accessible at `*.pages.dev` URL
- [ ] Deep links work (`/app/dashboard` loads directly)
- [ ] Refresh works on any route (no 404)
- [ ] Supabase redirect URLs updated

---

## Support

- **Cloudflare Pages Docs**: https://developers.cloudflare.com/pages/
- **npm Workspaces**: https://docs.npmjs.com/cli/v10/using-npm/workspaces
