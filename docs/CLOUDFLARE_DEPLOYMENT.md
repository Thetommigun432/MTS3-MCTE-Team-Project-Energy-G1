# Cloudflare Pages Deployment Guide

**Project:** NILM Energy Monitor Frontend
**Date:** 2026-01-21
**Status:** Ready for Deployment

---

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Deployment Options](#deployment-options)
- [Option 1: Deploy via Cloudflare Dashboard (Recommended)](#option-1-deploy-via-cloudflare-dashboard-recommended)
- [Option 2: Deploy via Wrangler CLI](#option-2-deploy-via-wrangler-cli)
- [Environment Variables Configuration](#environment-variables-configuration)
- [Custom Domain Setup](#custom-domain-setup)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

---

## Overview

This guide walks you through deploying the React/Vite frontend to **Cloudflare Pages**, which provides:

✅ **Global CDN** - Fast delivery worldwide
✅ **Automatic HTTPS** - Free SSL certificates
✅ **Unlimited bandwidth** - No usage limits
✅ **Git integration** - Auto-deploy on push
✅ **Preview deployments** - Test before production
✅ **Free tier** - Generous limits for most projects

**Deployment Time:** ~5-10 minutes for first deployment

---

## Prerequisites

### Required

1. **Cloudflare Account**
   - Sign up at https://dash.cloudflare.com/sign-up
   - Free tier is sufficient

2. **GitHub Repository**
   - Code must be pushed to GitHub
   - Cloudflare will connect to your repo

3. **Backend Deployed**
   - FastAPI backend should be deployed first (e.g., Railway)
   - You'll need the backend URL for environment variables

### Optional

4. **Wrangler CLI** (for CLI deployment)
   ```bash
   npm install -g wrangler
   wrangler login
   ```

---

## Deployment Options

### Option 1: Cloudflare Dashboard (Recommended)
**Best for:** First-time deployment, teams, automatic deployments
**Pros:** Easy setup, Git integration, automatic deployments
**Cons:** Requires GitHub connection

### Option 2: Wrangler CLI
**Best for:** Advanced users, CI/CD pipelines, manual control
**Pros:** Scriptable, flexible, no Git required
**Cons:** Manual deployment required

---

## Option 1: Deploy via Cloudflare Dashboard (Recommended)

### Step 1: Connect to GitHub

1. **Go to Cloudflare Pages**
   - Login to https://dash.cloudflare.com
   - Navigate to **Workers & Pages** → **Pages**
   - Click **"Create application"** → **"Connect to Git"**

2. **Authorize GitHub**
   - Click **"Connect GitHub"**
   - Authorize Cloudflare to access your repositories
   - Select the repository: `MTS3-MCTE-Team-Project-Energy-G1`

3. **Configure Build Settings**

   **Project name:** `nilm-energy-monitor` (or your preferred name)

   **Production branch:** `main` (or `backend` if deploying from backend branch)

   **Build Settings:**
   ```
   Framework preset:         Vite
   Build command:            npm run build
   Build output directory:   dist
   Root directory:           apps/web
   ```

   **⚠️ Important:** Set `Root directory` to `apps/web` since we're in a monorepo!

### Step 2: Configure Environment Variables

Click **"Add environment variable"** and add the following:

#### Required Variables

| Variable Name | Value | Notes |
|---------------|-------|-------|
| `VITE_API_BASE_URL` | `https://your-backend.railway.app` | ⚠️ **Replace with your actual Railway backend URL** |
| `VITE_SUPABASE_URL` | `https://your-project.supabase.co` | From Supabase dashboard |
| `VITE_SUPABASE_PUBLISHABLE_KEY` | `eyJhbGc...` | Anon/publishable key from Supabase |

#### Optional Variables

| Variable Name | Value | Notes |
|---------------|-------|-------|
| `VITE_DEMO_MODE` | `false` | Set to `true` to enable demo mode |
| `VITE_DEMO_EMAIL` | `demo@example.com` | Demo user email (if demo mode enabled) |
| `VITE_DEMO_PASSWORD` | `demo123` | Demo password (if demo mode enabled) |

**⚠️ Security Note:**
- Only use `VITE_*` prefixed variables in frontend
- Never expose backend secrets (service role keys, admin tokens, etc.)

### Step 3: Deploy

1. Click **"Save and Deploy"**
2. Cloudflare will:
   - Clone your repository
   - Install dependencies (`npm install`)
   - Run build command (`npm run build`)
   - Deploy to global CDN

**Build Time:** ~3-5 minutes

### Step 4: Get Your URL

After deployment completes:
- Production URL: `https://nilm-energy-monitor.pages.dev`
- Or your custom domain (see [Custom Domain Setup](#custom-domain-setup))

---

## Option 2: Deploy via Wrangler CLI

### Step 1: Install Wrangler

```bash
# Install globally
npm install -g wrangler

# Login to Cloudflare
wrangler login
```

### Step 2: Configure wrangler.toml

The `apps/web/wrangler.toml` is already configured:

```toml
name = "nilm-energy-monitor"
pages_build_output_dir = "dist"
compatibility_date = "2024-01-01"
```

### Step 3: Build the Project

```bash
# Navigate to frontend
cd apps/web

# Install dependencies
npm install

# Build for production
npm run build
```

Verify the build succeeded:
```bash
ls -la dist/
# Should see: index.html, assets/, etc.
```

### Step 4: Deploy

```bash
# Deploy to Cloudflare Pages
npx wrangler pages deploy dist --project-name=nilm-energy-monitor

# First time deployment will prompt:
# - Do you want to create a new project? → Yes
# - Enter production branch: → main (or backend)
```

**Output:**
```
✨ Compiled Worker successfully
✨ Uploading...
✨ Deployment complete!
🌐 https://nilm-energy-monitor.pages.dev
```

### Step 5: Set Environment Variables via CLI

```bash
# Set production environment variables
npx wrangler pages secret put VITE_API_BASE_URL --project-name=nilm-energy-monitor
# Enter value: https://your-backend.railway.app

npx wrangler pages secret put VITE_SUPABASE_URL --project-name=nilm-energy-monitor
# Enter value: https://your-project.supabase.co

npx wrangler pages secret put VITE_SUPABASE_PUBLISHABLE_KEY --project-name=nilm-energy-monitor
# Enter value: eyJhbGc...
```

**After setting variables, redeploy:**
```bash
cd apps/web
npm run build
npx wrangler pages deploy dist --project-name=nilm-energy-monitor
```

---

## Environment Variables Configuration

### Finding Your Backend URL

**If deployed to Railway:**
1. Go to Railway dashboard: https://railway.app
2. Select your backend project
3. Go to **Settings** → **Domains**
4. Copy the generated domain: `https://your-app.up.railway.app`

**Use this URL for `VITE_API_BASE_URL`**

### Finding Your Supabase Credentials

1. Go to Supabase dashboard: https://app.supabase.com
2. Select your project
3. Go to **Settings** → **API**
4. Copy:
   - **Project URL** → `VITE_SUPABASE_URL`
   - **anon public** key → `VITE_SUPABASE_PUBLISHABLE_KEY`

### Verifying Environment Variables

After deployment, check that variables are set:

**Via Dashboard:**
1. Go to Cloudflare Pages dashboard
2. Select your project
3. Go to **Settings** → **Environment variables**
4. Verify all variables are listed under "Production"

**Via CLI:**
```bash
npx wrangler pages secret list --project-name=nilm-energy-monitor
```

---

## Custom Domain Setup

### Option 1: Use Cloudflare Domain (Free)

If you already have a domain on Cloudflare:

1. **Go to your Cloudflare Pages project**
2. Click **"Custom domains"** tab
3. Click **"Set up a custom domain"**
4. Enter your domain: `app.yourdomain.com`
5. Click **"Activate domain"**

Cloudflare will automatically:
- Create DNS records
- Provision SSL certificate
- Redirect traffic to your Pages deployment

**Propagation time:** ~5-10 minutes

### Option 2: Use External Domain

If your domain is elsewhere (GoDaddy, Namecheap, etc.):

1. **Get your Pages domain:** `nilm-energy-monitor.pages.dev`

2. **Add DNS records at your domain provider:**
   ```
   Type: CNAME
   Name: app (or @, or www)
   Value: nilm-energy-monitor.pages.dev
   TTL: Auto
   ```

3. **Add custom domain in Cloudflare:**
   - Follow same steps as Option 1
   - Cloudflare will verify DNS records
   - SSL certificate will be provisioned

**Propagation time:** ~1-24 hours (depends on DNS provider)

---

## Verification

### Post-Deployment Checklist

After deployment, verify everything works:

#### ✅ **1. Frontend Loads**
```bash
curl -I https://nilm-energy-monitor.pages.dev
# Expected: HTTP/2 200
```

Open in browser:
- https://nilm-energy-monitor.pages.dev
- Should see login page or dashboard

#### ✅ **2. Backend Connection Works**

Open browser console (F12) and check:
- Network tab: Look for API calls to your backend URL
- Console: Should NOT see CORS errors
- Console: Should see `[API] GET /analytics/...` logs (in dev mode)

#### ✅ **3. Environment Variables Are Set**

In browser console, run:
```javascript
console.log({
  apiUrl: import.meta.env.VITE_API_BASE_URL,
  supabaseUrl: import.meta.env.VITE_SUPABASE_URL
});
```

Should show your configured URLs (not undefined).

#### ✅ **4. Authentication Works**

- Try logging in with Supabase credentials
- Should successfully authenticate
- Dashboard should load data from backend

#### ✅ **5. CORS Is Configured**

**Backend side (Railway):**
```bash
# SSH into Railway or check environment variables
# CORS_ORIGINS should include your Cloudflare URL
CORS_ORIGINS=http://localhost:8080,https://nilm-energy-monitor.pages.dev
```

If CORS errors occur:
1. Update backend `CORS_ORIGINS` environment variable
2. Restart backend service
3. Test again

---

## Troubleshooting

### Issue: Build Fails with "Module not found"

**Symptom:**
```
✘ [ERROR] Could not resolve "@/lib/..."
```

**Solution:**
1. Check that `Root directory` is set to `apps/web`
2. Verify `tsconfig.json` has correct path mappings
3. Check `vite.config.ts` has path aliases configured

### Issue: Environment Variables Not Working

**Symptom:**
- `import.meta.env.VITE_API_BASE_URL` is `undefined`
- API calls fail with CORS errors

**Solution:**
1. **Verify variables are set:**
   - Go to Cloudflare Pages → Settings → Environment variables
   - Check "Production" environment
2. **Redeploy after setting variables:**
   ```bash
   # Trigger a new deployment
   git commit --allow-empty -m "Redeploy"
   git push
   ```
   Or click "Retry deployment" in dashboard

### Issue: CORS Errors in Browser Console

**Symptom:**
```
Access to fetch at 'https://backend.railway.app/analytics/readings'
from origin 'https://nilm-energy-monitor.pages.dev' has been blocked by CORS
```

**Solution:**

1. **Update backend CORS configuration:**
   ```bash
   # In Railway backend settings
   CORS_ORIGINS=http://localhost:8080,https://nilm-energy-monitor.pages.dev
   ```

2. **Restart backend:**
   - Railway: Redeploy the backend service

3. **Verify CORS headers:**
   ```bash
   curl -I -H "Origin: https://nilm-energy-monitor.pages.dev" \
     https://your-backend.railway.app/live

   # Should include:
   # Access-Control-Allow-Origin: https://nilm-energy-monitor.pages.dev
   ```

### Issue: 404 on Page Refresh

**Symptom:**
- Direct URL navigation or page refresh returns 404
- Only works when navigating from home page

**Solution:**

Cloudflare Pages should automatically handle SPA routing. If not:

1. **Add `_redirects` file to public directory:**
   ```bash
   # apps/web/public/_redirects
   /* /index.html 200
   ```

2. **Or add redirect rule in `wrangler.toml`:**
   ```toml
   [[redirects]]
   from = "/*"
   to = "/index.html"
   status = 200
   ```

3. **Redeploy.**

### Issue: Build Size Too Large

**Symptom:**
```
✘ [ERROR] The build failed with exit code 1
Output size exceeds limit
```

**Solution:**

1. **Check build output size:**
   ```bash
   cd apps/web
   npm run build
   du -sh dist/
   ```

2. **Optimize if needed:**
   - Enable code splitting in `vite.config.ts`
   - Lazy load routes
   - Optimize images

Cloudflare Pages limit: **25 MB per deployment** (should be sufficient for this project)

### Issue: Deployment Stuck

**Symptom:**
- Build has been running for >10 minutes
- No progress updates

**Solution:**
1. Click "Cancel deployment" in dashboard
2. Check build logs for errors
3. Retry deployment

Common causes:
- Network issues during `npm install`
- Node version mismatch
- Dependency conflicts

---

## Continuous Deployment

### Automatic Deployments on Git Push

If using Cloudflare Dashboard (Option 1):

**Production deployments:**
- Every push to `main` branch triggers production deployment

**Preview deployments:**
- Every push to other branches creates preview URL
- Preview URL: `https://<branch-name>.nilm-energy-monitor.pages.dev`

**Pull Request deployments:**
- Each PR gets a unique preview URL
- Comment automatically added to PR with preview link

### Manual Deployments via CLI

```bash
# Build locally
cd apps/web
npm run build

# Deploy to production
npx wrangler pages deploy dist --project-name=nilm-energy-monitor --branch=main

# Deploy as preview (branch)
npx wrangler pages deploy dist --project-name=nilm-energy-monitor --branch=feature-xyz
```

---

## Performance Optimization

### Cloudflare Pages Performance Features

Automatically enabled:
- ✅ **Brotli compression** - Smaller file sizes
- ✅ **HTTP/2** - Faster page loads
- ✅ **Global CDN** - 200+ data centers
- ✅ **Automatic caching** - Edge caching for static assets

### Additional Optimizations

1. **Enable Cloudflare Analytics** (free)
   - Pages dashboard → Analytics tab
   - Monitor page views, performance, geography

2. **Configure Cache Headers**
   - Add `_headers` file to `public/` directory:
   ```
   /assets/*
     Cache-Control: public, max-age=31536000, immutable

   /
     Cache-Control: public, max-age=0, must-revalidate
   ```

3. **Use Cloudflare Images** (optional, paid)
   - Automatic image optimization
   - WebP conversion
   - Responsive images

---

## Monitoring & Logs

### View Deployment Logs

**Via Dashboard:**
1. Go to Cloudflare Pages dashboard
2. Select your project
3. Click on a deployment
4. View **Build logs** and **Function logs**

**Via CLI:**
```bash
# List deployments
npx wrangler pages deployment list --project-name=nilm-energy-monitor

# View deployment logs
npx wrangler pages deployment tail --project-name=nilm-energy-monitor
```

### Analytics

**Cloudflare Web Analytics** (free):
1. Pages dashboard → Analytics
2. View:
   - Page views
   - Visitors
   - Page load time
   - Geography
   - Browsers/devices

---

## Cost Estimate

### Cloudflare Pages Pricing

**Free Tier (recommended for this project):**
- ✅ Unlimited requests
- ✅ Unlimited bandwidth
- ✅ 500 builds/month
- ✅ 1 build at a time
- ✅ 100 custom domains

**Paid Tier ($20/month):**
- Everything in Free
- 5000 builds/month
- 5 concurrent builds
- Access to advanced features

**For this project:** Free tier is sufficient unless you have very frequent deployments.

---

## Next Steps

After successful deployment:

1. **✅ Test the deployed frontend**
   - Open your Pages URL
   - Test login, dashboard, all features
   - Check browser console for errors

2. **✅ Update backend CORS**
   - Add your Pages URL to `CORS_ORIGINS`
   - Restart backend

3. **✅ Set up custom domain** (optional)
   - Follow [Custom Domain Setup](#custom-domain-setup)

4. **✅ Enable monitoring**
   - Turn on Cloudflare Analytics
   - Set up alerts if desired

5. **✅ Document URLs**
   - Update README with production URLs
   - Share with team

---

## Quick Reference

### Deployment Commands

```bash
# Build locally
cd apps/web
npm install
npm run build

# Deploy via Wrangler
npx wrangler pages deploy dist --project-name=nilm-energy-monitor

# Set environment variable
npx wrangler pages secret put VARIABLE_NAME --project-name=nilm-energy-monitor

# List deployments
npx wrangler pages deployment list --project-name=nilm-energy-monitor
```

### URLs

- **Dashboard:** https://dash.cloudflare.com
- **Pages:** https://dash.cloudflare.com/pages
- **Your site:** https://nilm-energy-monitor.pages.dev
- **Wrangler docs:** https://developers.cloudflare.com/pages/

---

## Support Resources

- **Cloudflare Pages Docs:** https://developers.cloudflare.com/pages/
- **Cloudflare Community:** https://community.cloudflare.com/
- **Wrangler CLI:** https://developers.cloudflare.com/workers/wrangler/
- **Project Integration Audit:** `docs/integration-audit.md`
- **Local Development:** `docs/LOCAL_DEVELOPMENT.md`

---

**Deployment Guide Complete** ✅

Follow the steps above to deploy your NILM Energy Monitor frontend to Cloudflare Pages!
