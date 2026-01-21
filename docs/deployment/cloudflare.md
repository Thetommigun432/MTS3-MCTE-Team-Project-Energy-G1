# Cloudflare Pages Deployment Guide

Deploy the NILM Energy Monitor frontend to Cloudflare Pages with automated builds from GitHub.

## Prerequisites

- **Cloudflare Account**: Free tier works fine
- **GitHub Repository**: Code must be in GitHub
- **Supabase Credentials**: Project URL and anon key

---

## Step 1: Initial Cloudflare Pages Setup

### 1.1 Connect GitHub Repository

1. Log in to [Cloudflare Dashboard](https://dash.cloudflare.com/)
2. Navigate to: **Workers & Pages** → **Create application** → **Pages**
3. Click **Connect to Git**
4. Select **GitHub** and authorize Cloudflare
5. Select repository: `MTS3-MCTE-Team-Project-Energy-G1`
6. Click **Begin setup**

### 1.2 Configure Build Settings

**Framework preset**: Select **Vite**

**Build configuration:**
- **Build command**: `npm run build`
- **Build output directory**: `dist`
- **Root directory**: `frontend` ⚠️ IMPORTANT (monorepo structure)
- **Environment variables**: (configure in next step)

**Advanced settings:**
- **Node.js version**: `20` (reads from `.nvmrc`)
- **Branch deployments**: Enable for `main` branch

Click **Save and Deploy**

---

## Step 2: Environment Variables

### 2.1 Required Variables

Navigate to: **Pages project** → **Settings** → **Environment variables**

**Production variables:**

| Variable | Value | Where to find |
|----------|-------|---------------|
| `VITE_SUPABASE_URL` | `https://bhdcbvruzvhmcogxfkil.supabase.co` | Supabase Dashboard → Settings → API |
| `VITE_SUPABASE_PUBLISHABLE_KEY` | `eyJhb...` (anon key) | Supabase Dashboard → Settings → API |
| `VITE_SUPABASE_PROJECT_ID` | `bhdcbvruzvhmcogxfkil` | From Supabase URL |


**Optional variables:**

| Variable | Value | Purpose |
|----------|-------|---------|
| `VITE_DEMO_MODE` | `false` | Disable demo login in production |
| `VITE_LOCAL_MODE` | `false` | Disable local InfluxDB mode |

### 2.2 Apply to Environments

- **Production**: Set all variables
- **Preview**: Use same values OR separate demo Supabase project

Click **Save** after adding each variable.

---

## Step 3: Deploy and Verify

### 3.1 Trigger Deployment

- **Automatic**: Push to `main` branch triggers auto-deploy
- **Manual**: Dashboard → **Deployments** → **Retry deployment**

Wait 2-3 minutes for build to complete.

### 3.2 Get Deployment URL

After successful deployment:
- Production URL: `https://nilm-energy-monitor.pages.dev`
- Or: `https://[project-slug].pages.dev`

### 3.3 SPA Routing Verification

**Test deep links work** (must NOT 404):

1. Open: `https://your-site.pages.dev/app/dashboard`
   - Should load dashboard directly (not 404)

2. Open: `https://your-site.pages.dev/app/reports`
   - Should load reports page

3. Open: `https://your-site.pages.dev/login`
   - Should load login page

4. **Refresh test**: On any `/app/*` page, press `F5`
   - Should reload same page (not 404 or redirect to home)

### 3.4 Security Headers Verification

**Check headers with curl:**

```bash
curl -I https://your-site.pages.dev/
```

**Expected response headers:**
```
HTTP/2 200
x-content-type-options: nosniff
x-frame-options: DENY
referrer-policy: strict-origin-when-cross-origin
cache-control: public, max-age=0, must-revalidate
```

**Check asset caching:**

```bash
curl -I https://your-site.pages.dev/assets/index-[hash].js
```

**Expected:**
```
cache-control: public, max-age=31536000, immutable
```

---

## Step 4: Update Supabase Redirect URLs

### 4.1 Add Cloudflare Pages URL to Supabase

1. Go to: [Supabase Dashboard](https://supabase.com/dashboard)
2. Select project: `bhdcbvruzvhmcogxfkil`
3. Navigate to: **Authentication** → **URL Configuration**
4. **Add** to **Redirect URLs**:
   ```
   https://your-site.pages.dev/auth/**
   https://your-site.pages.dev/login
   https://your-site.pages.dev/verify-email
   https://your-site.pages.dev/reset-password
   ```

5. **Site URL**: Set to `https://your-site.pages.dev`
6. Click **Save**

### 4.2 Test Authentication Flow

1. Go to: `https://your-site.pages.dev/login`
2. Try to sign in
3. Verify redirect back to dashboard after login
4. Test email verification link (if applicable)

---

## Step 5: Custom Domain (Optional)

### 5.1 Add Custom Domain

1. Cloudflare Dashboard → Pages project → **Custom domains**
2. Click **Set up a custom domain**
3. Enter domain: `energy.yourdomain.com`
4. Follow DNS configuration instructions
5. Cloudflare provisions SSL automatically (free)

### 5.2 Update Supabase Redirect URLs

Add custom domain URLs to Supabase redirect list:
```
https://energy.yourdomain.com/**
```

---

## Troubleshooting

### Build Fails: "Module not found"

**Cause**: Missing dependencies or wrong root directory

**Fix:**
1. Check **Root directory** is set to `frontend`
2. Verify `package.json` has all dependencies
3. Check build logs for specific missing module

### Routes Return 404

**Cause**: Missing `_redirects` file or wrong syntax

**Fix:**
1. Verify `frontend/public/_redirects` exists
2. Content: `/*  /index.html  200`
3. Redeploy

### Environment Variables Not Working

**Cause**: Variables not set or misspelled

**Fix:**
1. Check spelling (must start with `VITE_`)
2. Verify set in **Production** environment
3. Redeploy after adding variables

### Security Headers Missing

**Cause**: Missing `_headers` file

**Fix:**
1. Verify `frontend/public/_headers` exists
2. Check syntax (no tabs, proper spacing)
3. Test with `curl -I https://your-site.pages.dev/`

---

## Comparison: Azure vs Cloudflare Pages

| Feature | Azure Storage | Cloudflare Pages |
|---------|---------------|------------------|
| **Deployment** | Manual (Azure CLI) | Auto (GitHub push) |
| **SPA Config** | `staticwebapp.config.json` | `_redirects` file |
| **Headers** | `staticwebapp.config.json` | `_headers` file |
| **SSL** | Manual setup | Automatic (free) |
| **Build** | Manual `npm run build` | Automated CI/CD |
| **Cost** | ~$0.01/GB/month | Free (500 builds/month) |
| **Speed** | Azure CDN | Cloudflare global CDN |

---

## Rollback to Azure (Emergency)

If Cloudflare deployment fails:

1. **Revert DNS** (if custom domain changed)
2. **Build locally**:
   ```bash
   cd frontend
   npm run build
   ```
3. **Deploy to Azure** (original method):
   ```bash
   az storage blob upload-batch \
     --account-name energymonitorstorage \
     --source ./dist \
     --destination '$web' \
     --overwrite
   ```

Azure site remains accessible at:
`https://energymonitorstorage.z1.web.core.windows.net/`

---

## Success Checklist

- [ ] Cloudflare Pages project created
- [ ] Build succeeds (green checkmark)
- [ ] Site accessible at `*.pages.dev` URL
- [ ] Deep links work (`/app/reports` loads directly)
- [ ] Refresh works on any route (no 404)
- [ ] Security headers present (`curl -I` check)
- [ ] Asset caching works (31536000 max-age)
- [ ] Login/signup authentication flows work
- [ ] Supabase redirect URLs updated
- [ ] Environment variables configured
- [ ] Custom domain configured (optional)

---

## Next Steps

1. **Monitor first deployment** - check build logs
2. **Test all routes** - smoke test critical paths
3. **Update documentation** - point team to new URL
4. **Sunset Azure Storage** - after successful migration (30 day grace)
5. **Set up preview deployments** - for PRs (Cloudflare auto-creates)

---

## Support

- **Cloudflare Pages Docs**: https://developers.cloudflare.com/pages/
- **Redirect Rules**: https://developers.cloudflare.com/pages/configuration/redirects/
- **Headers**: https://developers.cloudflare.com/pages/configuration/headers/
