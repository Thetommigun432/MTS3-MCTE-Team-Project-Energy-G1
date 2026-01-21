@echo off
REM Cloudflare Pages Deployment Script for Windows
REM Usage: deploy-cloudflare.bat

echo.
echo 🚀 NILM Energy Monitor - Cloudflare Pages Deployment
echo ======================================================
echo.

REM Check if we're in the right directory
if not exist "apps\web" (
  echo ❌ Error: Must run from repository root
  exit /b 1
)

REM Navigate to frontend
cd apps\web

echo 📦 Step 1: Installing dependencies...
call npm install

echo.
echo 🔨 Step 2: Building production bundle...
call npm run build

echo.
echo ✅ Build complete!
echo.

echo 🌐 Step 3: Deploying to Cloudflare Pages...
echo.

REM Deploy
call npx wrangler pages deploy dist --project-name=nilm-energy-monitor

echo.
echo ✅ Deployment complete!
echo.
echo 🔗 Your site is live at: https://nilm-energy-monitor.pages.dev
echo.
echo 📝 Next steps:
echo 1. Set environment variables in Cloudflare dashboard
echo 2. Update backend CORS_ORIGINS to include your Pages URL
echo 3. Test the deployment
echo.
echo 📚 See docs/CLOUDFLARE_DEPLOYMENT.md for detailed instructions
echo.

cd ..\..
