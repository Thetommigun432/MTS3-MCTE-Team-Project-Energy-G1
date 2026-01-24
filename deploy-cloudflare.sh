#!/bin/bash
# Cloudflare Pages Deployment Script
# Usage: ./deploy-cloudflare.sh

set -e

echo "🚀 NILM Energy Monitor - Cloudflare Pages Deployment"
echo "======================================================"
echo ""

# Check if we're in the right directory
if [ ! -d "apps/web" ]; then
  echo "❌ Error: Must run from repository root"
  exit 1
fi

# Navigate to frontend
cd apps/web

echo "📦 Step 1: Installing dependencies..."
npm install

echo ""
echo "🔨 Step 2: Building production bundle..."
npm run build

echo ""
echo "✅ Build complete!"
echo ""
echo "📊 Build size:"
du -sh dist/

echo ""
echo "🌐 Step 3: Deploying to Cloudflare Pages..."
echo ""

# Check if wrangler is installed
if ! command -v wrangler &> /dev/null; then
  echo "⚠️  Wrangler CLI not found. Installing..."
  npm install -g wrangler
fi

# Deploy
npx wrangler pages deploy dist --project-name=nilm-energy-monitor

echo ""
echo "✅ Deployment complete!"
echo ""
echo "🔗 Your site is live at: https://nilm-energy-monitor.pages.dev"
echo ""
echo "📝 Next steps:"
echo "1. Set environment variables in Cloudflare dashboard"
echo "2. Update backend CORS_ORIGINS to include your Pages URL"
echo "3. Test the deployment"
echo ""
echo "📚 See docs/CLOUDFLARE_DEPLOYMENT.md for detailed instructions"
