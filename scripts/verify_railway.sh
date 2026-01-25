
#!/bin/bash
set -e

echo "=== Verifying Railway Build Configuration ==="

# 1. Simulate Root Build Context (Standard Railway Behavior)
echo "1. Building Backend from Root Context..."
# Using the path specified in railway.json: apps/backend/Dockerfile
# But running from ROOT (current dir)
if docker build -f apps/backend/Dockerfile . > /dev/null 2>&1; then
    echo "✅ Backend build succeeded from Root Context"
else
    echo "❌ Backend build FAILED from Root Context"
    echo "   Railway requires 'COPY' paths to be relative to the build context."
    echo "   Current Dockerfile likely expects 'apps/backend' context."
fi

# 2. Simulate Subdirectory Build Context (Monorepo Setup)
echo "2. Building Backend from Subdirectory Context..."
if docker build -f apps/backend/Dockerfile apps/backend > /dev/null 2>&1; then
    echo "✅ Backend build succeeded from 'apps/backend' Context"
else
    echo "❌ Backend build FAILED from 'apps/backend' Context"
fi

echo "=== Recommendation ==="
echo "If (1) failed and (2) passed, you MUST configure Railway 'Root Directory' to '/apps/backend'."
echo "If both passed, standard configuration works."
