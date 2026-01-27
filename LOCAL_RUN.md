# Local Development Setup

This file is kept for backwards compatibility.

Use the canonical guide instead:
[docs/LOCAL_DEV.md](docs/LOCAL_DEV.md)
- Local proxy: `VITE_BACKEND_URL=/api`
- Direct: `VITE_BACKEND_URL=http://localhost:8000`

## Success Signals

When everything is working correctly, you should see:

1. **Smoke test passes:**
   ```bash
   ./scripts/local_smoke.sh
   # All checks show GREEN
   ```

2. **Redis window at capacity:**
   ```bash
   docker exec nilm-redis redis-cli LLEN nilm:building-1:window
   # Returns: 3600
   ```

3. **Worker producing predictions:**
   ```bash
   docker compose logs worker | tail -20
   # Shows: "Prediction written: building=building-1, model=..."
   ```

4. **Predictions endpoint returns data:**
   ```bash
   curl "http://localhost:8000/api/analytics/predictions?building_id=building-1&start=-5m&end=now()"
   # Returns JSON with count > 0
   ```
