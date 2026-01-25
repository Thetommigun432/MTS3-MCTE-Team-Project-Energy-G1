# NILM Production Inference Service
=====================================

## Overview

Real-time Non-Intrusive Load Monitoring (NILM) inference service that:

1. **Receives** raw building power data (1 sample/second)
2. **Preprocesses** into model-ready features
3. **Buffers** in Redis sliding window (1536 samples = ~25 minutes)
4. **Runs inference** on all 11 appliance models in parallel
5. **Publishes** predictions back to Redis

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌──────────────────┐
│   Sensors   │────▶│    Redis    │◀───▶│ Inference Service│
│  (1 Hz)     │     │   Buffer    │     │  (11 models)     │
└─────────────┘     └─────────────┘     └──────────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ Predictions │
                    │  (Pub/Sub)  │
                    └─────────────┘
```

## Quick Start

### With Docker Compose

```bash
cd production

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f inference

# Run with test data producer
docker-compose --profile testing up -d
```

### Local Development

```bash
# Install dependencies
pip install -r requirements-production.txt

# Start Redis (required)
docker run -d -p 6379:6379 redis:7-alpine

# Run inference service
python -m production.inference_service \
    --redis-host localhost \
    --redis-port 6379 \
    --model-dir ./checkpoints \
    --mode subscriber

# In another terminal, run data producer
python -m production.data_producer --source simulation
```

### Test Mode (No Redis)

```bash
python -m production.inference_service --mode test
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_HOST` | localhost | Redis server hostname |
| `REDIS_PORT` | 6379 | Redis server port |
| `MODEL_DIR` | ./checkpoints | Directory with trained models |
| `BUILDING_ID` | building_1 | Unique building identifier |
| `WINDOW_SIZE` | 1536 | Inference window (samples) |
| `INFERENCE_INTERVAL` | 60 | Seconds between inferences |

### Command Line Arguments

```bash
python -m production.inference_service \
    --redis-host localhost \
    --redis-port 6379 \
    --model-dir ./checkpoints \
    --building-id building_1 \
    --window-size 1536 \
    --inference-interval 60 \
    --mode subscriber
```

## Redis Data Format

### Input (Raw Data)

Channel: `nilm:{building_id}:input`

```json
{
    "timestamp": 1706198400.0,
    "power_total": 3500.0,
    "voltage": 230.0,
    "current": 15.2,
    "power_factor": 0.95
}
```

### Output (Predictions)

Channel: `nilm:{building_id}:predictions`

```json
{
    "timestamp": 1706198460.0,
    "window_start": 1706196924.0,
    "window_end": 1706198460.0,
    "total_power": 3450.5,
    "inference_time_ms": 45.2,
    "predictions": [
        {
            "appliance": "HeatPump",
            "power_watts": 2100.5,
            "probability": 0.92,
            "is_on": true,
            "confidence": 0.84
        },
        {
            "appliance": "Refrigerator",
            "power_watts": 120.0,
            "probability": 0.98,
            "is_on": true,
            "confidence": 0.96
        }
        // ... 11 appliances total
    ]
}
```

## API Integration

### Python Client Example

```python
import redis
import json

r = redis.Redis(host='localhost', port=6379)
pubsub = r.pubsub()
pubsub.subscribe('nilm:building_1:predictions')

for message in pubsub.listen():
    if message['type'] == 'message':
        predictions = json.loads(message['data'])
        for p in predictions['predictions']:
            if p['is_on']:
                print(f"{p['appliance']}: {p['power_watts']:.0f}W")
```

### Publishing Sensor Data

```python
import redis
import json
import time

r = redis.Redis(host='localhost', port=6379)

while True:
    data = {
        'timestamp': time.time(),
        'power_total': read_power_sensor(),  # Your sensor reading
        'voltage': 230.0
    }
    r.publish('nilm:building_1:input', json.dumps(data))
    time.sleep(1)  # 1 Hz
```

## Models

The service loads trained WaveNILM v3 models for these appliances:

| Appliance | Status | Notes |
|-----------|--------|-------|
| HeatPump | ✅ | Trained |
| Dishwasher | ✅ | Trained |
| WashingMachine | 🔄 | Training |
| Dryer | ⏳ | Pending |
| Freezer | ⏳ | Pending |
| Refrigerator | ⏳ | Pending |
| ElectricOven | ⏳ | Pending |
| Microwave | ⏳ | Pending |
| ElectricKettle | ⏳ | Pending |
| Computer | ⏳ | Pending |
| Television | ⏳ | Pending |

## Monitoring

### Buffer Status

```bash
redis-cli HGETALL nilm:building_1:status
```

### Latest Predictions

```bash
redis-cli GET nilm:building_1:latest | jq
```

### Subscribe to Predictions

```bash
redis-cli SUBSCRIBE nilm:building_1:predictions
```

## Performance

- **Inference latency**: ~50ms (GPU), ~200ms (CPU)
- **Memory usage**: ~2GB (all models loaded)
- **GPU VRAM**: ~1.5GB
- **Throughput**: Handles 1 Hz input easily

## Troubleshooting

### Models not loading

```bash
# Check checkpoint files exist
ls -la checkpoints/*.pth

# Ensure correct naming convention
# wavenilm_v3_SOTA_{Appliance}_best.pth
```

### Redis connection issues

```bash
# Test Redis connectivity
redis-cli -h $REDIS_HOST -p $REDIS_PORT ping

# Check Redis logs
docker-compose logs redis
```

### GPU not detected

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# For Docker, ensure nvidia-docker is installed
docker run --gpus all nvidia/cuda:11.8-base nvidia-smi
```
