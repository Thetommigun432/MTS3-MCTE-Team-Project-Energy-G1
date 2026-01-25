"""
Prediction Persister
====================

Consumes prediction messages from Redis pub/sub and writes them to InfluxDB
using the CANONICAL WIDE format.

Schema:
    Bucket: predictions
    Measurement: prediction
    Tags: building_id, stream_key (optional), model_version
    Fields: predicted_kw_{appliance}, confidence_{appliance}
"""

import os
import json
import logging
import redis
from datetime import datetime, timezone
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def normalize_appliance_key(appliance: str) -> str:
    """
    Normalize appliance name to a safe field key.
    
    Examples:
        "Heat Pump" -> "HeatPump"
        "washing_machine" -> "WashingMachine"
        "EV Charger" -> "EVCharger"
    """
    # Remove spaces and convert to PascalCase-ish
    words = appliance.replace('_', ' ').split()
    return ''.join(word.capitalize() for word in words)


def main():
    # Configuration
    redis_host = os.environ.get('REDIS_HOST', 'localhost')
    redis_port = int(os.environ.get('REDIS_PORT', 6379))
    redis_url = os.environ.get('REDIS_URL')
    
    # Stream key is the legacy identifier (e.g., building_1)
    stream_key = os.environ.get('BUILDING_ID', 'building_1')
    # Building UUID from Supabase (if available)
    building_uuid = os.environ.get('BUILDING_UUID')
    
    influx_url = os.environ.get('INFLUX_URL', 'http://localhost:8086')
    influx_token = os.environ.get('INFLUX_TOKEN', 'influx-admin-token-2026-secure')
    influx_org = os.environ.get('INFLUX_ORG', 'energy-monitor')
    influx_bucket = os.environ.get('INFLUX_BUCKET_PRED', 'predictions')
    influx_measurement = os.environ.get('INFLUX_MEASUREMENT_PRED', 'prediction')
    
    # Optional Run ID for E2E isolation
    e2e_run_id = os.environ.get('E2E_RUN_ID')
    
    channel = f"nilm:{stream_key}:predictions"
    
    # Determine building_id to use for Influx tags
    # Prefer UUID if available, fallback to stream_key
    building_id = building_uuid if building_uuid else stream_key
    
    # 1. Connect to Redis
    if redis_url:
        r = redis.from_url(redis_url, decode_responses=True)
    else:
        r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
    
    pubsub = r.pubsub()
    pubsub.subscribe(channel)
    
    # 2. Connect to InfluxDB
    influx_client = InfluxDBClient(url=influx_url, token=influx_token, org=influx_org)
    write_api = influx_client.write_api(write_options=SYNCHRONOUS)
    
    logger.info("Starting Prediction Persister (CANONICAL WIDE FORMAT)...")
    logger.info(f"Redis Channel: {channel}")
    logger.info(f"Influx Bucket: {influx_bucket}")
    logger.info(f"Influx Measurement: {influx_measurement}")
    logger.info(f"Building ID (for tags): {building_id}")
    if e2e_run_id:
        logger.info(f"E2E Run ID: {e2e_run_id}")
    if stream_key != building_id:
        logger.info(f"Stream Key: {stream_key}")
    
    # 3. Listen and Write
    logger.info("Listening for predictions...")
    
    for message in pubsub.listen():
        if message['type'] != 'message':
            continue
            
        try:
            payload = json.loads(message['data'])
            
            # Message format from inference_service:
            # {
            #   "timestamp": float,
            #   "predictions": [ { "appliance": str, "power_watts": float, "probability": float, ... } ]
            # }
            
            ts_float = payload['timestamp']
            ts = datetime.fromtimestamp(ts_float, tz=timezone.utc)
            
            # Get model version (all predictions should have same version)
            model_version = "unknown"
            if payload.get('predictions'):
                model_version = payload['predictions'][0].get('model_version', 'unknown')
            
            # Build WIDE format point - one point with all appliances as fields
            point = (
                Point(influx_measurement)
                .tag("building_id", building_id)
                .tag("model_version", model_version)
                .time(ts)
            )

            # Add E2E Run ID tag if present
            if e2e_run_id:
                point = point.tag("run_id", e2e_run_id)
            
            # Add stream_key tag if different from building_id
            if stream_key and stream_key != building_id:
                point = point.tag("stream_key", stream_key)
            
            # Add fields for each appliance prediction
            for pred in payload['predictions']:
                appliance = pred.get('appliance', 'unknown')
                appliance_key = normalize_appliance_key(appliance)
                
                power_kw = float(pred.get('power_watts', 0)) / 1000.0  # Convert W to kW
                confidence = float(pred.get('confidence', pred.get('probability', 0)))
                
                point = point.field(f"predicted_kw_{appliance_key}", max(0.0, power_kw))
                point = point.field(f"confidence_{appliance_key}", min(1.0, max(0.0, confidence)))
                
                # Add is_on field if available
                if 'is_on' in pred:
                    point = point.field(f"is_on_{appliance_key}", bool(pred['is_on']))
            
            # Write single point with all appliances
            write_api.write(bucket=influx_bucket, org=influx_org, record=point)
            
            appliance_count = len(payload.get('predictions', []))
            logger.info(f"Persisted {appliance_count} appliances for timestamp {ts.isoformat()}")
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")


if __name__ == "__main__":
    main()
