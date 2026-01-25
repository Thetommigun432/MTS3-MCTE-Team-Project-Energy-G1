
import os
import json
import logging
import redis
from datetime import datetime
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    # Configuration
    redis_host = os.environ.get('REDIS_HOST', 'localhost')
    redis_port = int(os.environ.get('REDIS_PORT', 6379))
    building_id = os.environ.get('BUILDING_ID', 'building_1')
    
    influx_url = os.environ.get('INFLUX_URL', 'http://localhost:8086')
    influx_token = os.environ.get('INFLUX_TOKEN', 'admin-token')
    influx_org = os.environ.get('INFLUX_ORG', 'energy-monitor')
    influx_bucket = os.environ.get('INFLUX_BUCKET_PRED', 'predictions')
    
    channel = f"nilm:{building_id}:predictions"
    
    # 1. Connect to Redis
    r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
    pubsub = r.pubsub()
    pubsub.subscribe(channel)
    
    # 2. Connect to InfluxDB
    influx_client = InfluxDBClient(url=influx_url, token=influx_token, org=influx_org)
    write_api = influx_client.write_api(write_options=SYNCHRONOUS)
    
    logger.info(f"Starting Prediction Persister...")
    logger.info(f"Redis Channel: {channel}")
    logger.info(f"Influx Bucket: {influx_bucket}")
    
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
            #   "predictions": [ { "appliance": str, "power_watts": float, "is_on": bool, ... } ]
            # }
            
            ts_float = payload['timestamp']
            ts = datetime.fromtimestamp(ts_float)
            
            points = []
            
            for pred in payload['predictions']:
                p = Point("nilm_predictions") \
                    .tag("building_id", building_id) \
                    .tag("appliance", pred['appliance']) \
                    .tag("model", "dummy-v1") \
                    .field("power_watts", float(pred['power_watts'])) \
                    .field("probability", float(pred['probability'])) \
                    .field("confidence", float(pred['confidence'])) \
                    .time(ts)
                points.append(p)
                
            if points:
                write_api.write(bucket=influx_bucket, org=influx_org, record=points)
                logger.info(f"Persisted {len(points)} predictions for timestamp {ts}")
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")

if __name__ == "__main__":
    main()
