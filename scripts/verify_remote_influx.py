import asyncio
import os
import sys

# Ensure project root is in path
sys.path.append(os.path.join(os.getcwd(), "apps", "backend", "src"))

from app.infra.influx import get_influx_client

async def verify_predictions():
    print(f"--- Verifying InfluxDB Predictions ---")
    print(f"URL: {os.getenv('INFLUX_URL')}")
    print(f"Org: {os.getenv('INFLUX_ORG')}")
    print(f"Bucket: {os.getenv('INFLUX_BUCKET_PRED')}")
    
    client = get_influx_client()
    
    try:
        # 1. Check Connection
        ready = await client.ping()
        print(f"Connection Status: {'✅ Connected' if ready else '❌ Failed'}")
        if not ready:
            return

        # 2. Check Buckets
        buckets = await client.list_buckets()
        bucket_names = [b.name for b in buckets]
        print(f"Available Buckets: {bucket_names}")
        
        target_bucket = os.getenv("INFLUX_BUCKET_PRED", "predictions")
        if target_bucket not in bucket_names:
            print(f"❌ Target bucket '{target_bucket}' NOT FOUND!")
            return

        # 3. Query Predictions (Last 24h)
        print(f"Querying '{target_bucket}' (last 24h)...")
        # Query for *any* measurement in the bucket
        query = f'''
        from(bucket: "{target_bucket}")
            |> range(start: -24h)
            |> count()
        '''
        results = await client.query(query)
        
        total_records = 0
        for table in results:
            for record in table.records:
                total_records += record.get_value() or 0
                
        print(f"Total Prediction Records (Last 24h): {total_records}")
        
        if total_records > 0:
            print("✅ Predictions are being stored!")
        else:
            print("⚠️ No predictions found in the last 24h.")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(verify_predictions())
