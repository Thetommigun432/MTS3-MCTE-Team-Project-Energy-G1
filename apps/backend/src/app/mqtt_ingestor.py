"""
NILM MQTT Ingestor - Real-time Smappee Data
============================================

Connects to the Howest Energy Lab MQTT broker to receive real-time
power data from the Smappee monitoring system.

Two modes:
1. Aggregate only (main): Sends totalPower to backend for NILM predictions
2. With ground truth: Also logs individual appliance readings for validation

Usage:
    python -m app.mqtt_ingestor

Environment Variables:
    BACKEND_URL: Backend base URL (default: http://backend:8000)
    BUILDING_ID: Building identifier (default: building-1)
    MQTT_HOST: MQTT broker host (default: mqtt.howest-energylab.be)
    MQTT_PORT: MQTT broker port (default: 10591)
    MQTT_USER: MQTT username
    MQTT_PASS: MQTT password
    LOG_GROUND_TRUTH: Whether to log appliance readings (default: false)
"""

import asyncio
import json
import os
import signal
import ssl
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import httpx
import paho.mqtt.client as mqtt

# Logging setup
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Appliance mapping from publishIndex to name
APPLIANCE_MAP = {
    3: "WashingMachine",
    4: "Dishwasher", 
    5: "Dryer",  # TumbleDryer
    6: "Oven",
    7: "RangeHood",  # Dampkap/Hood
    13: "RainwaterPump",
    14: "Stove_L1",  # CookingPlate_L1
    15: "Stove_L2",  # CookingPlate_L2
    16: "HeatPump",
    17: "EVCharger_L1",  # ChargingStation_L1
    18: "EVCharger_L2",  # ChargingStation_L2
    19: "EVSocket",  # Smal_ChargingStation
}


class MQTTIngestor:
    """Real-time MQTT data ingestor for Smappee."""

    def __init__(self):
        # Configuration from environment
        self.backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
        self.building_id = os.getenv("BUILDING_ID", "building-1")
        self.mqtt_host = os.getenv("MQTT_HOST", "mqtt.howest-energylab.be")
        self.mqtt_port = int(os.getenv("MQTT_PORT", "10591"))
        self.mqtt_user = os.getenv("MQTT_USER", "student_CTAI")
        self.mqtt_pass = os.getenv("MQTT_PASS", "vSsbTGB5nKAwluWNpIt8wQDpZ3ofpknm")
        self.log_ground_truth = os.getenv("LOG_GROUND_TRUTH", "false").lower() == "true"
        
        # MQTT topic
        self.topic = "servicelocation/33a8340b-f03c-4851-9f9f-99b98e2c4cc9/realtime/#"
        
        # State
        self.running = False
        self.readings_sent = 0
        self.errors = 0
        self.last_reading_time = None
        self.http_client: Optional[httpx.Client] = None
        
        # Latest readings cache (for ground truth logging)
        self.latest_appliances = {}
        
    def on_connect(self, client, userdata, flags, reason_code, properties):
        """MQTT connection callback."""
        if reason_code.is_failure:
            logger.error(f"Failed to connect to MQTT: {reason_code}")
        else:
            logger.info(f"Connected to MQTT broker at {self.mqtt_host}:{self.mqtt_port}")
            client.subscribe(self.topic)
            logger.info(f"Subscribed to: {self.topic}")

    def on_message(self, client, userdata, message):
        """MQTT message callback - process incoming readings."""
        try:
            msg = json.loads(message.payload.decode('utf-8'))
            
            # Extract total power (aggregate)
            if "totalPower" in msg:
                total_power_w = msg["totalPower"]
                self.send_to_backend(total_power_w)
                
            # Extract individual appliance powers (ground truth)
            if "channelPowers" in msg and self.log_ground_truth:
                for channel in msg["channelPowers"]:
                    idx = channel.get("publishIndex")
                    power = channel.get("power", 0)
                    if idx in APPLIANCE_MAP:
                        appliance = APPLIANCE_MAP[idx]
                        self.latest_appliances[appliance] = power
                
                # Log ground truth periodically
                if self.readings_sent % 10 == 0:
                    self.log_appliance_readings()
                    
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse MQTT message: {e}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def send_to_backend(self, power_watts: float):
        """Send reading to backend for NILM prediction."""
        try:
            timestamp = datetime.now(timezone.utc)
            
            payload = {
                "readings": [
                    {
                        "building_id": self.building_id,
                        "aggregate_kw": power_watts / 1000.0,  # Convert W to kW
                        "ts": timestamp.isoformat(),
                    }
                ]
            }
            
            resp = self.http_client.post(
                f"{self.backend_url}/api/ingest/readings",
                json=payload,
                timeout=10.0,
            )
            
            if resp.status_code in (200, 202):
                self.readings_sent += 1
                self.last_reading_time = timestamp
                
                if self.readings_sent % 10 == 0:
                    logger.info(
                        f"Sent {self.readings_sent} readings | "
                        f"Latest: {power_watts:.0f}W ({power_watts/1000:.2f}kW)"
                    )
            else:
                self.errors += 1
                logger.warning(f"Backend returned {resp.status_code}: {resp.text[:100]}")
                
        except Exception as e:
            self.errors += 1
            logger.error(f"Failed to send to backend: {e}")

    def log_appliance_readings(self):
        """Log current appliance readings (ground truth)."""
        if not self.latest_appliances:
            return
            
        # Combine Stove phases
        stove_total = self.latest_appliances.get("Stove_L1", 0) + self.latest_appliances.get("Stove_L2", 0)
        # Combine EV Charger phases
        ev_total = (self.latest_appliances.get("EVCharger_L1", 0) + 
                   self.latest_appliances.get("EVCharger_L2", 0))
        
        logger.info("=" * 50)
        logger.info("GROUND TRUTH (Real Appliance Readings):")
        logger.info(f"  HeatPump:       {self.latest_appliances.get('HeatPump', 0):>6.0f} W")
        logger.info(f"  Dryer:          {self.latest_appliances.get('Dryer', 0):>6.0f} W")
        logger.info(f"  Dishwasher:     {self.latest_appliances.get('Dishwasher', 0):>6.0f} W")
        logger.info(f"  WashingMachine: {self.latest_appliances.get('WashingMachine', 0):>6.0f} W")
        logger.info(f"  Oven:           {self.latest_appliances.get('Oven', 0):>6.0f} W")
        logger.info(f"  Stove (L1+L2):  {stove_total:>6.0f} W")
        logger.info(f"  RangeHood:      {self.latest_appliances.get('RangeHood', 0):>6.0f} W")
        logger.info(f"  RainwaterPump:  {self.latest_appliances.get('RainwaterPump', 0):>6.0f} W")
        logger.info(f"  EVCharger:      {ev_total:>6.0f} W")
        logger.info(f"  EVSocket:       {self.latest_appliances.get('EVSocket', 0):>6.0f} W")
        logger.info("=" * 50)

    def wait_for_backend(self, timeout: int = 300) -> bool:
        """Wait for backend to be ready."""
        logger.info(f"Waiting for backend at {self.backend_url}/ready...")
        
        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = self.http_client.get(f"{self.backend_url}/ready", timeout=5.0)
                if resp.status_code == 200:
                    logger.info("Backend is ready!")
                    return True
            except Exception as e:
                logger.debug(f"Backend not ready yet: {e}")
            time.sleep(2)
            
        logger.error(f"Backend did not become ready within {timeout}s")
        return False

    def run(self):
        """Main run loop."""
        logger.info("=" * 60)
        logger.info("NILM MQTT Ingestor Starting")
        logger.info("=" * 60)
        logger.info(f"Backend URL: {self.backend_url}")
        logger.info(f"Building ID: {self.building_id}")
        logger.info(f"MQTT Broker: {self.mqtt_host}:{self.mqtt_port}")
        logger.info(f"Log Ground Truth: {self.log_ground_truth}")
        logger.info("=" * 60)
        
        # Initialize HTTP client
        self.http_client = httpx.Client()
        
        # Wait for backend
        if not self.wait_for_backend():
            logger.error("Backend not available, exiting")
            sys.exit(1)
        
        # Setup MQTT client
        mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        mqttc.on_connect = self.on_connect
        mqttc.on_message = self.on_message
        
        # Authentication
        mqttc.username_pw_set(self.mqtt_user, self.mqtt_pass)
        
        # TLS setup
        mqttc.tls_set(certfile=None, keyfile=None, cert_reqs=ssl.CERT_NONE)
        
        # Connect
        logger.info(f"Connecting to MQTT broker...")
        mqttc.connect(self.mqtt_host, self.mqtt_port, 60)
        
        # Signal handling for graceful shutdown
        def signal_handler(sig, frame):
            logger.info("Shutdown signal received")
            mqttc.loop_stop()
            mqttc.disconnect()
            if self.http_client:
                self.http_client.close()
            logger.info(f"Final stats: {self.readings_sent} sent, {self.errors} errors")
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Run forever
        self.running = True
        logger.info("Starting MQTT loop - receiving real-time data...")
        mqttc.loop_forever()


def main():
    """Entry point."""
    ingestor = MQTTIngestor()
    ingestor.run()


if __name__ == "__main__":
    main()
