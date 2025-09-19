import random
import time
import json
import paho.mqtt.client as mqtt

# MQTT Setup
broker = "aqwpremxt8sbq-ats.iot.us-east-2.amazonaws.com"
port = 8883
topic = "env/sensor_data"
client_id = "env_sensor_1"

# AWS IoT certificates
ca_path = "/Users/linnettuscano/Downloads/AmazonRootCA1.pem"
cert_path = "/Users/linnettuscano/Downloads/6ba22e7b5a34da3c9be29b9b1b53cb363ad41c35a2d495fcdb028ee31dcee24c-certificate.pem.crt"
key_path = "/Users/linnettuscano/Downloads/6ba22e7b5a34da3c9be29b9b1b53cb363ad41c35a2d495fcdb028ee31dcee24c-private.pem.key"

print("Initializing MQTT Client...")

# Define MQTT client
client = mqtt.Client(client_id)

# Enable TLS for security
client.tls_set(ca_path, certfile=cert_path, keyfile=key_path)

print("Connecting to MQTT Broker...")
client.connect(broker, port)
print("Connected successfully!")

def generate_data():
    temp = random.uniform(-50, 50)  # Temperature between -50 and 50 Celsius
    humidity = random.uniform(0, 100)  # Humidity between 0 and 100%
    co2 = random.randint(300, 2000)  # CO2 between 300 and 2000 ppm
    return {"temperature": temp, "humidity": humidity, "co2": co2}

while True:
    sensor_data = generate_data()
    payload = json.dumps(sensor_data)  # Convert the data to JSON format
    
    print(f"Publishing Data: {payload}")  # Print data before sending
    
    # Publish the data to the MQTT topic
    client.publish(topic, payload)
    
    time.sleep(30)  # Wait for 5 seconds before publishing next data
