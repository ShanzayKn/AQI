import requests
import pandas as pd
from datetime import datetime
import os

def fetch_karachi_aqi():
    token = "89ff788a85614e9577b02565117565062c523895"  # Replace with your AQICN token
    url = f"https://api.waqi.info/feed/karachi/?token={token}"

    response = requests.get(url).json()
    if response['status'] != 'ok':
        print("❌ API Error:", response)
        return

    data = response['data']
    iaqi = data.get('iaqi', {})

    record = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'aqi': data.get('aqi'),
        'pm2_5': iaqi.get('pm25', {}).get('v'),
        'pm10': iaqi.get('pm10', {}).get('v'),
        'temperature': iaqi.get('t', {}).get('v'),
        'humidity': iaqi.get('h', {}).get('v'),
    }

    df = pd.DataFrame([record])

    path = 'data/karachi_aqi_log.csv'
    if os.path.exists(path):
        df.to_csv(path, mode='a', header=False, index=False)
    else:
        df.to_csv(path, index=False)

    print("✅ Logged:", record)

if __name__ == "__main__":
    fetch_karachi_aqi()
