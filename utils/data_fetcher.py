import requests
import pandas as pd
from datetime import datetime

def fetch_aqi_data(city='Lahore'):
    API_TOKEN = 'YOUR_API_KEY'  # Register at https://aqicn.org/data-platform/token/
    url = f'https://api.waqi.info/feed/{city}/?token={API_TOKEN}'
    response = requests.get(url)
    data = response.json()

    if data['status'] != 'ok':
        raise ValueError('Failed to fetch data')

    result = data['data']
    df = pd.DataFrame({
        'timestamp': [datetime.now()],
        'aqi': [result.get('aqi')],
        'temperature': [result['iaqi'].get('t', {}).get('v')],
        'humidity': [result['iaqi'].get('h', {}).get('v')],
        'pm2_5': [result['iaqi'].get('pm25', {}).get('v')],
        'pm10': [result['iaqi'].get('pm10', {}).get('v')],
    })

    return df
