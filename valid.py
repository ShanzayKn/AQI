import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_openaq_v3_history():
    token = "YOUR_API_KEY"  # Replace with your real API key

    location = "Karachi Central District"
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=30)

    url = (
        "https://api.openaq.org/v3/measurements?"
        f"location={location.replace(' ', '%20')}"
        f"&date_from={start_date}T00:00:00Z"
        f"&date_to={end_date}T23:59:59Z"
        "&parameters=pm25,pm10,no2"
        "&limit=10000&sort=desc"
    )

    headers = {
        'X-API-Key': token
    }

    print("Requesting:", url)
    response = requests.get(url, headers=headers)
    data = response.json()

    if 'results' not in data:
        print("⚠️ API did not return 'results':")
        print(data)
        return

    records = []
    for item in data['results']:
        records.append({
            'timestamp': item['date']['utc'],
            'parameter': item['parameter'],
            'value': item['value'],
            'unit': item['unit'],
            'location': item['location']
        })

    df = pd.DataFrame(records)
    df.to_csv("data/karachi_openaq_history.csv", index=False)
    print("✅ Saved historical data to data/karachi_openaq_history.csv")

if __name__ == "__main__":
    fetch_openaq_v3_history()
