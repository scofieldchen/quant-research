"""
Fetch binance interest rates for a given symbol and time period.
"""

import datetime as dt


import requests
import pandas as pd

url = "https://api.binance.com/sapi/v1/margin/interestRateHistory"

params = {
    "asset": "BTC",
    "startTime": int((dt.datetime.now() - dt.timedelta(days=60)).timestamp() * 1000),
    "endTime": int(dt.datetime.now().timestamp() * 1000),
}

response = requests.get(url, params=params)
response.raise_for_status()

print(response.json())
print(len(response.json()))
