import requests
import pandas as pd


url = "https://charts.bgeometrics.com/files/sth_realized_price.json"
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
}

resp = requests.get(url, headers=headers, timeout=3)
resp.raise_for_status()
print(resp.json())

df = pd.DataFrame.from_records(resp.json())
df.columns = ["datetime", "sth_realized_price"]
df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
df.sort_values("datetime", ascending=True, inplace=True)
df.set_index("datetime", inplace=True)

print(df.head())
print(df.tail())
print(df.info())
