"""
从官网下载贪婪和恐慌指数的数据
"""

import requests
import pandas as pd


def fetch_fear_greed_index(limit: int = 10) -> pd.DataFrame:
    url = "https://api.alternative.me/fng/"
    resp = requests.get(url, params={"limit": limit})
    resp.raise_for_status()
    data = resp.json()

    return (
        pd.DataFrame.from_records(data["data"], exclude=["time_until_update"])
        .astype({"value": "float64", "timestamp": "int"})
        .assign(date=lambda x: pd.to_datetime(x.timestamp, unit="s"))
        .sort_values("date", ascending=True)
        .set_index("date")
        .drop(columns=["timestamp"])
        .rename(columns={"value_classification": "classification"})
    )


def main():
    years_to_fetch = 8
    output_file = "../data/fear_greed_index.csv"

    print("Fetching fear and greed index data...")
    res = fetch_fear_greed_index(years_to_fetch * 365)

    print(f"Writing data to {output_file}")
    res.to_csv(output_file, index=True)


if __name__ == "__main__":
    main()
