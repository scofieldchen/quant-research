"""
Fetch Binance funding rates for a given symbol and time period.
"""

import datetime as dt

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_fixed


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def _fetch_funding_rates(
    symbol: str, start_timestamp: int, end_timestamp: int, limit: int = 1000
) -> list[dict]:
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    response = requests.get(
        url=url,
        params={
            "symbol": symbol,
            "startTime": start_timestamp,
            "endTime": end_timestamp,
            "limit": limit,
        },
    )
    response.raise_for_status()
    return response.json()


def fetch_funding_rates(
    symbol: str, start_date: dt.datetime, end_date: dt.datetime
) -> pd.DataFrame:
    start_timestamp = int(start_date.timestamp() * 1000)
    end_timestamp = int(end_date.timestamp() * 1000)
    all_data = []

    while start_timestamp < end_timestamp:
        data = _fetch_funding_rates(symbol, start_timestamp, end_timestamp)
        if not data:
            break
        all_data.extend(data)
        start_timestamp = data[-1]["fundingTime"] + 1

    if not all_data:
        return pd.DataFrame()

    return process_funding_rates(all_data).loc[start_date:end_date]


def process_funding_rates(data: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(data)
    return (
        df.assign(
            fundingTime=pd.to_datetime(df["fundingTime"], unit="ms"),
            fundingRate=pd.to_numeric(df["fundingRate"]),
            markPrice=pd.to_numeric(df["markPrice"]),
        )
        .rename(
            columns={
                "fundingTime": "funding_time",
                "fundingRate": "funding_rate",
                "markPrice": "mark_price",
            }
        )
        .set_index("funding_time")
    )


if __name__ == "__main__":
    symbol = "BTCUSDT"
    start_date = dt.datetime(2020, 1, 1)
    end_date = dt.datetime(2024, 12, 1)
    df = fetch_funding_rates(symbol, start_date, end_date)
    print(df)
    print(df.info())
