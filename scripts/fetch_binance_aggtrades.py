"""Fetch aggregated trades data from Binance."""

import datetime as dt
import time
from typing import List

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential


def get_daily_agg_trades(
    symbol: str,
    date: dt.date,
    limit: int = 1000,
    request_delay: float = 0.05,
) -> pd.DataFrame:
    """Fetch all aggregated trades for a symbol on a specific date.

    Args:
        symbol: Trading pair symbol (e.g. 'BTCUSDT')
        date: UTC date to fetch trades for
        limit: Number of trades to fetch per request
        request_delay: Delay between API requests in seconds

    Returns:
        DataFrame containing the aggregated trades with columns:
        - trade_id: Aggregate trade ID
        - timestamp: Trade timestamp (UTC)
        - price: Trade price
        - quantity: Trade quantity
        - is_buyer_maker: Whether the buyer was the maker

    Raises:
        requests.exceptions.RequestException: If API request fails after retries
    """
    # Convert date to UTC timestamp range
    start_ts = int(dt.datetime.combine(date, dt.time.min).timestamp() * 1000)
    end_ts = int(dt.datetime.combine(date, dt.time.max).timestamp() * 1000)

    @retry(
        stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _fetch_trades(start_time: int, end_time: int) -> List[dict]:
        """Helper function to fetch trades with retry logic."""
        params = {
            "symbol": symbol,
            "startTime": start_time,
            "endTime": end_time,
            "limit": limit,
        }

        response = requests.get(
            "https://api.binance.com/api/v3/aggTrades", params=params
        )
        response.raise_for_status()
        return response.json()

    all_trades = []
    current_end = end_ts

    while True:
        trades = _fetch_trades(start_ts, current_end)
        if not trades:
            break

        # Sort by timestamp ascending
        trades.sort(key=lambda x: x["T"])

        # Add to our collection
        all_trades.extend(trades)

        if len(trades) < limit:
            # Less than limit means we got all trades
            break

        # Update start timestamp to get next batch
        # Use the last trade's timestamp + 1ms to avoid duplicates
        start_ts = trades[-1]["T"] + 1

        if start_ts >= end_ts:
            break

        time.sleep(request_delay)

    if not all_trades:
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(all_trades)
    df = df.rename(
        columns={
            "a": "trade_id",
            "T": "timestamp",
            "p": "price",
            "q": "quantity",
            "m": "is_buyer_maker",
        }
    )

    # Convert types
    df["trade_id"] = df["trade_id"].astype("int64")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["price"] = df["price"].astype("float64")
    df["quantity"] = df["quantity"].astype("float64")
    df["is_buyer_maker"] = df["is_buyer_maker"].astype("bool")

    # Sort by timestamp
    df = df.sort_values("timestamp")

    # Verify data is within requested date
    df = df[(df["timestamp"].dt.date == date)]

    return df[["trade_id", "timestamp", "price", "quantity", "is_buyer_maker"]]


def main():
    df = get_daily_agg_trades("BTCUSDT", dt.date(2024, 2, 1))
    print(f"Fetched {len(df)} trades")
    print(df.head())
    print(df.tail())
    print(df.info())


if __name__ == "__main__":
    main()
