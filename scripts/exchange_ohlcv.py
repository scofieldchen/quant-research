"""
Download ohlcv data from exchanges
"""

import datetime as dt

import ccxt
import pandas as pd


def get_ohlcv(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    start_date: dt.datetime,
    end_date: dt.datetime,
) -> pd.DataFrame:
    """
    Download ohlcv data from exchange using ccxt.

    Args:
        exchange (ccxt.Exchange): The exchange object.
        symbol (str): The trading symbol.
        timeframe (str): The timeframe for the data.
        start_date (datetime.datetime): The start date for the data.
        end_date (datetime.datetime): The end date for the data.

    Returns:
        pd.DataFrame: The downloaded ohlcv data.
    """
    start_timestamp = int(start_date.timestamp() * 1000)
    end_timestamp = int(end_date.timestamp() * 1000)
    all_data = []

    while start_timestamp < end_timestamp:
        data = exchange.fetch_ohlcv(
            symbol, timeframe, since=start_timestamp, limit=1000
        )
        if data:
            all_data.extend(data)
            start_timestamp = data[-1][0] + 1
        else:
            # Skip empty time period if start_date is too early, default skip 1000 bars
            start_timestamp += exchange.parse_timeframe(timeframe) * 1000 * 1000

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(
        all_data, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)

    return df.loc[start_date:end_date]


if __name__ == "__main__":
    exchange = ccxt.bitstamp()
    symbol = "BTC/USD"
    timeframe = "1d"
    start_date = dt.datetime(2012, 1, 1)
    end_date = dt.datetime(2024, 3, 1)
    output_file = "../data/bitstamp_btcusd_1d.csv"

    ohlcv = get_ohlcv(exchange, symbol, timeframe, start_date, end_date)
    ohlcv.to_csv(output_file, index=True)
