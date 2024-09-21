"""
Download ohlcv data from exchanges
"""

import datetime as dt
import os

import ccxt
import click
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


def parse_date(date_str: str, default: dt.datetime) -> dt.datetime:
    if date_str is None:
        return default
    return dt.datetime.strptime(date_str, "%Y-%m-%d")


@click.command()
@click.option("--exchange_name", type=str, required=True, help="Name of the exchange")
@click.option(
    "--symbol", type=str, required=True, help="Trading pair symbol, format: BTC/USD"
)
@click.option(
    "--timeframe",
    type=click.Choice(["5m", "30m", "1h", "4h", "1d"]),
    default="1d",
    show_default=True,
    help="Timeframe for OHLCV data",
)
@click.option(
    "--start_date",
    type=str,
    default=None,
    show_default=True,
    help="Start date for data in YYYY-MM-DD format",
)
@click.option(
    "--end_date",
    type=str,
    default=None,
    show_default=True,
    help="End date for data in YYYY-MM-DD format",
)
@click.option(
    "--output_dir",
    type=str,
    default="../data/",
    help="Directory to save the output CSV file",
)
def main(
    exchange_name: str,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    output_dir: str,
) -> None:
    exchange_name = exchange_name.lower()
    symbol = symbol.upper()

    assert (
        exchange_name in ccxt.exchanges
    ), f"Exchange '{exchange_name}' is not supported by ccxt"
    assert "/" in symbol, "Symbol must be in the format 'BTC/USD'"

    end_date_dt = parse_date(end_date, dt.datetime.today())
    start_date_dt = parse_date(start_date, end_date_dt - dt.timedelta(days=365 * 10))

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir, f"{exchange_name}_{symbol.replace('/', '').lower()}_{timeframe}.csv"
    )

    print(f"Fetching data from {exchange_name} for {symbol} on {timeframe} timeframe")
    exchange = getattr(ccxt, exchange_name)()
    ohlcv = get_ohlcv(exchange, symbol.upper(), timeframe, start_date_dt, end_date_dt)

    print(f"Saving data to {output_file}")
    ohlcv.to_csv(output_file, index=True)


if __name__ == "__main__":
    main()
