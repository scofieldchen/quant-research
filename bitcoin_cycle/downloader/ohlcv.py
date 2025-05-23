import datetime as dt
from pathlib import Path

import yfinance as yf
from rich.console import Console

# 若无法从雅虎财经下载数据，设置 vpn 作为网络代理
yf.set_config(proxy={"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"})

console = Console()


def download_ohlcv(
    filepath: Path, symbol: str, start_date: dt.datetime, end_date: dt.datetime
) -> None:
    """从雅虎财经下载历史价格"""
    data = yf.download(
        tickers=symbol,
        start=start_date,
        end=end_date,
        ignore_tz=True,
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )

    data = data[["Open", "High", "Low", "Close", "Volume"]]
    data.index.name = "datetime"
    data.columns = [x.lower() for x in data.columns]

    data.to_csv(filepath, index=True)

    console.print(f"✅ Downloaded ohlcv for {symbol}, last:{data.index.max():%Y-%m-%d}")


if __name__ == "__main__":
    download_ohlcv(
        filepath=Path("/users/scofield/quant-research/bitcoin_cycle/data/btcusd.csv"),
        symbol="BTC-USD",
        start_date=dt.datetime(2014, 1, 1, tzinfo=dt.timezone.utc),
        end_date=dt.datetime(2025, 5, 30, tzinfo=dt.timezone.utc),
    )
