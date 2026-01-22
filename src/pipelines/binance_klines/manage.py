"""Binance k线数据管道测试管理器。"""

import datetime as dt
from pathlib import Path

import typer
import duckdb
from rich.console import Console

from src.pipelines.binance_klines.downloader import (
    AssetType,
    Granularity,
    PartitionInterval,
    fetch_api,
    fetch_historical,
)
from src.pipelines.binance_klines.task import backfill, update


app = typer.Typer(help="测试 Binance k线数据管道")
console = Console()

# 测试数据目录
DATA_DIR = Path("data/tests")
DATA_DIR.mkdir(parents=True, exist_ok=True)


@app.command()
def test_downloader():
    """测试下载模块"""
    ticker = "btcusdt"
    date = dt.datetime(2023, 1, 1)

    # 测试历史数据下载
    try:
        df_hist = fetch_historical(
            ticker,
            Granularity.KLINE_1_MINUTE,
            date,
            AssetType.PERP,
            PartitionInterval.DAILY,
        )
        console.print(df_hist.head())
        console.print(df_hist.tail())
    except Exception as e:
        console.print(e)

    # 测试 API 数据下载，本地测试需要使用代理
    try:
        start = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=5)
        proxy = {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}
        df_api = fetch_api(ticker, start, proxy=proxy)
        console.print(df_api.head())
        console.print(df_api.tail())
    except Exception as e:
        console.print(e)


@app.command()
def test_backfill():
    """测试回填历史数据"""
    # 指定交易对和时间范围，下载数据并且存储到正确的目录
    # 多次运行能够覆盖原始parquet文件

    symbol = "btcusdt,ethusdt"
    start_date = "20251101"
    end_date = "20251201"

    backfill(symbols=symbol, start_date=start_date, end_date=end_date, max_workers=3)


@app.command()
def test_update():
    """测试更新增量数据"""
    # 更新指定交易对和时间范围的数据，并且成功覆盖原始文件

    symbol = "btcusdt,ethusdt"
    start_date = None
    end_date = None
    proxy = "http://127.0.0.1:7890"

    update(symbols=symbol, start_date=start_date, end_date=end_date, max_workers=3, proxy=proxy)



if __name__ == "__main__":
    app()
