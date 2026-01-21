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
from src.pipelines.binance_klines.storage import (
    get_last_timestamp,
    save_cleaned_data,
)


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
def test_storage():
    """测试存储模块"""
    import pandas as pd

    # 创建测试目录
    test_cleaned_dir = DATA_DIR / "cleaned"
    test_cleaned_dir.mkdir(parents=True, exist_ok=True)

    # 创建测试数据（固定时间戳，便于验证）
    sample_data = pd.DataFrame(
        {
            "datetime": pd.date_range(
                "2023-01-01 10:00:00", periods=10, freq="1min", tz="UTC"
            ),
            "ticker": ["btcusdt"] * 10,
            "open": [100] * 10,
            "high": [101] * 10,
            "low": [99] * 10,
            "close": [100.5] * 10,
            "volume": [1000] * 10,
        }
    )

    # 先保存清理数据
    try:
        save_cleaned_data(sample_data, test_cleaned_dir)
        console.print("清理数据保存成功")
    except Exception as e:
        console.print(f"清理数据保存失败: {e}")
        raise

    # 然后查询最后时间戳
    try:
        last_ts = get_last_timestamp("btcusdt", test_cleaned_dir)
        console.print(f"最后时间戳: {last_ts}")
    except Exception as e:
        console.print(f"最后时间戳查询失败: {e}")
        raise

    # res = pd.read_parquet(test_cleaned_dir)
    # console.print(res)

    query = f"""
    SELECT *
    FROM read_parquet('{test_cleaned_dir}/**/*.parquet', hive_partitioning=true)
    """
    df = duckdb.sql(query).df()
    console.print(df)


@app.command()
def run_all():
    """运行所有测试"""
    console.print("开始运行所有测试")
    try:
        test_downloader()
        test_storage()
        console.print("所有测试通过")
    except Exception as e:
        console.print(f"测试失败: {e}")
        raise


if __name__ == "__main__":
    app()
