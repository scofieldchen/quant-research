import datetime as dt
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.core.bgeometrics import BGClient
from src.core.logger import get_logger

# 数据目录
RAW_DATA_DIR = Path("data/raw/sth_mvrv")

# 需要设置代理才能请求数据
yf.config.network.proxy = {
    "http": "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890",
}

logger = get_logger("sthmvrv")


def download_sth_mvrv():
    """下载 STH-MVRV 指标的历史数据"""
    client = BGClient()
    df = client.get_sth_mvrv()

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = dt.datetime.now().strftime("%Y%m%d")
    file_path = RAW_DATA_DIR / f"sth_mvrv_{timestamp}.csv"

    df.to_csv(file_path, index=True)
    logger.info(f"STH-MVRV 数据已下载并保存到 {file_path}")


def download_btcusd():
    """从雅虎财经下载 BTCUSD 历史价格"""
    data = yf.download(
        tickers="BTC-USD",
        start=dt.datetime(2014, 1, 1),
        end=dt.datetime.now(),
        ignore_tz=True,
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )

    data = data[["Open", "High", "Low", "Close", "Volume"]]
    data.index.name = "datetime"
    data.columns = [x.lower() for x in data.columns]

    timestamp = dt.datetime.now().strftime("%Y%m%d")
    file_path = RAW_DATA_DIR / f"BTCUSD_{timestamp}.csv"

    data.to_csv(file_path, index=True)
    logger.info(f"BTCUSD 数据已下载并保存到 {file_path}")


if __name__ == "__main__":
    download_sth_mvrv()
    download_btcusd()
