import datetime as dt
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.core.bgeometrics import BGClient
from src.core.logger import get_logger

# 数据目录
RAW_DATA_DIR = Path("data/raw/sth_mvrv")
CLEANED_DATA_DIR = Path("data/cleaned")

# 需要设置代理才能请求数据
yf.config.network.proxy = {
    "http": "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890",
}

logger = get_logger("sthmvrv")


def download_sth_mvrv() -> Path:
    """下载 STH-MVRV 指标的历史数据"""
    client = BGClient()
    df = client.get_sth_mvrv()

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = dt.datetime.now().strftime("%Y%m%d")
    file_path = RAW_DATA_DIR / f"sth_mvrv_{timestamp}.csv"

    df.to_csv(file_path, index=True)
    logger.info(f"STH-MVRV 数据已下载并保存到 {file_path}")

    return file_path


def download_btcusd() -> Path:
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

    return file_path


def preprocess_data(sth_mvrv_file_path: Path, btcusd_file_path: Path):
    """预处理数据：合并、清理并保存到 cleaned 目录"""
    # 读取 STH-MVRV 数据
    logger.info(f"使用 STH-MVRV 数据: {sth_mvrv_file_path}")
    sth_mvrv_df = pd.read_csv(sth_mvrv_file_path, index_col=0, parse_dates=True)
    sth_mvrv_df.index.name = "datetime"

    # 读取 BTCUSD 数据
    logger.info(f"使用 BTCUSD 数据: {btcusd_file_path}")
    btcusd_df = pd.read_csv(btcusd_file_path, index_col=0, parse_dates=True)
    btcusd_df.index.name = "datetime"
    btcusd_df = btcusd_df[["close"]]

    # 合并数据，按照时间戳合并
    merged_df = pd.merge(
        sth_mvrv_df, btcusd_df, left_index=True, right_index=True, how="inner"
    )

    # 删除所有缺失值
    merged_df.dropna(inplace=True)

    # 保存到 cleaned 目录
    cleaned_file_path = CLEANED_DATA_DIR / f"sth_mvrv.parquet"
    merged_df.to_parquet(cleaned_file_path, index=True)
    logger.info(f"预处理后的数据已保存到 {cleaned_file_path}")


if __name__ == "__main__":
    sth_mvrv_path = download_sth_mvrv()
    btcusd_path = download_btcusd()
    preprocess_data(sth_mvrv_path, btcusd_path)
