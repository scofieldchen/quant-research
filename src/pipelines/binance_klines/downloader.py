"""Binance k线数据下载模块。"""

import datetime as dt
import time
import zipfile
from enum import Enum
from io import BytesIO

import ccxt
import pandas as pd
import requests
from requests.exceptions import ConnectionError, ConnectTimeout
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.core.logger import get_logger

logger = get_logger(__name__)

KLINE_FIELDS = [
    "open_time",  # k线开盘时间，unix 时间戳格式
    "open",  # 开盘价
    "high",  # 最高价
    "low",  # 最低价
    "close",  # 收盘价
    "volume",  # 基础资产成交量
    "close_time",  # k线收盘时间，unix 时间戳格式
    "quote_volume",  # 报价资产成交量
    "count",  # 成交笔数
    "taker_buy_volume",  # 该期间 taker 买入基础资产量
    "taker_buy_quote_volume",  # 该期间 taker 买入报价资产量
    "ignore",  # 忽略
]


class AssetType(Enum):
    SPOT = "spot"
    PERP = "perp"


class PartitionInterval(Enum):
    DAILY = "daily"
    MONTHLY = "monthly"


class Granularity(Enum):
    KLINE_1_SECOND = "1s"
    KLINE_1_MINUTE = "1m"
    KLINE_3_MINUTE = "3m"
    KLINE_5_MINUTE = "5m"
    KLINE_15_MINUTE = "15m"
    KLINE_30_MINUTE = "30m"
    KLINE_1_HOUR = "1h"
    KLINE_2_HOUR = "2h"
    KLINE_4_HOUR = "4h"
    KLINE_6_HOUR = "6h"
    KLINE_12_HOUR = "12h"
    KLINE_1_DAY = "1d"
    KLINE_3_DAY = "3d"
    KLINE_1_WEEK = "1w"
    KLINE_1_MONTH = "1mo"


def get_spot_daily_url(ticker: str, granularity: Granularity, date: dt.datetime) -> str:
    base_url = "https://data.binance.vision/data/spot/daily/klines"
    url = f"{base_url}/{ticker.upper()}/{granularity.value}/{ticker.upper()}-{granularity.value}-{date:%Y-%m-%d}.zip"
    return url


def get_spot_monthly_url(
    ticker: str, granularity: Granularity, date: dt.datetime
) -> str:
    base_url = "https://data.binance.vision/data/spot/monthly/klines"
    url = f"{base_url}/{ticker.upper()}/{granularity.value}/{ticker.upper()}-{granularity.value}-{date:%Y-%m}.zip"
    return url


def get_perp_daily_url(ticker: str, granularity: Granularity, date: dt.datetime) -> str:
    base_url = "https://data.binance.vision/data/futures/um/daily/klines"
    url = f"{base_url}/{ticker.upper()}/{granularity.value}/{ticker.upper()}-{granularity.value}-{date:%Y-%m-%d}.zip"
    return url


def get_perp_monthly_url(
    ticker: str, granularity: Granularity, date: dt.datetime
) -> str:
    base_url = "https://data.binance.vision/data/futures/um/monthly/klines"
    url = f"{base_url}/{ticker.upper()}/{granularity.value}/{ticker.upper()}-{granularity.value}-{date:%Y-%m}.zip"
    return url


GET_URLS = {
    (AssetType.SPOT, PartitionInterval.DAILY): get_spot_daily_url,
    (AssetType.SPOT, PartitionInterval.MONTHLY): get_spot_monthly_url,
    (AssetType.PERP, PartitionInterval.DAILY): get_perp_daily_url,
    (AssetType.PERP, PartitionInterval.MONTHLY): get_perp_monthly_url,
}


def get_url(
    ticker: str,
    granularity: Granularity,
    date: dt.datetime,
    asset_type: AssetType,
    partition: PartitionInterval,
) -> str:
    """获取下载数据的 URL"""
    try:
        get_url_func = GET_URLS[(asset_type, partition)]
    except KeyError:
        raise ValueError(f"无效的 asset_type {asset_type} 和 partition {partition}")
    else:
        return get_url_func(ticker, granularity, date)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((ConnectionError, ConnectTimeout)),
)
def get_zipfile(url: str, columns: list[str]) -> pd.DataFrame:
    """从 zipfile 获取数据帧

    Args:
        url: zipfile URL
        columns: 数据帧列名

    Returns:
        pd.DataFrame
    """
    resp = requests.get(url)
    resp.raise_for_status()

    with zipfile.ZipFile(BytesIO(resp.content)) as zip_file:
        with zip_file.open(zip_file.namelist()[0]) as csv_file:
            # 某些 csv 文件没有表头
            # 如果第一个字符是数字，则无表头，否则有表头
            first_char = csv_file.read(1)
            csv_file.seek(0)
            if str(first_char, encoding="UTF8").isdigit():
                return pd.read_csv(csv_file, header=None, names=columns)
            else:
                return pd.read_csv(csv_file)


def fetch_historical(
    ticker: str,
    granularity: Granularity,
    date: dt.datetime,
    asset_type: AssetType,
    partition: PartitionInterval,
) -> pd.DataFrame:
    """从 Binance Vision 获取历史 k线数据"""
    url = get_url(ticker, granularity, date, asset_type, partition)
    df = get_zipfile(url, KLINE_FIELDS)
    df = df.drop(columns=["close_time", "ignore"])
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.drop(columns=["open_time"])
    df = df.set_index("datetime")
    return df


def fetch_api(
    ticker: str,
    start_date: dt.datetime,
    end_date: dt.datetime | None = None,
    proxy: dict | None = None,
) -> pd.DataFrame:
    """从 Binance API 获取实时数据，使用 ccxt。

    Args:
        ticker: 交易对，例如 'btcusdt'
        start_date: 数据开始日期
        end_date: 数据结束日期，默认当前时间
        proxy: 代理设置

    Returns:
        pd.DataFrame: OHLCV 数据
    """
    exchange = ccxt.binanceusdm(
        {
            "enableRateLimit": True,
            "rateLimit": 1000,  # 每秒 1 个请求
        }
    )
    if proxy:
        exchange.proxies = proxy

    if end_date is None:
        end_date = dt.datetime.now(dt.timezone.utc)

    start_ts = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)

    res = []
    while start_ts < end_ts:
        try:
            data = exchange.fetch_ohlcv(
                ticker.upper(), timeframe="1m", since=start_ts, limit=1000
            )
            if not data:
                break
            res.extend(data)
            start_ts = data[-1][0] + 60000  # 下一分钟
            time.sleep(1)  # 遵守限速
        except Exception as e:
            logger.error(f"获取 {ticker} 数据时出错: {e}")
            break

    if not res:
        return pd.DataFrame(
            columns=["datetime", "open", "high", "low", "close", "volume"]
        )

    df = (
        pd.DataFrame(
            res, columns=["datetime", "open", "high", "low", "close", "volume"]
        )
        .assign(datetime=lambda x: pd.to_datetime(x["datetime"], unit="ms", utc=True))
        .set_index("datetime")
        .sort_index()
        .loc[start_date:end_date]
    )

    return df
