## 任务

创建数据管道`src/pipelines/binance_klines`，下载和更新币安永续合约的历史k线。

## 背景和目标

数据管道需要解决两个问题：1. 下载全部永续合约的1分钟历史k线；2. 增量更新1分钟最新数据。

这些数据将作为后续量化研究和回溯检验的基础。

运行管道：`uv run python -m src.pipelines.binance_klines.task`

从binance数据仓库（https://data.binance.vision/）下载1分钟k线数据，保存到`data/raw/binance_klines_perp`。

历史数据分区策略：交易对 + 日期时间

```
data/binance_klines_perp
  btcusdt
    20210101.csv
    20210102.csv
    ...
  ethusdt
    20250101.csv
    20250102.csv
    ...
```

从官方API获取最新数据，使用`ccxt`实现。

## 核心逻辑

- 读取`data/cleaned/binance_tickers_perp.parquet`，获取全部交易对。
- 遍历交易对（多线程）
  - 若数据库不存在先创建
  - 检查是否包含该交易对，如果不包含，表明这是新的交易对，下载全部历史数据。如果包含，获取最后的时间戳，下载增量数据。
    - 下载增量数据时优先从历史仓库下载，如果仓库没有更新，从APIxi下载。
  - 将原始数据保存到`data/raw/binance_klines_perp`
- 下载完成后清洗数据，聚合所有交易对的分钟k线，更新`data/cleaned/binance_klines_perp_m1.parquet`

更新数据的细节：
- 从清洗数据(parquet)查询最新信息，确定最后一天的时间戳，覆盖最后一天的数据。
- 例如BTCUSDT的最后一条记录的时间戳是`2026-01-18 14:25:00+00`，从`2026-01-18 00:00:00+00`开始下载最新数据，目的是覆盖最后一天的数据，以确保分钟级数据没有遗漏。

清洗数据核心字段：
- datetime: 日期时间，包含时区，使用UTC
- open: 开盘价
- high: 最高价
- low: 最低价
- close: 收盘价
- volume: 成交量
- 其它字段

## 技术约束

- 使用`requests`处理网络请求，`tenacity`处理重试逻辑
- 使用`ccxt`从官方API获取1分钟k线
- 使用`duckdb`作为查询引擎，操作parquet数据库
- 使用`pandas`操作数据
- 使用`src.core.logger`创建的自定义logger
- 使用`typer`添加命令行参数

## 验证标准

因为要处理的交易对非常多（几百个），因此需要创建单独的命令行参数来提供测试，例如：

`uv run python -m src.pipelines.binance_klines.task -s btcusdt,ethusdt --start-date 20260101 --end-date 20260103`

## 待定问题（需要讨论）

1. 历史数据的分区策略是否合理？是否符合最佳实践？
2. 清洗数据时是否需要进行重采样？例如将1分钟k线重采样为5分钟，1小时，四小时，日图k线？后续分析需要使用不同时间框架的数据，是在获取数据时进行重采样？还是在分析阶段直接读取1分钟k线进行重采样？哪种方法的效率更高？

## 可以复用的代码

从历史数据仓库下载数据，这些代码来自于我的其它项目，可以复用其核心逻辑。

```python
import datetime as dt
import zipfile
from enum import Enum
from io import BytesIO

import pandas as pd
import requests
from requests.exceptions import ConnectionError, ConnectTimeout
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

KLINE_FIELDS = [
    "open_time",  # Kline Open time in unix time format
    "open",  # Open price
    "high",  # High price
    "low",  # Low price
    "close",  # Close price
    "volume",  # Base asset volume
    "close_time",  # Kline close time in unix time format
    "quote_volume",  # Quote asset volume
    "count",  # Number of trades
    "taker_buy_volume",  # Taker buy base asset volume during this period
    "taker_buy_quote_volume",  # Taker buy quote asset volume during this period
    "ignore",  # Ignore
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
    """Get url for downloading data"""
    try:
        get_url_func = GET_URLS[(asset_type, partition)]
    except KeyError:
        raise ValueError(f"Invalid asset_type {asset_type} and partition {partition}")
    else:
        return get_url_func(ticker, granularity, date)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((ConnectionError, ConnectTimeout)),
)
def get_zipfile(url: str, columns: list[str]) -> pd.DataFrame:
    """Get zipfile into dataframe

    Args:
        url (str): url of zipfile
        columns (List[str]): column names of dataframe

    Returns:
        pd.DataFrame
    """
    resp = requests.get(url)
    resp.raise_for_status()

    with zipfile.ZipFile(BytesIO(resp.content)) as zip_file:
        with zip_file.open(zip_file.namelist()[0]) as csv_file:
            # some csv files don't have headers
            # if the first character is digit, no header, otherwise has header
            first_char = csv_file.read(1)
            csv_file.seek(0)
            if str(first_char, encoding="UTF8").isdigit():
                return pd.read_csv(csv_file, header=None, names=columns)
            else:
                return pd.read_csv(csv_file)


def get_klines(
    ticker: str,
    granularity: Granularity,
    date: dt.datetime,
    asset_type: AssetType,
    partition: PartitionInterval,
) -> pd.DataFrame:
    """Get kline data from online store"""
    url = get_url(ticker, granularity, date, asset_type, partition)
    df = get_zipfile(url, KLINE_FIELDS)
    return (
        df.drop(columns=["close_time", "ignore"])
        .assign(open_time=lambda x: pd.to_datetime(x["open_time"], unit="ms", utc=True))
        .set_index("open_time")
    )
```

从binance api下载数据，可以复用代码并适当优化。

```python
def fetch_live(
    base_token: str, quote_token: str, date: dt.datetime, proxy: dict | None = None
) -> pd.DataFrame:
    """从 binance api 获取合约的分钟k线

    Args:
        base_token: 基础货币，如 "BTC"
        quote_token: 计价货币，如 "USDT"
        date: 日期时间
        proxy: 代理设置
    """
    # 创建 ccxt 交易所对象
    exchange = ccxt.binanceusdm()
    if proxy:
        exchange.proxies = proxy

    # 明确指定日期的开始和结束时间
    start_time = date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_time = date.replace(hour=23, minute=59, second=59, microsecond=999999)

    # 开始时间减5分钟以确保包含 00:00:00 这根k线，部分交易所不包含开始时间戳的数据
    start_ts = int((start_time - dt.timedelta(minutes=5)).timestamp() * 1000)
    end_ts = int(end_time.timestamp() * 1000)

    # 获取分钟数据
    res = []
    while start_ts < end_ts:
        # 新版 ccxt 的 symbol 格式不再添加分隔符“/”
        data = exchange.fetch_ohlcv(
            f"{base_token}{quote_token}", timeframe="1m", since=start_ts
        )
        if not data:
            break
        res.extend(data)
        start_ts = data[-1][0] + 1  # 更新开始时间戳
        time.sleep(exchange.rateLimit / 1000)  # 避免过度请求

    if not res:
        raise Exception("Data not found")

    # 清洗数据
    df = (
        pd.DataFrame(
            res, columns=["datetime", "open", "high", "low", "close", "volume"]
        )
        .assign(datetime=lambda x: pd.to_datetime(x["datetime"], unit="ms", utc=True))
        .set_index("datetime")
        .sort_index()
        .loc[start_time:end_time]
    )

    return df
```
