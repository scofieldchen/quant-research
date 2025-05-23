import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from rich.console import Console
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

console = Console()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(requests.RequestException),
    reraise=True,
)
def _fetch_funding_rate_page(
    symbol: str,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    limit: int = 1000,
) -> List[Dict[str, Any]]:
    """获取单页融资利率数据

    Args:
        symbol (str): 交易对名称，例如 "BTCUSDT"
        start_time (Optional[int], optional): 开始时间戳（毫秒）
        end_time (Optional[int], optional): 结束时间戳（毫秒）
        limit (int, optional): 每页数据量，默认1000（API最大值）

    Returns:
        List[Dict[str, Any]]: 融资利率数据列表

    Raises:
        NetworkError: 网络连接错误
        DataFetchError: 数据获取错误
    """
    base_url = "https://fapi.binance.com/fapi/v1/fundingRate"
    params = {"symbol": symbol, "limit": limit}

    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time

    response = requests.get(base_url, params=params, timeout=10)
    response.raise_for_status()

    return response.json()


def get_funding_rate(
    symbol: str,
    start_date: dt.datetime,
    end_date: dt.datetime,
    max_workers: int = 5,
) -> pd.DataFrame:
    """获取指定时间范围内的永续合约融资利率数据

    该函数会自动处理分页和并发请求，确保获取指定日期范围内的所有数据。

    Args:
        symbol (str): 交易对名称，例如 "BTCUSDT"
        start_date (dt.datetime): 开始日期，时区默认为 utc
        end_date (dt.datetime): 结束日期，时区默认为 utc
        max_workers (int, optional): 最大并发请求数，默认为5

    Returns:
        pd.DataFrame: 包含融资利率数据的DataFrame，列包括:
            - fundingTime: 融资时间
            - fundingRate: 融资利率
            - symbol: 交易对名称
            - date: 日期时间格式的融资时间
    """
    symbol = symbol.upper()
    start_timestamp = int(start_date.timestamp() * 1000)
    end_timestamp = int(end_date.timestamp() * 1000)

    # Binance API限制每次请求最多返回1000条记录
    # 融资利率每8小时结算一次，所以每天3条记录
    # 计算需要的时间窗口数量
    time_span = end_timestamp - start_timestamp
    days = time_span / (24 * 60 * 60 * 1000)
    estimated_records = days * 3

    # 如果预计记录数超过单次请求限制，需要分页请求
    if estimated_records > 1000:
        # 计算时间窗口
        # 每个窗口最多包含约330天的数据（1000条记录）
        window_size = 330 * 24 * 60 * 60 * 1000  # 毫秒

        # 创建时间窗口列表
        time_windows = []
        current_start = start_timestamp

        while current_start < end_timestamp:
            current_end = min(current_start + window_size, end_timestamp)
            time_windows.append((current_start, current_end))
            current_start = current_end

        # 并发请求各个时间窗口的数据
        all_data = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_fetch_funding_rate_page, symbol, start, end)
                for start, end in time_windows
            ]

            for future in futures:
                try:
                    data = future.result()
                    all_data.extend(data)
                except Exception as e:
                    print(e)
    else:
        # 数据量较小，直接请求
        try:
            all_data = _fetch_funding_rate_page(symbol, start_timestamp, end_timestamp)
        except Exception as e:
            print(e)

    # 如果没有数据，引发异常
    if not all_data:
        raise Exception("No data found")

    # 转换为DataFrame
    df = pd.DataFrame(all_data)

    return df


def process_funding_rate(df: pd.DataFrame) -> pd.DataFrame:
    """处理融资利率，转化为日频数据"""
    return (
        df.assign(
            datetime=lambda x: pd.to_datetime(x["fundingTime"], unit="ms", utc=True),
            fundingRate=lambda x: pd.to_numeric(x["fundingRate"]),
        )
        .sort_values("datetime")
        .set_index("datetime")
        .drop(columns=["symbol", "fundingTime", "markPrice"])
        .rename(columns={"fundingRate": "funding_rate"})
        .resample("D")
        .sum()  # 计算日总融资利率
    )


def download_funding_rate(
    filepath: Path, symbol: str, start_date: dt.datetime, end_date: dt.datetime
) -> None:
    """下载指定交易对和时间范围的融资利率并保存到本地"""
    try:
        df = get_funding_rate(symbol, start_date, end_date)
        df_processed = process_funding_rate(df)
        df_processed.index = df_processed.index.tz_convert(None)
        console.print(
            f"✅ Downloaded funding rate for {symbol}, last:{df_processed.index.max():%Y-%m-%d}"
        )
    except Exception as e:
        console.print(f"[red]Failed to download funding rate for {symbol}: {str(e)}")
    else:
        df_processed.to_csv(filepath, index=True)


if __name__ == "__main__":
    filepath = Path(
        "/users/scofield/quant-research/bitcoin_cycle/data/funding_rate.csv"
    )
    start_date = dt.datetime(2019, 1, 1, tzinfo=dt.timezone.utc)
    end_date = dt.datetime(2025, 5, 30, tzinfo=dt.timezone.utc)
    download_funding_rate(filepath, "BTCUSDT", start_date, end_date)
