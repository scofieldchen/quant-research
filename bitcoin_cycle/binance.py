import datetime as dt
import io
import zipfile
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from rich import print
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# API下载和处理指标数据的逻辑
METRICS_CONFIG = {
    "open_interest": {
        "api_url": "/futures/data/openInterestHist",
        "rename_columns": {
            "sumOpenInterest": "sum_open_interest",
            "sumOpenInterestValue": "sum_open_interest_value",
        },
        "drop_columns": ["symbol"],
    },
    "top_trader_long_short_account_ratio": {
        "api_url": "/futures/data/topLongShortAccountRatio",
        "rename_columns": {
            "longShortRatio": "count_toptrader_long_short_ratio",
        },
        "drop_columns": ["symbol", "longAccount", "shortAccount"],
    },
    "top_trader_long_short_position_ratio": {
        "api_url": "/futures/data/topLongShortPositionRatio",
        "rename_columns": {
            "longShortRatio": "sum_toptrader_long_short_ratio",
        },
        "drop_columns": ["symbol", "longAccount", "shortAccount"],
    },
    "global_long_short_account_ratio": {
        "api_url": "/futures/data/globalLongShortAccountRatio",
        "rename_columns": {
            "longShortRatio": "count_long_short_ratio",
        },
        "drop_columns": ["symbol", "longAccount", "shortAccount"],
    },
    "taker_long_short_ratio": {
        "api_url": "/futures/data/takerlongshortRatio",
        "rename_columns": {"buySellRatio": "sum_taker_long_short_vol_ratio"},
        "drop_columns": ["buyVol", "sellVol"],
    },
}


def generate_date_range(
    start_date: dt.datetime, end_date: dt.datetime
) -> List[dt.datetime]:
    return [
        start_date + dt.timedelta(days=i)
        for i in range((end_date - start_date).days + 1)
    ]


class DataFetchError(Exception):
    pass


class InvalidRequestError(DataFetchError):
    pass


class NetworkError(DataFetchError):
    pass


class DataProcessError(Exception):
    pass


class FutureMetricsDownloader(ABC):
    """下载永续合约交易指标的抽象基类"""

    def __init__(self, data_directory: str):
        self.data_directory = Path(data_directory)

    @abstractmethod
    def fetch_metrics_by_date(self, symbol: str, date: dt.datetime) -> pd.DataFrame:
        """下载指定交易对和日期的指标数据"""
        pass

    def _download(
        self, symbol: str, date: dt.datetime, data_dir: Path
    ) -> Tuple[bool, Any]:
        """下载指定交易对和日期的数据并存储到csv"""
        try:
            df = self.fetch_metrics_by_date(symbol, date)
            filepath = data_dir / f"{date:%Y%m%d}.csv"
            df.to_csv(filepath, index=True)
            return True, None
        except Exception as e:
            return False, e

    def download(
        self,
        symbol: str,
        start_date: dt.datetime,
        end_date: dt.datetime,
        max_workers: int = 3,
    ) -> None:
        """并行下载指定日期范围内的数据

        Args:
            symbol: 交易对符号，'BTCUSDT'
            start_date: 开始日期
            end_date: 结束日期
            max_workers: 最大线程数
        """
        if start_date > end_date:
            raise ValueError("Start date must be before end date")

        # 创建数据目录
        self.data_directory.mkdir(parents=True, exist_ok=True)
        symbol_dir = self.data_directory / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)

        # 生成日期列表
        dates = generate_date_range(start_date, end_date)

        # 使用线程池并行下载
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有下载任务
            futures = {
                executor.submit(self._download, symbol, date, symbol_dir): date
                for date in dates
            }

            # 处理完成的任务
            for future in as_completed(futures):
                date = futures[future]
                success, exception = future.result()
                if success:
                    print(f"Downloaded {symbol} data for {date:%Y-%m-%d}")
                else:
                    print(
                        f"Failed to download {symbol} data for {date:%Y-%m-%d}: {str(exception)}"
                    )


class HistoricalFutureMetricsDownloader(FutureMetricsDownloader):
    """从历史仓库下载合约的交易指标数据"""

    BASE_URL = "https://data.binance.vision/data/futures/um/daily/metrics/"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(NetworkError),
        reraise=True,
    )
    def _fetch_zip(self, symbol: str, date: dt.datetime) -> requests.Response:
        # 构建请求url
        date_str = date.strftime("%Y-%m-%d")
        filename = f"{symbol}/{symbol}-metrics-{date_str}.zip"
        url = HistoricalFutureMetricsDownloader.BASE_URL + filename

        # 下载数据（zip压缩文件）
        try:
            response = requests.get(url, stream=True, timeout=5)
            if response.status_code == 404:
                raise InvalidRequestError(f"Invalid symbol({symbol}) or date({date})")
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise NetworkError(f"Request failed due to network errors: {e}")

        return response

    def _process_zip(self, response: requests.Response) -> pd.DataFrame:
        try:
            # 使用内存中的BytesIO对象处理zip文件，避免写入磁盘
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                # 获取zip文件中的CSV文件名（只有一个文件）
                csv_files = [
                    name for name in zip_file.namelist() if name.endswith(".csv")
                ]
                csv_filename = csv_files[0]

                # 读取CSV文件到数据框
                with zip_file.open(csv_filename) as csv_file:
                    df = pd.read_csv(
                        csv_file, index_col="create_time", parse_dates=True
                    )
                    df.index.name = "datetime"
                    df.index = df.index.tz_localize("UTC")
                    df.drop(columns=["symbol"], inplace=True)
                    return df
        except Exception as e:
            raise DataProcessError(str(e))

    def fetch_metrics_by_date(self, symbol: str, date: dt.datetime) -> pd.DataFrame:
        resp = self._fetch_zip(symbol, date)
        return self._process_zip(resp)


class APIFutureMetricsDownloader(FutureMetricsDownloader):
    """从API下载合约交易指标数据"""

    BASE_URL = "https://fapi.binance.com"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(NetworkError),
        reraise=True,
    )
    def _fetch_one_metric_by_date(
        self,
        api_url: str,
        symbol: str,
        date: dt.datetime,
    ) -> List[Dict[str, Any]]:
        """下载单个指标指定日期的数据"""
        # 从历史仓库下载的日数据从当天的05:00开始，并包含次日第一根5分钟柱子
        date_start = date.replace(hour=0, minute=5, second=0, microsecond=0)
        date_end = (date + dt.timedelta(days=1)).replace(
            hour=0, minute=5, second=0, microsecond=0
        )
        start_timestamp = int(date_start.timestamp() * 1000)
        end_timestamp = int(date_end.timestamp() * 1000)

        try:
            response = requests.get(
                APIFutureMetricsDownloader.BASE_URL + api_url,
                params={
                    "symbol": symbol,
                    "period": "5m",
                    "limit": 500,
                    "startTime": start_timestamp,
                    "endTime": end_timestamp,
                },
            )

            # 当参数错误时响应状态码为200且返回空列表
            data = response.json()
            if not data:
                raise InvalidRequestError(f"Invalid symbol({symbol}) or date({date})")

            return data

        except requests.exceptions.RequestException as e:
            raise NetworkError(f"Request failed due to network errors: {e}")

    def _process_metric(
        self,
        data: List[Dict[str, Any]],
        rename_columns: Optional[Dict[str, str]] = None,
        drop_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """处理响应数据并转化为数据框"""
        # 通用处理逻辑
        df = pd.DataFrame.from_records(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.sort_values("timestamp", ascending=True, inplace=True)
        df.rename(columns={"timestamp": "datetime"}, inplace=True)
        df.set_index("datetime", inplace=True)

        # 重命名字段
        if rename_columns:
            df.rename(columns=rename_columns, inplace=True)

        # 删除字段
        if drop_columns:
            df.drop(columns=drop_columns, inplace=True)

        return df

    def fetch_metrics_by_date(self, symbol: str, date: dt.datetime) -> pd.DataFrame:
        """获取指定交易对和日期的指标数据"""
        # 用列表存储所有指标
        all_data: List[pd.DataFrame] = []

        # 逐个获取指标
        for metric_name, config in METRICS_CONFIG.items():
            data = self._fetch_one_metric_by_date(config["api_url"], symbol, date)
            processed_data = self._process_metric(
                data, config["rename_columns"], config["drop_columns"]
            )

            # 净持仓量数据需要向右移动一期，才能跟历史仓库数据保持一致，理由不明
            if metric_name == "taker_long_short_ratio":
                processed_data = processed_data.shift(1).dropna()

            all_data.append(processed_data)

        # 合并数据
        joined_data = pd.concat(all_data, axis=1)
        joined_data = joined_data.astype(float)
        joined_data = joined_data.dropna()

        return joined_data


if __name__ == "__main__":
    # 参数
    symbol = "BTCUSDT"
    start_date = dt.datetime(2025, 3, 10, tzinfo=dt.UTC)
    end_date = dt.datetime(2025, 3, 15, tzinfo=dt.UTC)
    data_directory = "/users/scofield/quant-research/bitcoin_cycle/data2"

    # 从历史仓库下载数据
    # historical_downloader = HistoricalFutureMetricsDownloader(data_directory)
    # historical_downloader.download(symbol, start_date, end_date)

    # 从API下载数据
    api_downloader = APIFutureMetricsDownloader(data_directory)
    api_downloader.download(symbol, start_date, end_date)
