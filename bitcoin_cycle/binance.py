import datetime as dt
import io
import time
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)


class DataProcessError(Exception):
    pass


class FutureMetricsDownloader(ABC):
    """下载永续合约交易指标的抽象基类"""

    @abstractmethod
    def download(
        self, symbol: str, start_date: dt.datetime, end_date: dt.datetime
    ) -> None:
        pass


class HistoricalFutureMetricsDownloader(FutureMetricsDownloader):
    """从历史仓库下载合约交易指标数据"""

    BASE_URL = "https://data.binance.vision/data/futures/um/daily/metrics/"

    def __init__(self, data_directory: str) -> None:
        self.data_directory = Path(data_directory)

        # 创建存储数据的文件夹
        self.data_directory.mkdir(parents=True, exist_ok=True)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(
            (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError,
            )
        ),
    )
    def _fetch_daily_metrics(self, symbol: str, date: dt.datetime) -> requests.Response:
        # 构建请求url
        date_str = date.strftime("%Y-%m-%d")
        filename = f"{symbol}/{symbol}-metrics-{date_str}.zip"
        url = HistoricalFutureMetricsDownloader.BASE_URL + filename

        # 下载数据，如果参数错误直接引发异常，如果网络连接错误则进行重试
        response = requests.get(url, stream=True, timeout=5)
        response.raise_for_status()

        return response

    def _process_daily_metrics(self, response: requests.Response) -> pd.DataFrame:
        try:
            # 使用内存中的BytesIO对象处理zip文件，避免写入磁盘
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                # 获取zip文件中的CSV文件名（通常只有一个文件）
                csv_files = [
                    name for name in zip_file.namelist() if name.endswith(".csv")
                ]
                csv_filename = csv_files[0]

                # 读取CSV文件内容
                with zip_file.open(csv_filename) as csv_file:
                    df = pd.read_csv(
                        csv_file, index_col="create_time", parse_dates=True
                    )
                    df.index.name = "datetime"
                    return df
        except Exception as e:
            raise DataProcessError(str(e))

    def download_daily_metrics(self, symbol: str, date: dt.datetime) -> None:
        # 先创建存储交易对数据的子文件夹
        symbol_data_dir = self.data_directory / symbol
        if not symbol_data_dir.exists():
            symbol_data_dir.mkdir()

        # 下载并存储数据
        try:
            resp = self._fetch_daily_metrics(symbol, date)
            df = self._process_daily_metrics(resp)
        except requests.exceptions.HTTPError as e:
            print(
                f"下载数据失败，参数 symbol({symbol}) 或 date({date:%Y-%m-%d}) 可能无效: {str(e)}"
            )
        except RetryError as e:
            print(
                f"下载数据失败（多次重试失败），symbol={symbol} date={date:%Y-%m-%d}，检查网络连接"
            )
        except DataProcessError as e:
            print(f"处理数据失败，symbol={symbol} date={date:%Y-%m-%d}: {str(e)}")
        else:
            filepath = symbol_data_dir / f"{date.strftime("%Y%m%d")}.csv"
            df.to_csv(filepath, index=True)

    def download(
        self, symbol: str, start_date: dt.datetime, end_date: dt.datetime
    ) -> None:
        pass


class APIFutureMetricsDownloader(FutureMetricsDownloader):
    """从API下载合约交易指标数据"""

    pass


symbol = "BTCUSDT"
date = dt.datetime(2025, 3, 26)
data_directory = "/users/scofield/quant-research/bitcoin_cycle/data"

historical_metrics_downloader = HistoricalFutureMetricsDownloader(data_directory)
historical_metrics_downloader.download_daily_metrics(symbol, date)
