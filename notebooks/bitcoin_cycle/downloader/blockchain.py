from pathlib import Path
from typing import List

import pandas as pd
import requests
from rich.console import Console
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential

# 模块级配置
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
REQUEST_TIMEOUT = 5
METRICS_URL_MAP = {
    "sth_realized_price": "https://charts.bgeometrics.com/files/sth_realized_price.json",
    "sth_sopr": "https://charts.bgeometrics.com/files/sth_sopr.json",
    "sth_nupl": "https://charts.bgeometrics.com/files/sth_nupl.json",
    "sth_mvrv": "https://charts.bgeometrics.com/files/sth_mvrv.json",
    "miner_sell_presure": "https://charts.bgeometrics.com/files/miner_sell_presure.json",
    "nrpl": "https://charts.bgeometrics.com/files/nrpl.json",
    "realized_profit_loss_ratio": "https://charts.bgeometrics.com/files/realized_profit_loss_ratio.json",
    "rhodl": "https://charts.bgeometrics.com/files/rhodl.json",
    "vdd_multiple": "https://charts.bgeometrics.com/files/vdd_multiple.json",
    "puell_multiple": "https://charts.bgeometrics.com/files/puell_multiple.json",
    "nupl": "https://charts.bgeometrics.com/files/nupl.json",
    "mvrv_zscore": "https://charts.bgeometrics.com/files/mvrv_zscore.json",
}

console = Console()


class DataFetchError(Exception):
    """获取数据失败时抛出的异常"""

    def __init__(self, metric_name: str, error_message: str) -> None:
        message = f"下载 {metric_name} 数据时发生错误: {error_message}"
        super().__init__(message)


class DataProcessError(Exception):
    """处理数据失败时抛出的异常"""

    def __init__(self, metric_name: str, error_message: str) -> None:
        message = f"处理 {metric_name} 数据时发生错误: {error_message}"
        super().__init__(message)


class BGClient:
    """bgeometrics.com 数据获取客户端"""

    def __init__(self, timeout: int = REQUEST_TIMEOUT):
        """初始化客户端

        Args:
            timeout: 请求超时时间(秒)
        """
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _fetch_data(self, url: str) -> requests.Response:
        """内部数据获取方法

        Args:
            url: 请求的API端点URL

        Returns:
            包含API响应的Response对象

        Raises:
            requests.exceptions.RequestException: 当请求失败时抛出
        """
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response

    def _process_response(
        self, response: requests.Response, metric_name: str
    ) -> pd.DataFrame:
        """处理API响应数据

        Args:
            response: 包含API数据的Response对象
            metric_name: 指标名称

        Returns:
            处理后的DataFrame，包含datetime索引和指标值

        Raises:
            DataProcessError: 当数据处理失败时抛出
        """
        try:
            data = response.json()
            df = pd.DataFrame.from_records(data)
            df.columns = ["datetime", metric_name]
            df.dropna(inplace=True)
            df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
            df.sort_values("datetime", ascending=True, inplace=True)
            df.set_index("datetime", inplace=True)
            return df
        except Exception as e:
            raise DataProcessError(metric_name, str(e))

    def get_metric(self, metric_name: str) -> pd.DataFrame:
        """获取指定指标数据

        Args:
            metric_name: 指标名称，必须是METRICS_URL_MAP中定义的

        Returns:
            包含指标数据的DataFrame

        Raises:
            ValueError: 当请求不支持的指标时抛出
            DataFetchError: 当数据获取失败时抛出
            DataProcessError: 当数据处理失败时抛出
        """
        if metric_name not in METRICS_URL_MAP:
            raise ValueError(
                f"不支持的指标: {metric_name}。可用指标: {list(METRICS_URL_MAP.keys())}"
            )

        try:
            url = METRICS_URL_MAP[metric_name]
            response = self._fetch_data(url)
            return self._process_response(response, metric_name)
        except RetryError as e:
            raise DataFetchError(metric_name, str(e))

    def get_sth_realized_price(self) -> pd.DataFrame:
        return self.get_metric("sth_realized_price")

    def get_sth_sopr(self) -> pd.DataFrame:
        return self.get_metric("sth_sopr")

    def get_sth_nupl(self) -> pd.DataFrame:
        return self.get_metric("sth_nupl")

    def get_sth_mvrv(self) -> pd.DataFrame:
        return self.get_metric("sth_mvrv")

    def get_miner_sell_presure(self) -> pd.DataFrame:
        return self.get_metric("miner_sell_presure")

    def get_nrpl(self) -> pd.DataFrame:
        return self.get_metric("nrpl")

    def get_realized_profit_loss_ratio(self) -> pd.DataFrame:
        return self.get_metric("realized_profit_loss_ratio")

    def get_rhodl(self) -> pd.DataFrame:
        return self.get_metric("rhodl")

    def get_vdd_multiple(self) -> pd.DataFrame:
        return self.get_metric("vdd_multiple")

    def get_puell_multiple(self) -> pd.DataFrame:
        return self.get_metric("puell_multiple")

    def get_nupl(self) -> pd.DataFrame:
        return self.get_metric("nupl")

    def get_mvrv_zscore(self) -> pd.DataFrame:
        return self.get_metric("mvrv_zscore")


def download_blockchain_metrics(data_directory: Path, metric_names: List[str]) -> None:
    """从 bgeometrics 下载区块链数据"""
    client = BGClient()

    for metric in metric_names:
        try:
            df = client.get_metric(metric)
            console.print(
                f"✅ Downloaded blockchain metric: {metric}, last:{df.index.max():%Y-%m-%d}"
            )
        except Exception as e:
            console.print(
                f"[red]Failed to download blockchain metric: {metric}, error:{str(e)}"
            )
        else:
            filepath = data_directory / f"{metric}.csv"
            df.to_csv(filepath, index=True)


if __name__ == "__main__":
    metrics = [
        # "sth_realized_price",
        # "sth_sopr",
        # "sth_nupl",
        # "sth_mvrv",
        # "nrpl",
        "rhodl",
        "vdd_multiple",
        "puell_multiple",
        "nupl",
        "mvrv_zscore",
    ]
    download_blockchain_metrics(
        Path("/users/scofield/quant-research/bitcoin_cycle/data"), metrics
    )
