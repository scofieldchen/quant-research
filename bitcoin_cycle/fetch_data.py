import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError


class DataFetchError(Exception):
    def __init__(self, metric_name: str, error_message: str) -> None:
        message = f"下载 {metric_name} 数据时发生错误: {error_message}"
        super().__init__(message)


class DataProcessError(Exception):
    def __init__(self, metric_name: str, error_message: str) -> None:
        message = f"处理 {metric_name} 数据时发生错误: {error_message}"
        super().__init__(message)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_bgeometrics(url: str, timeout: int = 5) -> requests.Response:
    """从 bgeometrics 网站获取区块链数据。

    Args:
        url: 数据API端点URL
        timeout: 请求超时时间(秒)，默认为5秒

    Returns:
        包含API响应的Response对象

    Raises:
        requests.exceptions.RequestException: 当所有重试都失败时抛出
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response


def process_response(response: requests.Response, metric_name: str) -> pd.DataFrame:
    """处理响应数据并转化为pandas数据框

    所有响应数据都遵循以下格式：
    [
        [
            1325376000000,  # 时间戳，精确到毫秒
            1309.6871123    # 指标值
        ],
        [
            1325462400000,
            1648.8334755
        ],
    ]

    Args:
        response: 包含API数据的Response对象
        metric_name: 指标名称

    Returns:
        包含预处理数据的数据框，包含字段:
        - datetime: 时间戳(UTC时区)
        - metric_name: 指标值
    """
    data = response.json()
    df = pd.DataFrame.from_records(data)
    df.columns = ["datetime", metric_name]
    df.dropna(inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
    df.sort_values("datetime", ascending=True, inplace=True)
    df.set_index("datetime", inplace=True)
    return df


def get_data(url: str, metric_name: str) -> pd.DataFrame:
    """获取数据的公共接口"""
    try:
        resp = fetch_bgeometrics(url)
        return process_response(resp, metric_name)
    except RetryError as e:
        raise DataFetchError(metric_name, str(e))
    except Exception as e:
        raise DataProcessError(metric_name, str(e))


def get_sth_realized_price() -> pd.DataFrame:
    return get_data(
        "https://charts.bgeometrics.com/files/sth_realized_price.json",
        "sth_realized_price",
    )


def get_sth_sopr() -> pd.DataFrame:
    return get_data("https://charts.bgeometrics.com/files/sth_sopr.json", "sth_sopr")


def get_sth_nupl() -> pd.DataFrame:
    return get_data("https://charts.bgeometrics.com/files/sth_nupl.json", "sth_nupl")


def get_sth_mvrv() -> pd.DataFrame:
    return get_data("https://charts.bgeometrics.com/files/sth_mvrv.json", "sth_mvrv")


def get_miner_sell_presure() -> pd.DataFrame:
    return get_data(
        "https://charts.bgeometrics.com/files/miner_sell_presure.json",
        "miner_sell_presure",
    )


if __name__ == "__main__":
    # df = get_sth_realized_price()
    # df = get_sth_sopr()
    # df = get_sth_nupl()
    # df = get_sth_mvrv()
    df = get_miner_sell_presure()
    print(df.head())
    print(df.tail())
    print(df.info())
