"""Binance k线数据管道异常模块。"""

from __future__ import annotations
import datetime as dt


class DataNotFoundError(Exception):
    """数据不存在异常，用于标识数据源 404 的情况。

    该异常表示请求的数据在数据源中不存在（如月份未生成或交易对未上市），
    区别于网络错误或其他临时问题。不应进行重试。
    """

    def __init__(self, message: str, url: str | None = None) -> None:
        self.url = url
        super().__init__(message)


class IncompleteUpdateError(Exception):
    """更新不完整异常。

    当增量更新某些交易对时，如果其中某些日期的数据下载失败，抛出此异常。
    包含失败的具体日期列表，以便进行补漏。
    """

    def __init__(self, message: str, symbol: str, failed_dates: list[dt.date]) -> None:
        self.symbol = symbol
        self.failed_dates = failed_dates
        super().__init__(message)
