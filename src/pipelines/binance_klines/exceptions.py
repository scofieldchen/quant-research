"""Binance k线数据管道异常模块。"""

from __future__ import annotations


class DataNotFoundError(Exception):
    """数据不存在异常，用于标识数据源 404 的情况。

    该异常表示请求的数据在数据源中不存在（如月份未生成或交易对未上市），
    区别于网络错误或其他临时问题。不应进行重试。
    """

    def __init__(self, message: str, url: str | None = None) -> None:
        self.url = url
        super().__init__(message)
