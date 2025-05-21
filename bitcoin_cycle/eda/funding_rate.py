import marimo

__generated_with = "0.13.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import sys

    sys.path.insert(0, "/users/scofield/quant-research/bitcoin_cycle/")

    import datetime as dt
    from typing import List, Dict, Any, Optional, Union
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    import requests
    import yfinance as yf
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_exponential,
        retry_if_exception_type,
    )

    from signals.funding_rate import FundingRate

    yf.set_config(
        proxy={"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}
    )
    return (
        Any,
        Dict,
        FundingRate,
        List,
        Optional,
        ThreadPoolExecutor,
        dt,
        pd,
        requests,
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
        yf,
    )


@app.cell
def _(
    Any,
    Dict,
    List,
    Optional,
    ThreadPoolExecutor,
    dt,
    mo,
    pd,
    requests,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    yf,
):
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
            start_date (dt.datetime): 开始日期
            end_date (dt.datetime): 结束日期
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
                all_data = _fetch_funding_rate_page(
                    symbol, start_timestamp, end_timestamp
                )
            except Exception as e:
                print(e)

        # 如果没有数据，引发异常
        if not all_data:
            raise Exception("No data found")

        # 转换为DataFrame
        df = pd.DataFrame(all_data)

        return df


    def process_funding_rate(df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.assign(
                datetime=lambda x: pd.to_datetime(
                    x["fundingTime"], unit="ms", utc=True
                ),
                fundingRate=lambda x: pd.to_numeric(x["fundingRate"]),
            )
            .sort_values("datetime")
            .set_index("datetime")
            .drop(columns=["symbol", "fundingTime", "markPrice"])
            .rename(columns={"fundingRate": "funding_rate"})
            .resample("D")
            .sum()  # 计算日总融资利率
        )


    def get_ohlcv(ticker: str) -> pd.DataFrame:
        return yf.download(
            tickers=ticker,
            ignore_tz=True,
            auto_adjust=True,
            progress=False,
            multi_level_index=False,
        )


    @mo.cache
    def get_all_data() -> pd.DataFrame:
        # 获取比特币历史价格
        btcusd = get_ohlcv("BTC-USD")

        # 获取融资利率
        start_date = dt.datetime(2019, 1, 1, tzinfo=dt.timezone.utc)
        end_date = dt.datetime(2025, 5, 30, tzinfo=dt.timezone.utc)
        rates = get_funding_rate("BTCUSDT", start_date, end_date)
        daily_rates = process_funding_rate(rates)
        daily_rates.index = daily_rates.index.tz_convert(None)

        return (
            pd.concat([daily_rates, btcusd["Close"]], join="outer", axis=1)
            .rename(columns={"Close": "btcusd"})
            .ffill()
            .dropna()
        )
    return (get_all_data,)


@app.cell
def _(get_all_data):
    data = get_all_data()
    data
    return (data,)


@app.cell
def _(FundingRate, data):
    metric = FundingRate(
        data,
        cumulative_days=30,
        rolling_period=200,
        upper_band_percentile=0.95,
        lower_band_percentile=0.05,
    )
    metric.generate_signals()
    fig = metric.generate_chart()

    fig
    return (metric,)


@app.cell
def _(metric):
    metric.signals
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
