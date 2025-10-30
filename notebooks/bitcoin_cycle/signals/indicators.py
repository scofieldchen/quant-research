import datetime as dt
from typing import List, Tuple

import pandas as pd
import numpy as np


def calculate_mmi(data: pd.Series, period: int) -> pd.Series:
    # 参数验证
    if not isinstance(data, pd.Series):
        raise TypeError("data must be a pandas Series")
    if not isinstance(period, int):
        raise TypeError("period must be an integer")
    if period < 2:
        raise ValueError("period must be greater than 1")
    if len(data) < period:
        raise ValueError("data length must be >= period")

    def _mmi(data: np.ndarray) -> float:
        # 反转数据，索引0表示最新数据
        series = data[::-1]

        # 计算中位数
        median = np.median(series)

        nh = nl = 0
        for i in range(1, len(series)):
            if series[i] > median and series[i] > series[i - 1]:
                nl += 1
            elif series[i] < median and series[i] < series[i - 1]:
                nh += 1

        # 计算MMI
        return 100.0 * (nl + nh) / (len(series) - 1)

    return data.rolling(window=period, min_periods=period).apply(_mmi, raw=True)


def calculate_fractal_dimension(data: pd.Series, period: int) -> pd.Series:
    if not isinstance(data, pd.Series):
        raise TypeError("data must be a pandas Series")
    if not isinstance(period, int):
        raise TypeError("period must be an integer")
    if period < 2:
        raise ValueError("period must be greater than 1")
    if len(data) < period:
        raise ValueError("data length must be >= period")

    def _fractal_dimension(data: np.ndarray) -> float:
        # 反转数据，索引0表示最新数据
        series = data[::-1]

        # 计算半周期
        period = len(data)
        period2 = period // 2

        n1 = (max(series[0:period2]) - min(series[0:period2])) / period2
        n2 = (max(series[period2:period]) - min(series[period2:period])) / period2
        n3 = (max(series[0:period]) - min(series[0:period])) / period

        if n1 + n2 <= 0 or n3 <= 0:
            return 1.0

        return (np.log(n1 + n2) - np.log(n3)) / np.log(2)

    # 确保period是偶数
    # 如果period是奇数，将其减1，如果period是偶数，保持不变
    period = period & ~1

    return data.rolling(window=period, min_periods=period).apply(
        _fractal_dimension, raw=True
    )


def super_smoother_two_pole(x: pd.Series, period: int = 10) -> pd.Series:
    a = np.exp(-1.414 * np.pi / period)
    b = 2 * a * np.cos(1.414 * np.pi / period)
    c2 = b
    c3 = -a * a
    c1 = 1 - c2 - c3

    out = np.zeros(len(x))
    out[0] = x[0]
    out[1] = x[1]

    for i in range(2, len(x)):
        out[i] = c1 * (x[i] + x[i - 1]) / 2 + c2 * out[i - 1] + c3 * out[i - 2]

    return pd.Series(out, index=x.index)


def super_smoother_three_pole(x: pd.Series, period: int = 10) -> pd.Series:
    a = np.exp(-np.pi / period)
    b = 2 * a * np.cos(1.738 * np.pi / period)
    c = a * a
    d2 = b + c
    d3 = -(c + b * c)
    d4 = c * c
    d1 = 1 - d2 - d3 - d4

    out = np.zeros(len(x))
    out[0] = x[0]
    out[1] = x[1]
    out[2] = x[2]

    for i in range(3, len(x)):
        out[i] = (
            d1 * (x[i] + x[i - 1]) / 2
            + d2 * out[i - 1]
            + d3 * out[i - 2]
            + d4 * out[i - 3]
        )

    return pd.Series(out, index=x.index)


def super_smoother(
    x: pd.Series, period: int = 10, method: str = "two_pole"
) -> pd.Series:
    if method == "two_pole":
        return super_smoother_two_pole(x, period)
    elif method == "three_pole":
        return super_smoother_three_pole(x, period)
    else:
        raise ValueError("Invalid arg method")


def lowpass_filter(x: pd.Series, period: int = 10) -> pd.Series:
    a = 2.0 / (1 + period)

    out = np.zeros(len(x))
    out[0] = x.iloc[0]
    out[1] = x.iloc[1]

    for i in range(2, len(x)):
        out[i] = (
            (a - 0.25 * a * a) * x.iloc[i]
            + 0.5 * a * a * x.iloc[i - 1]
            - (a - 0.75 * a * a) * x.iloc[i - 2]
            + (2.0 - 2.0 * a) * out[i - 1]
            - (1.0 - a) * (1.0 - a) * out[i - 2]
        )

    return pd.Series(out, index=x.index)


def peak(series: pd.Series) -> pd.Series:
    length = len(series)

    if length < 3:
        return pd.Series([], index=[])

    peak_mask = np.zeros(length, dtype=int)

    for i in range(1, length - 1):
        if series.iloc[i - 1] < series.iloc[i] and series.iloc[i + 1] < series.iloc[i]:
            peak_mask[i] = 1

    return pd.Series(peak_mask, index=series.index)


def valley(series: pd.Series) -> pd.Series:
    length = len(series)

    if length < 3:
        return pd.Series([], index=[])

    valley_mask = np.zeros(length, dtype=int)

    for i in range(1, length - 1):
        if series.iloc[i - 1] > series.iloc[i] and series.iloc[i + 1] > series.iloc[i]:
            valley_mask[i] = 1

    return pd.Series(valley_mask, index=series.index)


def fisher_transform(series: pd.Series, period: int = 10) -> pd.Series:
    highest = series.rolling(period, min_periods=1).max()
    lowest = series.rolling(period, min_periods=1).min()
    values = np.zeros(len(series))
    fishers = np.zeros(len(series))

    for i in range(1, len(series)):
        values[i] = (
            0.66
            * (
                (series.iloc[i] - lowest.iloc[i]) / (highest.iloc[i] - lowest.iloc[i])
                - 0.5
            )
            + 0.67 * values[i - 1]
        )
        values[i] = max(min(values[i], 0.999), -0.999)
        fishers[i] = (
            0.5 * np.log((1 + values[i]) / (1 - values[i])) + 0.5 * fishers[i - 1]
        )

    return pd.Series(fishers, index=series.index)


def find_trend_periods(series: pd.Series) -> List[Tuple[dt.datetime, dt.datetime]]:
    periods = []
    start = None

    for i in range(len(series)):
        if series.iloc[i] == 1 and start is None:
            start = series.index[i]
        elif series.iloc[i] == 0 and start is not None:
            end = series.index[i - 1]
            periods.append((start, end))
            start = None

    if start is not None:
        end = series.index[-1]
        periods.append((start, end))

    return periods
