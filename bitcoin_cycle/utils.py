import datetime as dt
from typing import List, Tuple

import numpy as np
import pandas as pd


def fisher_transform(series: pd.Series, period: int = 10) -> pd.Series:
    """实现费舍尔转换"""
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
    """找到连续的1的开始时间和结束时间"""
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
