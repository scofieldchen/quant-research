import pandas as pd
import numpy as np


def calculate_mmi(data: pd.Series, period: int) -> pd.Series:
    """
    Calculate the Market Meanness Index (MMI) for asset prices.

    Args:
        data (pd.Series): The price series.
        period (int): The lookback period of the indicator, must be greater than 1.

    Returns:
        pd.Series: The Market Meanness Index (MMI) values.
    """
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
    """
    Calculate the Fractal Dimension for asset prices.

    Args:
        data (pd.Series): The price series.
        period (int): The lookback period of the indicator, must be greater than 1
            and will be converted to even number if odd.

    Returns:
        pd.Series: The Fractal Dimension values.
    """
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
