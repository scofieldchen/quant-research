import numpy as np
import pandas as pd


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


def bandpass(series: pd.Series, period: int = 10, bandwidth: float = 0.5) -> pd.Series:
    const = bandwidth * 2 * np.pi / period
    beta = np.cos(2 * np.pi / period)
    gamma = 1 / np.cos(const)
    alpha1 = gamma - np.sqrt(gamma**2 - 1)
    alpha2 = (np.cos(0.25 * const) + np.sin(0.25 * const) - 1) / np.cos(0.25 * const)
    alpha3 = (np.cos(1.5 * const) + np.sin(1.5 * const) - 1) / np.cos(1.5 * const)

    hp = np.zeros(len(series))
    bp = np.zeros(len(series))
    peaks = np.zeros(len(series))
    signals = np.zeros(len(series))

    for i in range(2, len(series)):
        hp[i] = (1 + alpha2 / 2) * (series.iloc[i] - series.iloc[i - 1]) + (
            1 - alpha2
        ) * hp[i - 1]
        bp[i] = (
            0.5 * (1 - alpha1) * (hp[i] - hp[i - 2])
            + beta * (1 + alpha1) * bp[i - 1]
            - alpha1 * bp[i - 2]
        )
        peaks[i] = 0.991 * peaks[i - 1]
        if abs(bp[i]) > peaks[i]:
            peaks[i] = abs(bp[i])
        if peaks[i] != 0:
            signals[i] = bp[i] / peaks[i]

    return pd.Series(signals, index=series.index)


def _center_gravity(series: pd.Series) -> float:
    nm = 0
    dm = 0
    reversed_series = series[::-1]
    for i, value in enumerate(reversed_series):
        nm += (i + 1) * value
        dm += value
    try:
        return -nm / dm + (len(series) + 1) / 2
    except ZeroDivisionError:
        return 0


def stoch_center_gravity_osc(series: pd.Series, period: int = 10) -> pd.Series:
    center_gravity = series.rolling(period, min_periods=1).apply(_center_gravity)
    max_cg = center_gravity.rolling(period, min_periods=1).max()
    min_cg = center_gravity.rolling(period, min_periods=1).min()
    cg_range = (max_cg - min_cg).replace({0: np.nan})
    stoch = ((center_gravity - min_cg) / cg_range).fillna(0)
    smooth_stoch = (
        4 * stoch + 3 * stoch.shift(1) + 2 * stoch.shift(2) + stoch.shift(3)
    ) / 10
    return 2 * (smooth_stoch - 0.5)
