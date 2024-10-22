import datetime as dt
import os

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# UI参数
SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "FTM/USDT"]
TIMEFRAMES = ["4h", "1d"]
DATA_DIRECTORY = "~/quant-research/data/binance"

# 技术指标参数
FISHER_PERIOD = 10  # 菲舍尔转换的回溯期
BP_PERIOD = 20  # 带通滤波器的回溯期
BP_WIDTH = 0.3  # 带通滤波器的带宽
CGOSC_PERIOD = 8  # 超级震荡指标的回溯期


@st.cache_data
def read_ohlcv(
    symbol: str, timeframe: str, start_date: dt.datetime, end_date: dt.datetime
) -> pd.DataFrame:
    filepath = os.path.join(
        DATA_DIRECTORY,
        f"binance_{symbol.replace("/", "").lower()}_{timeframe}_ohlcv.csv",
    )
    df = pd.read_csv(filepath, index_col="timestamp", parse_dates=True)
    return df[(df.index >= start_date) & (df.index <= end_date)]


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


@st.cache_data
def calculate_indicators(
    ohlcv: pd.DataFrame,
    fisher_period: int,
    bp_period: int,
    bp_width: float,
    cgosc_period: int,
) -> pd.DataFrame:
    ohlcv["ft"] = fisher_transform(ohlcv["close"], period=fisher_period)
    ohlcv["bp"] = bandpass(ohlcv["close"], period=bp_period, bandwidth=bp_width)
    ohlcv["cgosc"] = stoch_center_gravity_osc(ohlcv["close"], period=cgosc_period)
    return ohlcv.dropna()


st.title("价格分析")

selected_symbol = st.selectbox("货币对", SYMBOLS)
selected_timeframe = st.selectbox("时间框架", TIMEFRAMES)
start_date = pd.to_datetime(
    st.date_input("开始日期", dt.date.today() - dt.timedelta(days=30))
)
end_date = pd.to_datetime(st.date_input("结束日期", dt.date.today()))

# 读取数据并显示
if start_date < end_date:
    df = read_ohlcv(selected_symbol, selected_timeframe, start_date, end_date)
    df = calculate_indicators(df, FISHER_PERIOD, BP_PERIOD, BP_WIDTH, CGOSC_PERIOD)
    # st.write("数据框:")
    # st.write(df)

    # 设置主题为 'plotly_dark'
    pio.templates.default = "plotly_dark"

    # 可视化
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02)

    # 收盘价
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["close"], mode="lines", name="close", showlegend=False
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(row=1, col=1, range=[np.min(df["close"]), np.max(df["close"])])

    # 使用for循环绘制所有与指标相关的子图
    indicators = [
        {
            "name": "Fisher Transform",
            "column": "ft",
            "upper_threshold": 2,
            "lower_threshold": -2,
        },
        {
            "name": "Bandpass",
            "column": "bp",
            "upper_threshold": 0.8,
            "lower_threshold": -0.8,
        },
        {
            "name": "Osc",
            "column": "cgosc",
            "upper_threshold": 0.8,
            "lower_threshold": -0.8,
        },
    ]

    for i, indicator in enumerate(indicators, start=2):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[indicator["column"]],
                mode="lines",
                name=indicator["name"],
                showlegend=False,
            ),
            row=i,
            col=1,
        )
        fig.update_yaxes(title_text=indicator["name"], row=i, col=1)
        fig.add_hrect(
            y0=indicator["upper_threshold"],
            y1=np.max(df[indicator["column"]]),
            line_width=0,
            fillcolor="red",
            opacity=0.2,
            row=i,
            col=1,
        )
        fig.add_hrect(
            y0=indicator["lower_threshold"],
            y1=np.min(df[indicator["column"]]),
            line_width=0,
            fillcolor="lime",
            opacity=0.2,
            row=i,
            col=1,
        )

    fig.update_layout(title=f"{selected_symbol}", height=1200, showlegend=False)
    st.plotly_chart(fig)
else:
    st.error("错误: 开始日期必须早于结束日期")
