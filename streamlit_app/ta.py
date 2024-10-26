import datetime as dt
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from plotly.subplots import make_subplots

from indicators import bandpass, fisher_transform, stoch_center_gravity_osc

pio.templates.default = "ggplot2"

# 使用自定义CSS
st.markdown(
    """
    <style>
    .stButton button {
        background-color: #4CAF50; /* 绿色按钮 */
        color: white;
        border-radius: 5px;
    }
    .stTextInput input {
        border-radius: 5px;
    }
    .stSelectbox select {
        border-radius: 5px;
    }
    .stDateInput input {
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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


def create_chart(df, selected_symbol):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02)

    # 收盘价
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["close"], mode="lines", name="close", showlegend=False
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(
        row=1,
        col=1,
        range=[np.min(df["close"]), np.max(df["close"])],
        title_text="Close",
    )

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
            "name": "CenterGravityOsc",
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

    fig.update_layout(
        title=f"Cycle analysis of {selected_symbol}",
        width=1200,
        height=800,
        showlegend=False,
    )
    return fig


st.title("📈 技术指标")

with st.expander("输入参数", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        selected_symbol = st.selectbox("💱 货币对", SYMBOLS)
        start_date = pd.to_datetime(
            st.date_input("📅 开始日期", dt.date.today() - dt.timedelta(days=30))
        )
    with col2:
        selected_timeframe = st.selectbox("⏳ 时间框架", TIMEFRAMES)
        end_date = pd.to_datetime(st.date_input("📅 结束日期", dt.date.today()))

if start_date < end_date:
    df = read_ohlcv(selected_symbol, selected_timeframe, start_date, end_date)
    df = calculate_indicators(df, FISHER_PERIOD, BP_PERIOD, BP_WIDTH, CGOSC_PERIOD)
    fig = create_chart(df, selected_symbol)
    st.plotly_chart(fig, theme=None)
else:
    st.error("⚠️ 开始日期必须早于结束日期")
