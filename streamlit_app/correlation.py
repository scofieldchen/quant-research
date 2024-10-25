import datetime as dt
import os

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st

# 全局参数
DATA_DIR = "../data/yahoo"  # 数据文件夹
ASSETS = [  # 数据文件夹包含的所有资产，作为下框架的选项
    "Bitcoin",
    "CAC",
    "Copper",
    "Crude oil",
    "DAX",
    "ESTX50",
    "EURUSD",
    "Ethereum",
    "FTSE100",
    "GBPUSD",
    "Gold",
    "ICE US Dollar Index",
    "NASDAQ",
    "Natural gas",
    "Nikkei225",
    "Platinum",
    "RBOB gasoline",
    "SP500",
    "SSE Composite",
    "Silver",
    "US 10-Year Bond",
    "US 2-Year Bond",
    "USDJPY",
]


def read_yahoo_ohlcv(filepath: str, asset: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    df["asset"] = asset
    return df


@st.cache_data
def read_yahoo(start_date: dt.datetime, end_date: dt.datetime) -> pd.DataFrame:
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    assets = [f.split(".")[0] for f in files]
    filepaths = [os.path.join(DATA_DIR, f) for f in files]

    data = pd.concat(
        (read_yahoo_ohlcv(fp, a) for fp, a in zip(filepaths, assets)), axis=0
    )

    prices = (
        data.pivot(columns="asset", values="Adj Close")
        .loc[start_date:end_date]
        .assign(weekday=lambda x: x.index.weekday)
        .query("weekday < 5")  # remove weekends, 0=monday, 4=friday
        .drop(columns="weekday")
        .ffill()
    )

    return prices


def plot_correlation_heatmap(corr, title: str, figsize=(12, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)  # remove grid
    ax = sns.heatmap(
        corr, vmin=-1, vmax=1, annot=True, fmt=".1f", ax=ax, annot_kws={"size": 9}
    )
    ax.set(title=title, xlabel="", ylabel="")
    return fig


# UI
st.title("相关性分析")
with st.expander("输入参数", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        asset1 = st.selectbox("选择第一个资产", ASSETS, index=ASSETS.index("Bitcoin"))
    with col2:
        asset2 = st.selectbox("选择第二个资产", ASSETS, index=ASSETS.index("SP500"))

    col3, col4, col5 = st.columns(3)
    with col3:
        start_date = st.date_input("选择开始日期", value=dt.datetime(2020, 1, 1))
    with col4:
        end_date = st.date_input("选择结束日期", value=dt.datetime.today())
    with col5:
        corr_period = st.number_input(
            "滚动系数窗口(天)", min_value=10, max_value=365, value=90
        )

# 读取数据
prices = read_yahoo(start_date, end_date)
asset1_prices = prices[asset1]
asset2_prices = prices[asset2]

# 计算滚动相关系数
rolling_corr = asset1_prices.rolling(window=corr_period).corr(asset2_prices)

# 使用plotly绘制滚动相关系数的折线图
fig = px.line(
    rolling_corr, title=f"{asset1} 和 {asset2} 的 {corr_period}-day 滚动相关系数"
)
st.plotly_chart(fig)
