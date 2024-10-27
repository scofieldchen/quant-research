import datetime as dt
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

plt.style.use("ggplot")

DATA_DIR = "../data/yahoo"
ASSETS = [
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

default_assets = [
    "Bitcoin",
    "Ethereum",
    "Gold",
    "Crude oil",
    "SP500",
    "NASDAQ",
    "US 10-Year Bond",
    "ICE US Dollar Index",
    "EURUSD",
    "USDJPY",
]


@st.cache_data
def read_data(
    assets: list[str], start_date: dt.datetime, end_date: dt.datetime
) -> pd.DataFrame:
    data = pd.concat(
        (
            pd.read_csv(
                os.path.join(DATA_DIR, f"{asset}.csv"), index_col=0, parse_dates=True
            )[["Adj Close"]].rename(columns={"Adj Close": asset})
            for asset in assets
        ),
        axis=1,
    )
    data = (
        data.loc[start_date:end_date]
        .assign(weekday=lambda x: x.index.weekday)
        .query("weekday < 5")
        .drop(columns="weekday")
        .ffill()
    )
    return data


st.title("📊 相关系数")

# 输入参数
col1, _ = st.columns(2)
with col1:
    with st.expander("输入参数", expanded=True):
        selected_assets = st.multiselect("🔍 选择资产", ASSETS, default=default_assets)
        window_period = st.number_input(
            "⏳ 选择窗口期（天数）", min_value=10, max_value=1000, value=30, step=1
        )

# 读取数据
start_date = dt.datetime(2022, 1, 1)
end_date = dt.datetime.today()
df = read_data(selected_assets, start_date, end_date)

# 计算相关系数
corr = df.tail(window_period).corr()

# 绘制热力图
col1, _ = st.columns([0.7, 0.3])
with col1:
    st.write("相关系数热力图")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.grid(False)
    ax = sns.heatmap(
        corr, vmin=-1, vmax=1, annot=True, fmt=".2f", ax=ax, annot_kws={"size": 9}
    )
    ax.set(title=f"{window_period}-day correlation", xlabel="", ylabel="")
    st.pyplot(fig)
