import datetime as dt
import os

import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st

pio.templates.default = "ggplot2"

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


@st.cache_data
def read_yahoo(
    start_date: dt.datetime, end_date: dt.datetime, asset1: str, asset2: str
) -> pd.DataFrame:
    filepath1 = os.path.join(DATA_DIR, f"{asset1}.csv")
    filepath2 = os.path.join(DATA_DIR, f"{asset2}.csv")

    df1 = pd.read_csv(filepath1, index_col=0, parse_dates=True)
    df2 = pd.read_csv(filepath2, index_col=0, parse_dates=True)

    prices1 = df1[["Adj Close"]].rename(columns={"Adj Close": asset1})
    prices2 = df2[["Adj Close"]].rename(columns={"Adj Close": asset2})

    prices = pd.concat([prices1, prices2], axis=1)
    prices = prices.loc[start_date:end_date]
    prices = (
        prices.assign(weekday=lambda x: x.index.weekday)
        .query("weekday < 5")
        .drop(columns="weekday")
        .ffill()
    )

    return prices


# def plot_correlation_heatmap(corr, title: str, figsize=(12, 8)):
#     fig, ax = plt.subplots(figsize=figsize)
#     ax.grid(False)  # remove grid
#     ax = sns.heatmap(
#         corr, vmin=-1, vmax=1, annot=True, fmt=".1f", ax=ax, annot_kws={"size": 9}
#     )
#     ax.set(title=title, xlabel="", ylabel="")
#     return fig


# UI: 用户输入参数
st.title("🔗滚动相关系数")

with st.expander("输入参数", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        asset1 = st.selectbox(
            "🔍 选择第一个资产", ASSETS, index=ASSETS.index("Bitcoin")
        )
    with col2:
        asset2 = st.selectbox("🔍 选择第二个资产", ASSETS, index=ASSETS.index("SP500"))

    col3, col4, col5 = st.columns(3)
    with col3:
        start_date = st.date_input("📅 开始日期", value=dt.datetime(2020, 1, 1))
    with col4:
        end_date = st.date_input("📅 结束日期", value=dt.datetime.today())
    with col5:
        corr_period = st.number_input(
            "📊 滚动系数窗口(天)", min_value=10, max_value=365, value=90
        )


# 检查用户是否选择了两个不同的资产
if asset1 == asset2:
    st.warning("请选择两个不同的资产以计算滚动相关系数")
else:
    # 读取数据
    prices = read_yahoo(start_date, end_date, asset1, asset2)

    # 计算滚动相关系数
    rolling_corr = prices[asset1].rolling(window=corr_period).corr(prices[asset2])

    # 使用plotly绘制滚动相关系数的折线图
    fig = px.line(
        rolling_corr,
        title=f"{corr_period}-days rolling correlation between {asset1} and {asset2}",
        width=800,
        height=500,
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, theme=None)
