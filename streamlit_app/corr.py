import datetime as dt
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

plt.style.use("ggplot")


# 全局参数
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

st.title("相关系数")

# 添加多选框，允许用户选择多个资产
selected_assets = st.multiselect(
    "选择资产",
    ASSETS,
    default=[
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
    ],
)

# 添加数字输入框，允许用户选择计算相关系数的窗口期
window_period = st.number_input(
    "选择窗口期（天数）", min_value=10, max_value=1000, value=30, step=1
)

# 根据用户输入读取数据
start_date = dt.datetime(2022, 1, 1)
end_date = dt.datetime.today()


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


# 读取数据
df = read_data(selected_assets, start_date, end_date)
# print(df.info())
# print(df.tail())

# 计算相关系数
corr = df.tail(window_period).corr()

# 绘制热力图显示相关系数
fig, ax = plt.subplots(figsize=(12, 8))
ax.grid(False)  # remove grid
ax = sns.heatmap(
    corr, vmin=-1, vmax=1, annot=True, fmt=".1f", ax=ax, annot_kws={"size": 9}
)
ax.set(title=f"{window_period}-day correlation", xlabel="", ylabel="")
st.pyplot(fig)
