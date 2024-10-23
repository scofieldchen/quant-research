import os
import datetime as dt

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# UI参数
DATA_DIR = "../data/yahoo"


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


st.title("相关性分析")
start_date = st.date_input("选择开始日期", dt.datetime(2023, 1, 1))
end_date = st.date_input("选择结束日期", dt.datetime.today())

# 读取数据并显示最后5行
prices = read_yahoo(start_date, end_date)
# st.write("数据框的最后5行：")
# st.dataframe(prices.tail())

# 用户输入计算相关系数的天数
corr_period = st.number_input("相关系数窗口(天)", min_value=10, max_value=365, value=90)

corr = prices.tail(corr_period).corr()

fig, ax = plt.subplots(figsize=(12, 8))
ax.grid(False)  # remove grid
ax = sns.heatmap(
    corr, vmin=-1, vmax=1, annot=True, fmt=".1f", ax=ax, annot_kws={"size": 9}
)
_ = ax.set(title=f"{corr_period}-day correlation", xlabel="", ylabel="")

# 在Streamlit前端显示热力图
st.pyplot(fig)
