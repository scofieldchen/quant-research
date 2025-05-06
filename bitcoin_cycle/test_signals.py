import pandas as pd

import signals
import plotly.offline as offline
from rich import print


def read_metrics(filepath_ohlcv: str, filepath_metric: str) -> pd.DataFrame:
    ohlcv = pd.read_csv(filepath_ohlcv, index_col="datetime", parse_dates=True)
    metric = pd.read_csv(filepath_metric, index_col="datetime", parse_dates=True)

    return (
        pd.concat([ohlcv["close"], metric], axis=1, join="outer")
        .rename(columns={"close": "btcusd"})
        .dropna()
    )


# 读取数据
filepath_ohlcv = "./data/btcusd.csv"
filepath_metric = "./data/sth_sopr.csv"
df = read_metrics(filepath_ohlcv, filepath_metric)
print(df.head())
print(df.tail())

# 计算信号
# metric = signals.STHRealizedPrice(
#     df, price_col="btcusd", sth_rp_col="sth_realized_price", period=200, threshold=2
# )
metric = signals.STHSOPR(
    df,
    price_col="btcusd",
    sopr_col="sth_sopr",
    bband_period=200,
    bband_upper_std=2,
    bband_lower_std=1.5,
)
metric.generate_signals()
print(metric.signals)
fig = metric.generate_chart()
offline.plot(fig, filename="test_chart.html", auto_open=True)
