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
        .ffill()
        .dropna()
    )


# 读取数据
filepath_ohlcv = "./data/btcusd.csv"
filepath_metric = "./data/nrpl.csv"
df = read_metrics(filepath_ohlcv, filepath_metric)
print(df.head())
print(df.tail())

# 计算信号
# metric = signals.STHRealizedPrice(df)
# metric = signals.STHSOPR(df)
# metric = signals.STHNUPL(df)
# metric = signals.STHMVRV(df)
metric = signals.NRPL(df)
metric.generate_signals()
print(metric.signals)
fig = metric.generate_chart()
offline.plot(fig, filename="test_chart.html", auto_open=True)
