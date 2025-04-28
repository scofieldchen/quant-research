import pandas as pd

import signals
import plotly.offline as offline


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
filepath_metric = "./data/sth_realized_price.csv"
df = read_metrics(filepath_ohlcv, filepath_metric)
# print(df.head())
# print(df.tail())

# 计算信号
metric = signals.STHRealizedPrice(
    df, price_col="btcusd", sth_rp_col="sth_realized_price", period=200, threshold=2
)
metric.generate_signals()
print(metric.signals)
fig = metric.generate_chart()
offline.plot(fig, filename="test_chart.html", auto_open=True)
