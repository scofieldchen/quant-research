import pandas as pd

import signals
import plotly.offline as offline


def read_metrics(filepath_ohlcv: str, filepath_metric: str) -> pd.DataFrame:
    ohlcv = pd.read_csv(filepath_ohlcv, index_col="datetime", parse_dates=True)
    metric = pd.read_csv(filepath_metric, index_col="datetime", parse_dates=True)

    return (
        pd.concat([ohlcv["close"], metric], axis=1, join="outer")
        .rename(columns={"close": "price"})
        .dropna()
    )


filepath_ohlcv = "./data/btcusd.csv"
filepath_metric = "./data/sth_realized_price.csv"
df = read_metrics(filepath_ohlcv, filepath_metric)
print(df.head())
print(df.tail())

metric = signals.STHRealizedPrice(
    data=df["sth_realized_price"], btc_prices=df["price"], period=200
)
metric.generate_signals()
print(metric.signals)
fig = metric.generate_chart()
offline.plot(fig, filename="test_chart.html", auto_open=True)
