import marimo

__generated_with = "0.13.10"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import sys

    sys.path.insert(0, "/users/scofield/quant-research/bitcoin_cycle/")

    import talib
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import requests
    import yfinance as yf
    from plotly.subplots import make_subplots

    from signals import Metric
    from signals.indicators import lowpass_filter, fisher_transform

    yf.set_config(
        proxy={"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}
    )
    return pd, yf


@app.cell
def _(mo, pd, yf):
    def get_ohlcv(ticker: str) -> pd.DataFrame:
        return yf.download(
            tickers=ticker,
            ignore_tz=True,
            auto_adjust=True,
            progress=False,
            multi_level_index=False,
        )


    def get_metric(filepath: str) -> pd.DataFrame:
        return pd.read_csv(filepath, index_col="datetime", parse_dates=True)


    @mo.cache
    def get_all_data(ticker: str, filepath: str) -> pd.DataFrame:
        btcusd = get_ohlcv(ticker)
        metric = get_metric(filepath)
        return (
            pd.concat([metric, btcusd["Close"]], join="outer", axis=1)
            .rename(columns={"Close": "btcusd"})
            .ffill()
            .dropna()
        )
    return (get_all_data,)


@app.cell
def _(get_all_data):
    data = get_all_data(
        ticker="BTC-USD",
        filepath="/users/scofield/quant-research/bitcoin_cycle/data/nrpl.csv",
    )
    data
    return (data,)


@app.cell
def _():
    from signals import NRPL
    return (NRPL,)


@app.cell
def _(NRPL, data):
    metric = NRPL(data, upper_band_percentile=0.95, lower_band_percentile=0.01)
    metric.generate_signals()
    fig = metric.generate_chart()

    fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
