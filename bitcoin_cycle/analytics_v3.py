import marimo

__generated_with = "0.11.31"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    import signals
    return go, make_subplots, np, pd, signals


@app.cell
def _(mo):
    mo.md(r"""## 综合分析""")
    return


@app.cell
def _(pd):
    def read_metrics(filepath_ohlcv: str, filepath_metric: str) -> pd.DataFrame:
        ohlcv = pd.read_csv(filepath_ohlcv, index_col="datetime", parse_dates=True)
        metric = pd.read_csv(
            filepath_metric, index_col="datetime", parse_dates=True
        )

        return (
            pd.concat([ohlcv["close"], metric], axis=1, join="outer")
            .rename(columns={"close": "btcusd"})
            .dropna()
        )
    return (read_metrics,)


@app.cell
def _(read_metrics):
    ## sth realized price
    filepath_ohlcv = "./data/btcusd.csv"
    filepath_metric = "./data/sth_realized_price.csv"
    df = read_metrics(filepath_ohlcv, filepath_metric)
    df
    return df, filepath_metric, filepath_ohlcv


@app.cell
def _(df, signals):
    metric = signals.STHRealizedPrice(
        df,
        price_col="btcusd",
        sth_rp_col="sth_realized_price",
        period=200,
        threshold=2,
    )
    metric.generate_signals()
    return (metric,)


@app.cell
def _(metric):
    metric.signals
    return


@app.cell
def _(mo):
    mo.md(r"""## 时间序列""")
    return


@app.cell
def _(mo):
    # UI
    period_ui = mo.ui.slider(
        start=100, stop=500, value=200, step=10, label="Lookback period"
    )
    threshold_ui = mo.ui.slider(
        start=1.0, stop=3.0, value=2.0, step=0.1, label="Threshold"
    )
    return period_ui, threshold_ui


@app.cell
def _(df, metric, mo, period_ui, signals, threshold_ui):
    # 数据可视化
    selected_metric = signals.STHRealizedPrice(
        df,
        price_col="btcusd",
        sth_rp_col="sth_realized_price",
        period=period_ui.value,
        threshold=threshold_ui.value,
    )
    selected_metric.generate_signals()
    fig = metric.generate_chart()

    # 显示结果
    mo.vstack([period_ui, threshold_ui, fig])
    return fig, selected_metric


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
