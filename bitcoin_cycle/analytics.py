import marimo

__generated_with = "0.11.31"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from typing import List
    import datetime as dt

    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    import signals
    return List, dt, go, make_subplots, np, pd, signals


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
            .ffill()
            .dropna()
        )


    def color_signal(value: str) -> str:
        if value == "peak":
            color = "red"
        elif value == "valley":
            color = "green"
        else:
            color = ""
        return f"color: {color}"
    return color_signal, read_metrics


@app.cell
def _(List, mo, pd, read_metrics, signals):
    # 参数
    btcusd_filepath = "./data/btcusd.csv"

    # 指标配置，存储在字典中，指标名称 -> 数据文件路径，信号类
    metric_config = {
        "sth_realized_price": {
            "filepath": "./data/sth_realized_price.csv",
            "class": signals.STHRealizedPrice,
            "params": {
                "period": mo.ui.number(value=200),
                "threshold": mo.ui.number(value=2.0),
            },
        },
        "sth_sopr": {
            "filepath": "./data/sth_sopr.csv",
            "class": signals.STHSOPR,
            "params": {
                "bband_period": mo.ui.number(value=200),
                "bband_upper_std": mo.ui.number(value=2.0),
                "bband_lower_std": mo.ui.number(value=1.5),
            },
        },
        "sth_nupl": {
            "filepath": "./data/sth_nupl.csv",
            "class": signals.STHNUPL,
            "params": {
                "smooth_period": mo.ui.number(value=10),
                "fisher_period": mo.ui.number(value=200),
                "threshold": mo.ui.number(value=2.0),
            },
        },
        "sth_mvrv": {
            "filepath": "./data/sth_mvrv.csv",
            "class": signals.STHMVRV,
            "params": {
                "smooth_period": mo.ui.number(value=10),
                "fisher_period": mo.ui.number(value=200),
                "threshold": mo.ui.number(value=2.0),
            },
        },
        "nrpl": {
            "filepath": "./data/nrpl.csv",
            "class": signals.NRPL,
            "params": {
                "bband_period": mo.ui.number(value=200),
                "bband_upper_std": mo.ui.number(value=2.0),
                "bband_lower_std": mo.ui.number(value=2.0),
            },
        },
    }

    # 读取数据，计算指标信号
    all_metrics: List[signals.Metric] = []

    for name, config in metric_config.items():
        data = read_metrics(btcusd_filepath, config["filepath"])
        metric_cls = config["class"](data)
        metric_cls.generate_signals()
        all_metrics.append(metric_cls)


    signals_df = (
        pd.concat({m.name: m.signals["signal"] for m in all_metrics}, axis=1)
        .ffill()
        .dropna()
        .astype(int)
        .replace({0: "neutral", 1: "peak", -1: "valley"})
    )
    signals_df
    return (
        all_metrics,
        btcusd_filepath,
        config,
        data,
        metric_cls,
        metric_config,
        name,
        signals_df,
    )


@app.cell
def _(dt, mo):
    # 日期控件
    start_date_ui = mo.ui.date(
        label="开始日期",
        value=(dt.datetime.today() - dt.timedelta(days=10)).date(),
    )
    end_date_ui = mo.ui.date(label="结束日期", value=dt.datetime.today().date())
    return end_date_ui, start_date_ui


@app.cell
def _(color_signal, end_date_ui, mo, pd, signals_df, start_date_ui):
    # 读取UI控件的值
    start_date_val = pd.Timestamp(start_date_ui.value)
    end_date_val = pd.Timestamp(end_date_ui.value)

    # 更新数据展示
    dashboard = signals_df.loc[start_date_val:end_date_val].T
    dashboard.columns = [col.strftime("%m.%d") for col in dashboard.columns]
    styled_dashboard = dashboard.style.map(color_signal)

    # 展示控件和结果
    mo.vstack([start_date_ui, end_date_ui, mo.md(styled_dashboard.to_html())])
    return dashboard, end_date_val, start_date_val, styled_dashboard


@app.cell
def _(mo):
    mo.md(r"""## 时间序列""")
    return


@app.cell
def _():
    # # UI
    # metric_selection_ui = mo.ui.dropdown(
    #     list(metric_config.keys()), value="sth_realized_price"
    # )
    # metric_selection_ui
    return


@app.cell
def _():
    # metric_params_ui = []
    # metric_params = metric_config[metric_selection_ui.value]["params"]

    # for k, v in metric_params.items():
    #     metric_params_ui.append(mo.md(k))
    #     metric_params_ui.append(v)

    # mo.vstack(metric_params_ui)
    return


@app.cell
def _():
    # selected_params = {k: v.value for k, v in metric_params.items()}
    # print(selected_params)
    return


@app.cell
def _():
    # selected_metric_name = metric_selection_ui.value
    # print(selected_metric_name)

    # selected_metric_config = metric_config[selected_metric_name]
    # print(selected_metric_config)

    # selected_metric_data = read_metrics(
    #     btcusd_filepath, selected_metric_config["filepath"]
    # )
    # print(selected_metric_data.tail())

    # selected_metric = selected_metric_config["class"](
    #     selected_metric_data, **selected_params
    # )
    # selected_metric.generate_signals()
    # print(selected_metric.signals.tail())

    # fig = selected_metric.generate_chart()
    # fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
