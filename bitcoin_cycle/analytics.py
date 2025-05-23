import marimo

__generated_with = "0.13.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from typing import List
    from pathlib import Path
    import datetime as dt

    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    import signals
    from signals.utils import calculate_percentile_bands
    from signals.indicators import find_trend_periods
    return (
        List,
        Path,
        dt,
        find_trend_periods,
        go,
        make_subplots,
        np,
        pd,
        signals,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 横截面分析

    ---
    """
    )
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
        if value.lower() == "peak":
            color = "red"
        elif value.lower() == "valley":
            color = "green"
        else:
            color = ""
        return f"color: {color}"
    return color_signal, read_metrics


@app.cell
def _(List, Path, mo, np, pd, read_metrics, signals):
    # 数据文件夹
    data_dir = Path("/users/scofield/quant-research/bitcoin_cycle/data")

    # 比特币数据路径
    btcusd_filepath = data_dir / "btcusd.csv"

    # 指标配置
    metric_config = {
        "sth_realized_price": {
            "filepath": data_dir / "sth_realized_price.csv",
            "class": signals.STHRealizedPrice,
            "params": {
                "smooth_period": mo.ui.number(value=7),
                "rolling_period": mo.ui.number(value=200),
                "lower_band_percentile": mo.ui.number(value=0.01),
                "upper_band_percentile": mo.ui.number(value=0.99),
            },
        },
        "sth_sopr": {
            "filepath": data_dir / "sth_sopr.csv",
            "class": signals.STHSOPR,
            "params": {
                "smooth_period": mo.ui.number(value=7),
                "rolling_period": mo.ui.number(value=200),
                "lower_band_percentile": mo.ui.number(value=0.05),
                "upper_band_percentile": mo.ui.number(value=0.95),
            },
        },
        "sth_nupl": {
            "filepath": data_dir / "sth_nupl.csv",
            "class": signals.STHNUPL,
            "params": {
                "smooth_period": mo.ui.number(value=7),
                "rolling_period": mo.ui.number(value=200),
                "lower_band_percentile": mo.ui.number(value=0.05),
                "upper_band_percentile": mo.ui.number(value=0.95),
            },
        },
        "sth_mvrv": {
            "filepath": data_dir / "sth_mvrv.csv",
            "class": signals.STHMVRV,
            "params": {
                "smooth_period": mo.ui.number(value=7),
                "rolling_period": mo.ui.number(value=200),
                "lower_band_percentile": mo.ui.number(value=0.05),
                "upper_band_percentile": mo.ui.number(value=0.95),
            },
        },
        "nrpl": {
            "filepath": data_dir / "nrpl.csv",
            "class": signals.NRPL,
            "params": {
                "smooth_period": mo.ui.number(value=7),
                "rolling_period": mo.ui.number(value=200),
                "lower_band_percentile": mo.ui.number(value=0.05),
                "upper_band_percentile": mo.ui.number(value=0.95),
            },
        },
        "fear_greed_index": {
            "filepath": data_dir / "fear_greed_index.csv",
            "class": signals.FearGreedIndex,
            "params": {
                "smooth_period": mo.ui.number(value=10),
                "extreme_greed_threshold": mo.ui.number(value=80),
                "extreme_fear_threshold": mo.ui.number(value=20),
            },
        },
        "greed_days": {
            "filepath": data_dir / "fear_greed_index.csv",
            "class": signals.ConsecutiveGreedDays,
            "params": {
                "extreme_level": mo.ui.number(value=40),
            },
        },
        "toptrader_long_short_ratio_account": {
            "filepath": data_dir / "long_short_ratio.csv",
            "class": signals.LongShortRatioAccount,
            "params": {
                "smooth_period": mo.ui.number(value=10),
                "rolling_period": mo.ui.number(value=200),
                "lower_band_percentile": mo.ui.number(value=0.05),
                "upper_band_percentile": mo.ui.number(value=0.95),
            },
        },
        "toptrader_long_short_ratio_position": {
            "filepath": data_dir / "long_short_ratio.csv",
            "class": signals.LongShortRatioPosition,
            "params": {
                "smooth_period": mo.ui.number(value=10),
                "rolling_period": mo.ui.number(value=200),
                "lower_band_percentile": mo.ui.number(value=0.05),
                "upper_band_percentile": mo.ui.number(value=0.95),
            },
        },
        "long_short_ratio": {
            "filepath": data_dir / "long_short_ratio.csv",
            "class": signals.LongShortRatio,
            "params": {
                "smooth_period": mo.ui.number(value=10),
                "rolling_period": mo.ui.number(value=200),
                "lower_band_percentile": mo.ui.number(value=0.05),
                "upper_band_percentile": mo.ui.number(value=0.95),
            },
        },
        "funding_rate": {
            "filepath": data_dir / "funding_rate.csv",
            "class": signals.FundingRate,
            "params": {
                "cumulative_days": mo.ui.number(value=30),
                "rolling_period": mo.ui.number(value=200),
                "lower_band_percentile": mo.ui.number(value=0.05),
                "upper_band_percentile": mo.ui.number(value=0.95),
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
        .replace({np.nan: 0})
        .astype(int)
        .replace({0: "Neutral", 1: "Peak", -1: "Valley"})
    )
    # signals_df.tail(10)
    return all_metrics, btcusd_filepath, metric_config, signals_df


@app.cell
def _(dt, mo):
    start_date_ui = mo.ui.date(
        label="开始日期",
        value=(dt.datetime.today() - dt.timedelta(days=10)).date(),
    )
    end_date_ui = mo.ui.date(label="结束日期", value=dt.datetime.today().date())

    mo.hstack([start_date_ui, end_date_ui], justify="start")
    return end_date_ui, start_date_ui


@app.cell
def _(color_signal, end_date_ui, mo, pd, signals_df, start_date_ui):
    start_date_val = pd.Timestamp(start_date_ui.value)
    end_date_val = pd.Timestamp(end_date_ui.value)

    dashboard = signals_df.loc[start_date_val:end_date_val].T
    dashboard.columns = [col.strftime("%m.%d") for col in dashboard.columns]

    table_styles = [
        {
            "selector": "th",  # 表头单元格
            "props": [
                ("background-color", "#F8F9FA"),  # 淡灰色背景
                ("color", "#212529"),  # 深灰色字体
                ("font-weight", "bold"),  # 字体加粗
                ("text-align", "center"),  # 文本居中
                ("padding", "10px 8px"),  # 内边距 (上下10px, 左右8px)
                ("border-bottom", "2px solid #DEE2E6"),  # 底部边框
            ],
        },
        {
            "selector": "td",  # 数据单元格
            "props": [
                ("text-align", "center"),  # 文本居中
                ("padding", "8px"),  # 内边距
                ("border", "1px solid #E9ECEF"),  # 细边框
            ],
        },
        {
            "selector": "tr:nth-child(even)",  # 偶数行
            "props": [
                ("background-color", "#F8F9FA")  # 淡灰色背景 (斑马纹)
            ],
        },
        {
            "selector": "tr:hover",  # 鼠标悬停在行上
            "props": [
                ("background-color", "#E9ECEF")  # 悬停时背景色变深
            ],
        },
        {
            "selector": "table",  # 整个表格
            "props": [
                ("border-collapse", "collapse"),  # 合并边框
                ("width", "100%"),  # 宽度100%
                ("font-family", '"Segoe UI", Arial, sans-serif'),  # 现代字体
                ("box-shadow", "0 2px 4px rgba(0,0,0,0.1)"),  # 轻微阴影效果
            ],
        },
    ]

    # 应用颜色信号和自定义表格样式
    styled_dashboard = dashboard.style.map(color_signal).set_table_styles(
        table_styles
    )

    mo.md(styled_dashboard.to_html())
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 综合信号

    ---
    """
    )
    return


@app.cell
def _(
    all_metrics,
    btcusd_filepath,
    find_trend_periods,
    go,
    make_subplots,
    np,
    pd,
):
    def calculate_composite_signal(df: pd.DataFrame) -> pd.Series:
        """计算综合性信号"""
        # 计算可用指标的信号总和
        sum_of_signals = df_signals.sum(axis=1, skipna=True)

        # 计算当天可用指标的数量
        count_of_available_signals = df_signals.notna().sum(axis=1)

        # 计算平均信号
        composite_signal = sum_of_signals.divide(count_of_available_signals)

        # 如果某天没有任何可用指标，结果会是 NaN，用0填充
        composite_signal = composite_signal.fillna(0)

        return composite_signal


    def plot_composite_signal(
        data: pd.DataFrame,
        peak_threshold: float = 0.7,
        valley_threshold: float = -0.5,
    ) -> go.Figure:
        """显示综合信号"""
        # 创建图表对象
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
        )

        # 比特币价格
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["btcusd"],
                name="BTC/USD",
                line=dict(color="#F7931A", width=2),
                hovertemplate="<b>Date</b>: %{x|%Y-%m-%d}<br>"
                + "<b>Price</b>: $%{y:,.0f}<br><extra></extra>",
            ),
            row=1,
            col=1,
        )

        # 添加背景颜色显示潜在的顶部和底部
        peak_periods = find_trend_periods(
            data["composite_signal"] >= peak_threshold
        )
        valley_periods = find_trend_periods(
            data["composite_signal"] <= valley_threshold
        )

        for x0, x1 in peak_periods:
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor="#FF6B6B",
                opacity=0.2,
                line_width=0,
                row=1,
                col=1,
            )

        for x0, x1 in valley_periods:
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor="#38A169",
                opacity=0.2,
                line_width=0,
                row=1,
                col=1,
            )

        # 综合信号
        # 信号高于0用绿色填充，信号低于0用红色填充
        # 边界线使用深蓝色
        x_vals = data.index
        y_signal = data["composite_signal"]

        y_positive = np.where(y_signal >= 0, y_signal, 0)
        y_negative = np.where(y_signal < 0, y_signal, 0)

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_positive,
                fill="tozeroy",
                fillcolor="rgba(0, 128, 0, 0.3)",
                line=dict(width=0),
                name="Positive Signal Area",
                hoverinfo="skip",
                legendgroup="signal",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_negative,
                fill="tozeroy",
                fillcolor="rgba(255, 0, 0, 0.3)",
                line=dict(width=0),
                name="Negative Signal Area",
                hoverinfo="skip",
                legendgroup="signal",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_signal,
                name="Composite Signal",
                line=dict(color="royalblue", width=1.5),
                hovertemplate="<b>Date</b>: %{x|%Y-%m-%d}<br>"
                + "<b>Signal</b>: %{y:.4f}<br><extra></extra>",
                legendgroup="signal",
            ),
            row=2,
            col=1,
        )

        # 添加表示综合信号极端水平的信号线
        fig.add_hline(
            0,
            row=2,
            col=1,
            line_dash="solid",
            line_color="darkgrey",
            line_width=1,
        )

        for level in [valley_threshold, peak_threshold]:
            fig.add_hline(
                level,
                row=2,
                col=1,
                line_dash="dot",
                line_color="grey",
                line_width=1,
                annotation_text=f"{level}",
                annotation_position="bottom right" if level < 0 else "top right",
                annotation_font_size=10,
            )

        # 更新图表布局
        fig.update_layout(
            title=dict(
                text=f"<b>Bitcoin Cycle Model</b>",
                x=0.5,
                y=0.95,
                font=dict(size=20),
            ),
            width=1000,
            height=750,
            template="plotly_white",
            showlegend=True,
            legend=dict(
                orientation="h",  # 图例水平排列
                yanchor="bottom",  # 图例垂直对其方式
                y=1.02,  # 将图例放置在图表上方
                x=0.5,  # 将图例放置在图表中间
                xanchor="center",  # 图例水平对其方式
            ),
            hovermode="x unified",
        )

        # 调整y轴样式
        fig.update_yaxes(
            row=1,
            col=1,
            title_text="<b>Price (USD)</b>",
            type="log",
            gridcolor="#E0E0E0",
            zerolinecolor="#C0C0C0",
            zerolinewidth=1,
            title_font=dict(size=14, family="Arial, sans-serif"),
            tickfont=dict(size=11),
        )

        fig.update_yaxes(
            row=2,
            col=1,
            title_text="<b>Composite Signal Value</b>",
            gridcolor="#E0E0E0",
            zerolinecolor="#C0C0C0",
            zerolinewidth=1,
            range=[
                -1.1,
                1.1,
            ],
            title_font=dict(size=14, family="Arial, sans-serif"),
            tickfont=dict(size=11),
            tickformat=".2f",
        )

        # 更新x轴样式
        fig.update_xaxes(
            gridcolor="#E0E0E0",
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikedash="dash",
            spikecolor="grey",
            spikethickness=1,
            title_text="<b>Date</b>",
            title_font=dict(size=14, family="Arial, sans-serif"),
            tickfont=dict(size=11),
            row=2,
            col=1,
        )

        return fig


    # 计算综合评分
    df_signals = pd.concat(
        {m.name: m.signals["signal"] for m in all_metrics}, axis=1
    )
    composite_signals = calculate_composite_signal(df_signals)

    # 获取比特币历史价格
    btcusd = pd.read_csv(btcusd_filepath, index_col="datetime", parse_dates=True)

    # 合并数据
    composite_signals_df = pd.concat(
        {"composite_signal": composite_signals, "btcusd": btcusd["close"]},
        axis=1,
        join="outer",
    )
    # composite_signals_df
    return composite_signals_df, plot_composite_signal


@app.cell
def _(mo):
    peak_threshold_ui = mo.ui.number(
        start=0, stop=1, step=0.01, value=0.5, label="顶部阈值"
    )
    valley_threshold_ui = mo.ui.number(
        start=-1, stop=0, step=0.05, value=-0.5, label="底部阈值"
    )

    mo.hstack([peak_threshold_ui, valley_threshold_ui], justify="start")
    return peak_threshold_ui, valley_threshold_ui


@app.cell
def _(
    composite_signals_df,
    peak_threshold_ui,
    plot_composite_signal,
    valley_threshold_ui,
):
    fig_composite_signal = plot_composite_signal(
        composite_signals_df, peak_threshold_ui.value, valley_threshold_ui.value
    )
    fig_composite_signal
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 时间序列分析

    ---
    """
    )
    return


@app.cell
def _(metric_config, mo):
    # 从下拉框中选择指标
    metric_ids = list(metric_config.keys())
    metric_dropdown_ui = mo.ui.dropdown(metric_ids, value="sth_realized_price")
    return (metric_dropdown_ui,)


@app.cell
def _(metric_dropdown_ui, mo):
    mo.md(f"""
    选择指标：{metric_dropdown_ui}
    """)

    # metric_dropdown_ui
    return


@app.cell
def _(metric_config, metric_dropdown_ui):
    # 动态渲染指标参数ui
    selected_metric_config = metric_config[metric_dropdown_ui.value]
    selected_metric_params = selected_metric_config["params"]
    selected_metric_params
    return selected_metric_config, selected_metric_params


@app.cell
def _(mo):
    btn = mo.ui.run_button(label="更新图表")
    btn
    return (btn,)


@app.cell
def _(
    btcusd_filepath,
    btn,
    read_metrics,
    selected_metric_config,
    selected_metric_params,
):
    chart = None

    if btn.value:
        args = {k: v.value for k, v in selected_metric_params.items()}
        selected_data = read_metrics(
            btcusd_filepath, selected_metric_config["filepath"]
        )
        selected_metric_ins = selected_metric_config["class"](
            selected_data, **args
        )
        selected_metric_ins.generate_signals()
        chart = selected_metric_ins.generate_chart()

    chart
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
