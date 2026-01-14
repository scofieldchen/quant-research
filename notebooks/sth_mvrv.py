import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    from datetime import datetime, date, timedelta
    from typing import Tuple, List

    import duckdb
    import ffn
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio

    pio.templates.default = "simple_white"
    return List, Tuple, date, datetime, duckdb, go, make_subplots, mo, np, pd


@app.cell
def _(mo):
    mo.md("""
    # STH-MVRV 动量系统 🚀
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 选择参数

    ---
    """)
    return


@app.cell
def _(date, mo):
    # Create UI controls for parameters
    today = date.today()
    default_start = date(2024, 1, 1)

    parameter_form = mo.md("""
        {zscore_window}

        {start_date}

        {end_date}
        """).batch(
        zscore_window=mo.ui.number(
            start=10, stop=200, step=1, value=50, label="标准分数窗口"
        ),
        start_date=mo.ui.date(value=default_start, label="开始日期"),
        end_date=mo.ui.date(value=today, label="结束日期"),
    )

    parameter_form
    return (parameter_form,)


@app.cell
def _(mo, parameter_form):
    mo.stop(not parameter_form.value)

    zscore_window = parameter_form.value["zscore_window"]
    start_date = parameter_form.value["start_date"]
    end_date = parameter_form.value["end_date"]

    if start_date >= end_date:
        print("⚠️ 开始日期必需小于结束日期")
        mo.stop(True)
    return end_date, start_date, zscore_window


@app.cell
def _(duckdb, end_date, pd, start_date, zscore_window):
    def load_sth_mvrv_data(
        start_date: str, end_date: str, lookback_days: int
    ) -> pd.DataFrame:
        data_start = pd.to_datetime(start_date) - pd.Timedelta(
            days=lookback_days + 10
        )
        data_start_str = data_start.strftime("%Y-%m-%d")

        file_path = "/users/scofield/quant-research/data/cleaned/sth_mvrv.parquet"

        sql_query = f"""
        SELECT datetime, sth_mvrv, btcusd 
        FROM '{file_path}'
        WHERE datetime >= '{data_start_str}' 
        AND datetime <= '{end_date}'
        ORDER BY datetime
        """

        df = duckdb.sql(sql_query).df()
        return df


    raw_df = load_sth_mvrv_data(start_date, end_date, zscore_window)
    # raw_df
    return (raw_df,)


@app.cell
def _(mo):
    mo.md("""
    ## STH-MVRV 指标分析

    ---
    """)
    return


@app.cell
def _(np, pd, raw_df, start_date, zscore_window):
    def calculate_sth_mvrv_zscore(df: pd.DataFrame, window: int) -> pd.DataFrame:
        return (
            df.copy()
            .assign(log_sth_mvrv=lambda x: np.log(x["sth_mvrv"]))
            .assign(
                rolling_mean=lambda x: x["log_sth_mvrv"]
                .rolling(window=window)
                .mean()
            )
            .assign(
                rolling_std=lambda x: x["log_sth_mvrv"]
                .rolling(window=window)
                .std()
            )
            .assign(
                sth_mvrv_zscore=lambda x: (x["log_sth_mvrv"] - x["rolling_mean"])
                / x["rolling_std"]
            )
            .dropna()
        )


    # 计算标准分数
    df_with_zscore = calculate_sth_mvrv_zscore(raw_df, zscore_window)

    # 筛选数据
    analysis_df = df_with_zscore[
        df_with_zscore["datetime"] >= pd.to_datetime(start_date)
    ].copy()

    # analysis_df
    return (analysis_df,)


@app.cell
def _(List, Tuple, datetime, pd):
    def find_trend_periods(
        series: pd.Series,
    ) -> List[Tuple[datetime, datetime]]:
        """找到连续的1对应的开始日期和结束日期

        Args:
            series: 时间序列，取值为1或者0，索引为时间戳
        """
        periods = []
        start = None

        for i in range(len(series)):
            if series.iloc[i] == 1 and start is None:
                start = series.index[i]
            elif series.iloc[i] == 0 and start is not None:
                end = series.index[i - 1]
                periods.append((start, end))
                start = None

        if start is not None:
            end = series.index[-1]
            periods.append((start, end))

        return periods


    # bullish_regime = analysis_df["sth_mvrv_zscore"] > 0
    # bullish_regime.index = analysis_df["datetime"]
    # bullish_regime

    # find_trend_periods(bullish_regime)
    return (find_trend_periods,)


@app.cell
def _(analysis_df, find_trend_periods, go, make_subplots, pd):
    def create_indicator_chart(df: pd.DataFrame) -> go.Figure:
        df_plot = df.copy()

        # 创建子图
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=(
                "Bitcoin Price & Market Regimes",
                "STH-MVRV Ratio",
                "STH-MVRV Z-Score",
            ),
        )

        # 行1: 比特币价格
        fig.add_trace(
            go.Scatter(
                x=df_plot["datetime"],
                y=df_plot["btcusd"],
                line=dict(color="#F7931A", width=2.5),
                hovertemplate="<b>%{x}</b><br>BTCUSD: %{y:,.0f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # 行2: STH-MVRV 比率
        fig.add_trace(
            go.Scatter(
                x=df_plot["datetime"],
                y=df_plot["sth_mvrv"],
                line=dict(color="#2E86AB", width=2),
                hovertemplate="<b>%{x}</b><br>STH-MVRV: %{y:.1f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        # 添加基准线1.0（盈亏平衡点）
        fig.add_hline(
            y=1.0,
            line_dash="dot",
            line_color="#E63946",
            line_width=1,
            row=2,
            col=1,
        )

        # 行3: 标准分数与颜色渐变
        fig.add_trace(
            go.Scatter(
                x=df_plot["datetime"],
                y=df_plot["sth_mvrv_zscore"],
                mode="markers+lines",
                line=dict(color="rgba(100,100,100,0.4)", width=1),
                marker=dict(
                    size=4,
                    color=df_plot["sth_mvrv_zscore"],
                    colorscale="RdYlGn_r",
                    # showscale=True,
                    # colorbar=dict(title="Z-Score", len=0.3, y=0.15),
                    # cmin=-3,
                    # cmax=3,
                ),
                hovertemplate="<b>%{x}</b><br>Zscore: %{y:.2f}<extra></extra>",
            ),
            row=3,
            col=1,
        )

        # 标准分数基准线
        fig.add_hline(y=0, line_width=2, line_color="black", row=3, col=1)
        fig.add_hline(
            y=2,
            line_dash="dot",
            line_color="#E63946",
            line_width=2,
            row=3,
            col=1,
            annotation_text="Overbought (+2σ)",
        )
        fig.add_hline(
            y=-2,
            line_dash="dot",
            line_color="#2A9D8F",
            line_width=2,
            row=3,
            col=1,
            annotation_text="Oversold (-2σ)",
        )

        # 使用 find_trend_periods 寻找看涨和看跌时间段并添加背景色
        # 看涨段落：zscore > 0
        bullish_series = df_plot["sth_mvrv_zscore"] > 0
        bullish_series.index = df_plot["datetime"]
        bullish_periods = find_trend_periods(bullish_series)

        for start, end in bullish_periods:
            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor="rgba(42, 157, 143, 0.7)",  # 绿色表示看涨
                line_width=0,
                layer="below",
                row=1,
                col=1,
            )

        # 看跌段落：zscore < 0
        bearish_series = df_plot["sth_mvrv_zscore"] < 0
        bearish_series.index = df_plot["datetime"]
        bearish_periods = find_trend_periods(bearish_series)

        for start, end in bearish_periods:
            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor="rgba(230, 57, 70, 0.7)",  # 红色表示看跌
                line_width=0,
                layer="below",
                row=1,
                col=1,
            )

        # 布局样式
        fig.update_layout(
            title=dict(
                text="STH-MVRV Market Regime Analysis",
                font=dict(size=20, color="#1f2937"),
                x=0.5,
            ),
            height=900,
            hovermode="x unified",
            showlegend=False,
            font=dict(family="Inter, sans-serif", size=12),
        )

        # 轴样式
        fig.update_yaxes(
            title="Price (USD)",
            row=1,
            col=1,
        )
        fig.update_yaxes(
            title="MVRV Ratio",
            row=2,
            col=1,
        )
        fig.update_yaxes(
            title="Z-Score",
            row=3,
            col=1,
        )

        return fig


    indicator_chart = create_indicator_chart(analysis_df)
    indicator_chart
    return


@app.cell
def _(mo):
    mo.md("""
    ## 向量化回溯检验

    ---
    """)
    return


@app.cell
def _(analysis_df, np, pd):
    def run_backtest(df: pd.DataFrame) -> pd.DataFrame:
        backtest_df = df.copy()

        # 删除多余行
        cols_to_drop = ["log_sth_mvrv", "rolling_mean", "rolling_std"]
        backtest_df = backtest_df.drop(
            columns=[col for col in cols_to_drop if col in backtest_df.columns]
        )

        # 生成信号
        backtest_df["signal"] = np.where(backtest_df["sth_mvrv_zscore"] > 0, 1, -1)

        # 将信号滞后1并生成头寸，规避前视偏误
        backtest_df["position"] = backtest_df["signal"].shift(1)

        # 计算日收益率
        backtest_df["market_return"] = backtest_df["btcusd"].pct_change()
        backtest_df["strategy_return"] = (
            backtest_df["position"] * backtest_df["market_return"]
        )

        # 计算净值曲线
        backtest_df["strategy_equity"] = (
            1 + backtest_df["strategy_return"].fillna(0)
        ).cumprod()
        backtest_df["market_equity"] = (
            1 + backtest_df["market_return"].fillna(0)
        ).cumprod()

        return backtest_df


    backtest_results = run_backtest(analysis_df)
    # backtest_results
    return (backtest_results,)


@app.cell
def _(backtest_results, go, make_subplots, pd):
    def create_backtest_chart(df: pd.DataFrame) -> go.Figure:
        # 创建子图
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=(
                "Strategy vs Market Performance",
                "Z-Score Signal",
                "Position History",
            ),
        )

        # 行1:净值曲线对比
        fig.add_trace(
            go.Scatter(
                x=df["datetime"],
                y=df["strategy_equity"],
                name="Strategy",
                line=dict(color="#2E86AB", width=3),
                hovertemplate="<b>%{x}</b><br>Strategy: %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df["datetime"],
                y=df["market_equity"],
                name="Buy & Hold",
                line=dict(color="#F7931A", width=2, dash="dash"),
                hovertemplate="<b>%{x}</b><br>Buy & Hold: %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # 行2: 标准分数
        fig.add_trace(
            go.Scatter(
                x=df["datetime"],
                y=df["sth_mvrv_zscore"],
                name="Z-Score",
                line=dict(color="#6A4C93", width=1.5),
                fill="tonexty",
                fillcolor="rgba(106, 76, 147, 0.1)",
                hovertemplate="<b>%{x}</b><br>Z-Score: %{y:.2f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        # 行3: 持仓变化
        fig.add_trace(
            go.Scatter(
                x=df["datetime"],
                y=df["position"],
                name="Position",
                mode="lines",
                line_shape="hv",
                line=dict(color="#2A9D8F", width=2),
                fill="tozeroy",
                fillcolor="rgba(42, 157, 143, 0.3)",
                hovertemplate="<b>%{x}</b><br>Position: %{y}<extra></extra>",
            ),
            row=3,
            col=1,
        )

        # 调整布局样式
        fig.update_layout(
            title=dict(
                text="STH-MVRV Strategy Backtest Analysis",
                font=dict(size=20, color="#1f2937"),
                x=0.5,
            ),
            height=900,
            hovermode="x unified",
            plot_bgcolor="white",
            font=dict(family="Inter, sans-serif", size=12),
        )

        # 坐标轴样式
        fig.update_yaxes(
            title="Equity Value",
            row=1,
            col=1,
        )
        fig.update_yaxes(
            title="Z-Score",
            row=2,
            col=1,
        )
        fig.update_yaxes(
            title="Position",
            row=3,
            col=1,
            tickvals=[-1, 0, 1],
            ticktext=["Short", "Neutral", "Long"],
        )

        return fig


    backtest_chart = create_backtest_chart(backtest_results)
    backtest_chart
    return


@app.cell
def _():
    # # Prepare performance data for ffn analysis
    # perf_data = backtest_results.set_index("datetime")[
    #     ["strategy_equity", "market_equity"]
    # ]
    # perf_data.columns = ["Strategy", "Benchmark"]

    # # Calculate comprehensive statistics
    # stats = ffn.calc_stats(perf_data)

    # # Display performance statistics
    # mo.md("**Detailed Performance Metrics:**")
    # stats.display()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
