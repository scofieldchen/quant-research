import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import json
    from datetime import datetime, date, timedelta
    from pathlib import Path

    import duckdb
    import ffn
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio

    pio.templates.default = "simple_white"

    output_dir = Path("notebooks/sth_mvrv/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    return date, datetime, duckdb, go, make_subplots, mo, np, output_dir, pd


@app.cell
def _(mo):
    mo.md("""
    # STH-MVRV 动量系统 🚀
    """)
    return


@app.cell
def _(duckdb, pd):
    # 读取数据


    def load_sth_mvrv_data() -> pd.DataFrame:
        file_path = "/users/scofield/quant-research/data/cleaned/sth_mvrv.parquet"

        sql_query = f"""
        SELECT datetime, sth_mvrv, open, close
        FROM '{file_path}'
        ORDER BY datetime
        """

        df = duckdb.sql(sql_query).df()
        return df


    raw_df = load_sth_mvrv_data()
    # raw_df
    return (raw_df,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 指标分析

    ---
    """)
    return


@app.cell
def _(np, pd):
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
    return (calculate_sth_mvrv_zscore,)


@app.cell
def _(datetime, pd):
    def find_trend_periods(
        series: pd.Series,
    ) -> list[tuple[datetime, datetime]]:
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
    return (find_trend_periods,)


@app.cell
def _(datetime, go, make_subplots, pd):
    def create_indicator_chart(
        df: pd.DataFrame,
        bullish_periods: list[tuple[datetime, datetime]],
        bearish_periods: list[tuple[datetime, datetime]],
    ) -> go.Figure:
        df_plot = df.copy()

        # 创建子图
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            shared_yaxes=False,
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
                x=df_plot.index,
                y=df_plot["close"],
                line=dict(color="#F7931A", width=2.5),
                hovertemplate="<b>%{x}</b><br>BTCUSD: %{y:,.0f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # 行2: STH-MVRV 比率
        fig.add_trace(
            go.Scatter(
                x=df_plot.index,
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
                x=df_plot.index,
                y=df_plot["sth_mvrv_zscore"],
                mode="markers+lines",
                line=dict(color="rgba(100,100,100,0.4)", width=1),
                marker=dict(
                    size=4,
                    color=df_plot["sth_mvrv_zscore"],
                    colorscale="RdYlGn_r",
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

        # 添加背景色显示看涨行情
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

        # 添加背景色显示看跌行情
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
            width=1000,
            height=800,
            hovermode="x unified",
            showlegend=False,
            font=dict(family="Inter, sans-serif", size=12),
        )

        # 轴样式
        fig.update_yaxes(
            title="Price (USD)",
            fixedrange=False,
            row=1,
            col=1,
        )
        fig.update_yaxes(
            title="MVRV Ratio",
            fixedrange=False,
            row=2,
            col=1,
        )
        fig.update_yaxes(
            title="Z-Score",
            fixedrange=False,
            row=3,
            col=1,
        )

        return fig
    return (create_indicator_chart,)


@app.cell
def _(date, mo):
    parameter_form = mo.md("""
        {zscore_window}

        {start_date}

        {end_date}
        """).batch(
        zscore_window=mo.ui.number(
            start=10, stop=200, step=1, value=50, label="标准分数窗口"
        ),
        start_date=mo.ui.date(value="2024-01-01", label="开始日期"),
        end_date=mo.ui.date(
            value=date.today().strftime("%Y-%m-%d"), label="结束日期"
        ),
    )

    parameter_form
    return (parameter_form,)


@app.cell
def _(
    calculate_sth_mvrv_zscore,
    create_indicator_chart,
    find_trend_periods,
    parameter_form,
    raw_df,
):
    # 计算指标
    zscore_df = calculate_sth_mvrv_zscore(
        raw_df, window=parameter_form.value["zscore_window"]
    )
    zscore_df.set_index("datetime", inplace=True)

    # 筛选可视化数据
    visualization_df = zscore_df.loc[
        parameter_form.value["start_date"] : parameter_form.value["end_date"],
        ["sth_mvrv", "close", "sth_mvrv_zscore"],
    ]

    # 识别看涨日期和看跌日期
    bullish_regime = visualization_df["sth_mvrv_zscore"] > 0
    bullish_dates = find_trend_periods(bullish_regime)

    bearish_regime = visualization_df["sth_mvrv_zscore"] < 0
    bearish_dates = find_trend_periods(bearish_regime)

    # 创建指标图表
    indicator_chart = create_indicator_chart(
        visualization_df,
        bullish_periods=bullish_dates,
        bearish_periods=bearish_dates,
    )
    return indicator_chart, visualization_df


@app.cell
def _(mo, visualization_df):
    mo.ui.table(
        visualization_df.tail(10).round(2),
        selection=None,
        show_column_summaries=False,
        show_data_types=False,
    )
    return


@app.cell
def _(indicator_chart):
    indicator_chart
    return


@app.cell
def _(indicator_chart, mo, output_dir):
    indicator_chart_path = output_dir / "indicator_chart.png"
    indicator_chart.write_image(indicator_chart_path, scale=1)
    mo.md(f"**指标图表保存到**: {indicator_chart_path}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 回溯检验

    ---
    """)
    return


@app.cell
def _(go, make_subplots, np, pd):
    class IterativeBacktest:
        """
        基于循环的回溯检验类，模拟真实的交易环境。

        遍历k线，根据昨天收盘的信号进行交易，以当前k线的开盘价进场和平仓。

        没有考虑交易成本，回测结果只能衡量核心信号和指标是否有效，不代表历史交易的真实结果。
        """

        def __init__(self, data: pd.DataFrame, initial_capital: float = 10000.0):
            required_cols = ["open", "close", "signal"]
            if not all(col in data.columns for col in required_cols):
                raise ValueError(f"输入数据必须包含以下列: {required_cols}")

            self.raw_data = data.copy()
            self.initial_capital = initial_capital

            self.results = None
            self.trades_list = []

            self._run_backtest_loop()

        def _run_backtest_loop(self):
            """
            使用 for 循环模拟逐日交易过程。
            """
            equity = self.initial_capital
            current_holding = 0

            # 交易记录变量
            entry_price = 0.0
            entry_time = None

            # 历史记录容器
            dates_record = []
            equities_record = []
            positions_record = []

            # Numpy 加速提取
            idx = self.raw_data.index
            opens = self.raw_data["open"].values
            closes = self.raw_data["close"].values
            signals = self.raw_data["signal"].values

            # 1. 主循环：遍历每一天
            for i in range(1, len(self.raw_data)):
                curr_date = idx[i]
                prev_close = closes[i - 1]
                curr_open = opens[i]
                curr_close = closes[i]

                # T日的决策由 T-1 信号决定
                target_pos = signals[i - 1]

                # --- 资金计算 ---
                # 1. 隔夜盈亏 (旧持仓)
                pct_overnight = (curr_open - prev_close) / prev_close
                equity = equity * (1 + current_holding * pct_overnight)

                # --- 交易执行检测 ---
                if target_pos != current_holding:
                    # 平掉旧仓位 (如果有) -> 标记为 Closed
                    if current_holding != 0:
                        exit_price = curr_open
                        exit_time = curr_date

                        trade_pnl = (
                            (exit_price - entry_price)
                            / entry_price
                            * current_holding
                        )

                        self.trades_list.append(
                            {
                                "entry_time": entry_time,
                                "entry_price": entry_price,
                                "exit_time": exit_time,
                                "exit_price": exit_price,
                                "position": current_holding,
                                "pnl_pct": trade_pnl,
                                "status": "Closed",  # 状态：已平仓
                            }
                        )

                    # 开新仓位
                    if target_pos != 0:
                        entry_price = curr_open
                        entry_time = curr_date

                    current_holding = target_pos

                # 2. 日内盈亏 (新持仓)
                pct_intraday = (curr_close - curr_open) / curr_open
                equity = equity * (1 + current_holding * pct_intraday)

                # 记录过程
                dates_record.append(curr_date)
                equities_record.append(equity)
                positions_record.append(current_holding)

            # 2. 循环结束后的收尾工作：处理最后一笔未平仓交易 (Open Trade)
            if current_holding != 0:
                # 使用最后一条数据的收盘价进行盯市(Mark-to-Market)
                last_price = closes[-1]
                last_date = idx[-1]

                trade_pnl = (
                    (last_price - entry_price) / entry_price * current_holding
                )

                self.trades_list.append(
                    {
                        "entry_time": entry_time,
                        "entry_price": entry_price,
                        "exit_time": last_date,  # 假设此时刻结算
                        "exit_price": last_price,  # 结算价
                        "position": current_holding,
                        "pnl_pct": trade_pnl,
                        "status": "Open",  # 状态：持仓中
                    }
                )

            # 保存结果
            self.results = pd.DataFrame(
                {"equity_curve": equities_record, "position": positions_record},
                index=dates_record,
            )

            self.results = self.results.join(
                self.raw_data[["open", "close", "signal"]]
            )

        def get_trades(self) -> pd.DataFrame:
            """获取交易记录"""
            df = pd.DataFrame(self.trades_list)
            # 确保列顺序美观（如果有数据的话）
            if not df.empty:
                cols = [
                    "status",
                    "entry_time",
                    "entry_price",
                    "exit_time",
                    "exit_price",
                    "position",
                    "pnl_pct",
                ]
                return df[cols]
            return df

        def get_performance_stats(self) -> dict:
            """
            计算业绩指标
            注意：Trade-based metrics 仅基于 'Closed' 交易计算。
            """
            all_trades = self.get_trades()

            # 筛选已平仓交易
            if not all_trades.empty:
                closed_trades = all_trades[all_trades["status"] == "Closed"]
            else:
                closed_trades = pd.DataFrame()

            # 1. 交易基础指标 (仅针对 Closed Trades)
            if closed_trades.empty:
                trade_stats = {
                    "Total Closed Trades": 0,
                    "Status": "No closed trades generated",
                }
            else:
                total = len(closed_trades)
                wins = len(closed_trades[closed_trades["pnl_pct"] > 0])
                losses = len(closed_trades[closed_trades["pnl_pct"] <= 0])
                win_rate = wins / total

                gross_p = closed_trades[closed_trades["pnl_pct"] > 0][
                    "pnl_pct"
                ].sum()
                gross_l = abs(
                    closed_trades[closed_trades["pnl_pct"] <= 0]["pnl_pct"].sum()
                )
                pf = gross_p / gross_l if gross_l != 0 else np.inf

                trade_stats = {
                    "Total Closed Trades": total,
                    "Win Rate": f"{win_rate:.2%}",
                    "Profit Factor": f"{pf:.2f}",
                    "Avg PnL (Closed)": f"{closed_trades['pnl_pct'].mean():.2%}",
                    "Open Positions": len(all_trades)
                    - len(closed_trades),  # 统计持仓数
                }

            # 2. 收益率基础指标 (基于净值曲线，包含 Open PnL)
            # ffn 计算的是基于 'equity_curve' 的，这已经隐含了未平仓盈亏
            equity_series = self.results["equity_curve"]
            if len(equity_series) > 10:
                perf = equity_series.calc_stats()
                return_stats = {
                    "Total Return": f"{perf.stats['total_return']:.2%}",
                    "CAGR": f"{perf.stats['cagr']:.2%}",
                    "Sharpe Ratio": f"{perf.stats['daily_sharpe']:.2f}",
                    "Max Drawdown": f"{perf.stats['max_drawdown']:.2%}",
                }
            else:
                return_stats = {"Status": "Not enough data for ffn"}

            return {**trade_stats, **return_stats}

        def plot_backtest_result(self, width: int = 1000, height: int = 600):
            if self.results is None or self.results.empty:
                print("No results")
                return

            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.7, 0.3],
                subplot_titles=("Equity", "Position"),
            )
            fig.add_trace(
                go.Scatter(
                    x=self.results.index,
                    y=self.results["equity_curve"],
                    mode="lines",
                    name="Equity",
                    line=dict(color="blue"),
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=self.results.index,
                    y=self.results["position"],
                    mode="lines",
                    name="Position",
                    line=dict(color="orange", width=1, shape="hv"),
                    fill="tozeroy",
                ),
                row=2,
                col=1,
            )
            fig.update_layout(
                title="Backtest Results",
                width=width,
                height=height,
                showlegend=False,
            )

            return fig
    return (IterativeBacktest,)


@app.cell
def _(date, mo):
    backtest_parameter_form = mo.md("""
        {zscore_window}

        {start_date}

        {end_date}
        """).batch(
        zscore_window=mo.ui.number(
            start=10, stop=200, step=1, value=50, label="标准分数窗口"
        ),
        start_date=mo.ui.date(value="2024-01-01", label="开始日期"),
        end_date=mo.ui.date(
            value=date.today().strftime("%Y-%m-%d"), label="结束日期"
        ),
    )

    backtest_parameter_form
    return (backtest_parameter_form,)


@app.cell
def _(
    IterativeBacktest,
    backtest_parameter_form,
    calculate_sth_mvrv_zscore,
    np,
    raw_df,
):
    # 获取参数
    backtest_zscore_window = backtest_parameter_form.value["zscore_window"]
    backtest_start_date = backtest_parameter_form.value["start_date"]
    backtest_end_date = backtest_parameter_form.value["end_date"]

    # 计算指标
    backtest_df = calculate_sth_mvrv_zscore(raw_df, backtest_zscore_window)
    backtest_df.set_index("datetime", inplace=True)
    backtest_df = backtest_df.loc[backtest_start_date:backtest_end_date]
    backtest_df.drop(
        columns=["log_sth_mvrv", "rolling_mean", "rolling_std"], inplace=True
    )

    # 生成信号
    # 标准分数 > 0，做多，用1表示多头信号
    # 标准分数 < 0，做空，用-1表示空头信号
    backtest_df["signal"] = np.where(backtest_df["sth_mvrv_zscore"] >= 0, 1, -1)

    # 运行回溯检验
    bt = IterativeBacktest(backtest_df)
    return (bt,)


@app.cell
def _(bt, mo):
    mo.ui.table(
        bt.results,
        selection=None,
        show_column_summaries=False,
        show_data_types=False,
        format_mapping={
            "equity_curve": "{:.1f}",
            "open": "{:.1f}",
            "close": "{:.1f}",
        },
    )
    return


@app.cell
def _(bt, mo):
    trades = bt.get_trades()


    def style_cell(_rowId, _columnName, value):
        if _columnName == "pnl_pct":
            if value > 0:
                return {
                    "color": "green",
                    "fontStyle": "italic",
                }
        return {}


    mo.ui.table(
        trades.tail(10),
        selection=None,
        show_column_summaries=False,
        show_data_types=False,
        format_mapping={
            "entry_price": "{:.1f}",
            "exit_price": "{:.1f}",
            "entry_time": "{:%Y-%m-%d}",
            "exit_time": "{:%Y-%m-%d}",
            "pnl_pct": "{:.1%}",
        },
        style_cell=style_cell,
    )
    return


@app.cell
def _(bt):
    bt.get_performance_stats()
    return


@app.cell
def _(bt):
    backtest_chart = bt.plot_backtest_result(width=900, height=650)
    backtest_chart
    return (backtest_chart,)


@app.cell
def _(backtest_chart, mo, output_dir):
    backtest_chart_path = output_dir / "backtest_chart.png"
    backtest_chart.write_image(backtest_chart_path, scale=1)
    mo.md(f"**指标图表保存到**: {backtest_chart_path}")
    return


@app.cell
def _(bt, output_dir, pd, raw_df, visualization_df):
    def generate_summary_report(raw_df, visualization_df, bt) -> str:
        """生成 STH-MVRV 分析报告的 Markdown 内容。

        Args:
            raw_df (pd.DataFrame): 原始数据帧，用于提取最后日期。
            visualization_df (pd.DataFrame): 可视化数据帧，用于指标数据表格。
            bt: 回溯检验对象，提供业绩指标和交易数据。

        Returns:
            str: 生成的 Markdown 文档字符串。
        """
        # 提取关键数据
        last_date = raw_df["datetime"].max().strftime("%Y-%m-%d")
        indicators_tail = visualization_df.tail(30)
        performance_stats = bt.get_performance_stats()
        trades_tail = bt.get_trades().tail(10)

        # 生成 Markdown 文档
        markdown_content = f"""# STH-MVRV 分析报告 - {last_date}

    ## 更新日期
    {last_date}

    ## 指标数据（最后30行）
    | datetime | sth_mvrv | close | sth_mvrv_zscore |
    |----------|----------|-------|-----------------|
    """

        for _, row in indicators_tail.iterrows():
            markdown_content += f"| {row.name.strftime('%Y-%m-%d')} | {row['sth_mvrv']:.4f} | {row['close']:.2f} | {row['sth_mvrv_zscore']:.4f} |\n"

        markdown_content += "\n## 回溯检验的业绩指标\n"
        for key, value in performance_stats.items():
            markdown_content += f"- {key}: {value}\n"

        markdown_content += "\n## 回溯检验的历史交易（最后10笔）\n"
        markdown_content += "| status | entry_time | entry_price | exit_time | exit_price | position | pnl_pct |\n"
        markdown_content += "|--------|------------|-------------|-----------|------------|----------|---------|\n"

        for _, row in trades_tail.iterrows():
            entry_time = (
                row["entry_time"].strftime("%Y-%m-%d")
                if pd.notna(row["entry_time"])
                else "N/A"
            )
            exit_time = (
                row["exit_time"].strftime("%Y-%m-%d")
                if pd.notna(row["exit_time"])
                else "N/A"
            )
            markdown_content += f"| {row['status']} | {entry_time} | {row['entry_price']:.2f} | {exit_time} | {row['exit_price']:.2f} | {row['position']} | {row['pnl_pct']:.2%} |\n"

        markdown_content += "\n## 图表\n"
        markdown_content += "![Indicator Chart](indicator_chart.png)\n\n"
        markdown_content += "![Backtest Chart](backtest_chart.png)\n"

        return markdown_content


    # 写入文件
    summary_path = output_dir / "summary.md"
    markdown_content = generate_summary_report(
        raw_df=raw_df, visualization_df=visualization_df, bt=bt
    )
    summary_path.write_text(markdown_content)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
