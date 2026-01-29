import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import duckdb
    import pandas as pd
    import altair as alt
    from datetime import datetime, date, timedelta
    from pathlib import Path
    import numpy as np

    TICKER_PATH = Path("data/cleaned/binance_tickers_perp.parquet")
    DATA_PATH = Path("data/cleaned/binance_klines_perp_m1")
    return DATA_PATH, Path, TICKER_PATH, alt, date, duckdb, mo, pd, timedelta


@app.cell
def _(mo):
    mo.md("""
    # 市场广度分析模型 📊

    本模型通过分析主流加密货币交易对的相对强弱，衡量整体市场情绪和趋势健康度。

    **核心指标：**
    1. **腾落线 (A/D Line)**: 衡量上涨与下跌资产数量的累积差额。
    2. **均线以上占比 (% Above MA)**: 价格高于特定均线的资产比例（衡量超买/超卖）。
    """)
    return


@app.cell
def _(date, mo, timedelta):
    # 使用 Form 封装参数组件
    params_form = (
        mo.md(
            r"""
            **配置分析参数**

            {top_num}

            {ma_window}

            {timeframe}

            {date_range}
            """
        )
        .batch(
            top_num=mo.ui.number(
                start=10, stop=100, step=10, value=20, label="最高市值的交易对数量"
            ),
            ma_window=mo.ui.number(
                start=1, stop=200, step=1, value=50, label="均线回溯期"
            ),
            timeframe=mo.ui.dropdown(
                options={"1小时": "1 hour", "4小时": "4 hours", "1天": "1 day"},
                value="1天",
                label="K线周期",
            ),
            date_range=mo.ui.date_range(
                start=date.today() - timedelta(days=200),
                stop=date.today(),
                label="时间范围",
            ),
        )
        .form(bordered=True)
    )

    params_form
    return (params_form,)


@app.cell
def _(DATA_PATH, Path, TICKER_PATH, duckdb, mo, params_form, timedelta):
    # 只有当表单提交后才执行
    mo.stop(params_form.value is None)


    def load_top_tickers(file_path: Path, limit: int = 30) -> list[str]:
        """加载市场排名最高的 binance 永续合约交易对"""
        query = f"""
        SELECT symbol,coingecko_market_cap
        FROM '{file_path}'
        WHERE status = 'TRADING' 
          AND quote_asset = 'USDT'
          AND onboard_date <= CAST('2024-01-01' AS TIMESTAMP)
          AND base_asset NOT IN ('USDC','BUSD','FUSD','T')
        ORDER BY coingecko_market_cap DESC
        LIMIT {limit}
        """
        df = duckdb.sql(query).df()
        return df["symbol"].to_list()


    def load_breadth_data(
        symbols: list[str],
        start_date,
        end_date,
        interval_str: str,
        ma_window: int,
    ):
        """
        加载数据，并包含足够的回溯期以计算均线。
        """
        # 计算回溯天数
        lookback_multiplier = 1
        if "hour" in interval_str:
            lookback_multiplier = int(interval_str.split()[0])
        elif "day" in interval_str:
            lookback_multiplier = 24

        # 额外增加 10 个周期作为缓冲区
        lookback_hours = (ma_window + 10) * lookback_multiplier
        query_start = start_date - timedelta(hours=lookback_hours)

        symbols_str = ", ".join([f"'{s}'" for s in symbols])

        sql = f"""
        SELECT 
            time_bucket(INTERVAL '{interval_str}', datetime AT TIME ZONE 'UTC') as bucket,
            symbol,
            arg_max(close, datetime) as close
        FROM read_parquet('{DATA_PATH}/*/*/data.parquet', hive_partitioning=1)
        WHERE symbol IN ({symbols_str})
          AND datetime >= '{query_start}'
          AND datetime <= '{end_date}'
        GROUP BY bucket, symbol
        ORDER BY bucket ASC
        """

        return duckdb.sql(sql).df()


    with mo.status.spinner(title="数据加载与计算中..."):
        # 获取表单值
        form_val = params_form.value
        top_tickers = load_top_tickers(TICKER_PATH, form_val["top_num"])
        raw_data = load_breadth_data(
            symbols=top_tickers,
            start_date=form_val["date_range"][0],
            end_date=form_val["date_range"][1],
            interval_str=form_val["timeframe"],
            ma_window=form_val["ma_window"],
        )
    return form_val, raw_data


@app.cell
def _(form_val, mo, params_form, pd, raw_data):
    mo.stop(params_form.value is None)


    def calculate_indicators(df, window: int, start_limit):
        """
        计算指标并过滤回用户选择的时间范围。
        """
        pivot_df = df.pivot(
            index="bucket", columns="symbol", values="close"
        ).ffill()

        # 1. 均线占比
        sma = pivot_df.rolling(window=window).mean()
        above_ma = (pivot_df > sma).sum(axis=1) / pivot_df.shape[1]

        # 2. 腾落线
        diff = pivot_df.diff()
        ad_line = ((diff > 0).sum(axis=1) - (diff < 0).sum(axis=1)).cumsum()

        # 合并
        res = pd.DataFrame(
            {
                "breadth": above_ma,
                "ad_line": ad_line,
                "btc_close": pivot_df["BTCUSDT"]
                if "BTCUSDT" in pivot_df.columns
                else None,
            },
            index=pivot_df.index,
        )

        # 过滤回用户选择的起始时间
        return res[res.index >= start_limit].dropna()


    results_df = calculate_indicators(
        raw_data, form_val["ma_window"], pd.Timestamp(form_val["date_range"][0])
    )
    return (results_df,)


@app.cell
def _(alt, mo, params_form, pd, results_df):
    mo.stop(params_form.value is None)


    def create_breadth_charts(df):
        plot_data = df.reset_index()

        # 基础配置：移除 X 轴标题，统一时间格式
        x_axis = alt.X("bucket:T", title=None)

        # BTC 价格线 (背景)
        btc_base = alt.Chart(plot_data).encode(x=x_axis)
        btc_line = btc_base.mark_line(
            color="#17becf", strokeWidth=1.5, opacity=0.7
        ).encode(
            y=alt.Y(
                "btc_close:Q",
                title="BTCUSDT",
                scale=alt.Scale(zero=False),
                axis=alt.Axis(orient="right", titleColor="#999"),
            )
        )

        # --- 图表 1: 均线占比 ---
        breadth_area = (
            alt.Chart(plot_data)
            .mark_area(
                line={"color": "#1f77b4", "strokeWidth": 2},
                color=alt.Gradient(
                    gradient="linear",
                    stops=[
                        alt.GradientStop(color="white", offset=0),
                        alt.GradientStop(color="#1f77b4", offset=1),
                    ],
                    x1=1,
                    x2=1,
                    y1=1,
                    y2=0,
                ),
                opacity=0.2,
            )
            .encode(
                x=x_axis,
                y=alt.Y(
                    "breadth:Q",
                    title="均线以上占比",
                    scale=alt.Scale(domain=[-0.1, 1.1]),
                    axis=alt.Axis(format="%"),
                ),
            )
        )

        thresholds = (
            alt.Chart(pd.DataFrame({"y": [0.2, 0.8], "color": ["green", "red"]}))
            .mark_rule(strokeDash=[4, 4])
            .encode(
                y=alt.Y(
                    "y:Q",
                    scale=alt.Scale(domain=[-0.1, 1.1]),
                    title=None,
                    axis=None,
                ),
                color=alt.Color("color:N", scale=None),
            )
        )

        chart1 = (
            alt.layer(btc_line, breadth_area, thresholds)
            .resolve_scale(y="independent")
            .properties(
                width=700, height=280, title="市场广度：均线以上占比 (对比 BTC)"
            )
        )

        # --- 图表 2: 腾落线 ---
        ad_line = (
            alt.Chart(plot_data)
            .mark_line(color="#ff7f0e", strokeWidth=2)
            .encode(
                x=x_axis,
                y=alt.Y(
                    "ad_line:Q",
                    title="腾落线 (A/D Line)",
                    scale=alt.Scale(zero=False),
                ),
            )
        )

        chart2 = (
            alt.layer(btc_line, ad_line)
            .resolve_scale(y="independent")
            .properties(width=700, height=280, title="市场情绪：腾落线 (对比 BTC)")
        )

        return chart1, chart2


    chart_breadth, chart_ad = create_breadth_charts(results_df)
    return chart_ad, chart_breadth


@app.cell
def _(chart_ad, chart_breadth, mo, params_form, results_df):
    mo.stop(params_form.value is None)

    latest = results_df.iloc[-1]
    prev = results_df.iloc[-2]

    # 状态统计卡片
    stats = mo.hstack(
        [
            mo.stat(
                value=f"{latest['breadth']:.1%}",
                label="均线以上占比",
            ),
            mo.stat(
                value=f"{latest['ad_line']:.0f}",
                label="腾落线 (A/D)",
            ),
            mo.stat(
                value=f"${latest['btc_close']:,.0f}",
                label="BTC 价格",
            ),
        ],
        justify="space-around",
    )

    # 布局展示
    mo.vstack(
        [
            mo.md("### 核心分析结果"),
            stats,
            mo.md("---"),
            chart_breadth,
            mo.md(" "),
            chart_ad,
        ],
        align="center",
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
