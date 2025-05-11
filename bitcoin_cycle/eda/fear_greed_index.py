import marimo

__generated_with = "0.11.31"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import sys

    sys.path.insert(0, "/users/scofield/quant-research/bitcoin_cycle/")

    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import requests
    import yfinance as yf
    from plotly.subplots import make_subplots

    from indicators import fisher_transform, lowpass_filter
    from signals import Metric

    yf.set_config(
        proxy={"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}
    )
    return (
        Metric,
        fisher_transform,
        go,
        lowpass_filter,
        make_subplots,
        np,
        pd,
        requests,
        sys,
        yf,
    )


@app.cell
def _(mo, pd, requests, yf):
    def fetch_fear_greed_index(limit: int = 10) -> pd.DataFrame:
        url = "https://api.alternative.me/fng/"
        resp = requests.get(url, params={"limit": limit})
        resp.raise_for_status()
        data = resp.json()

        return (
            pd.DataFrame.from_records(data["data"], exclude=["time_until_update"])
            .astype({"value": "float64", "timestamp": "int"})
            .assign(date=lambda x: pd.to_datetime(x.timestamp, unit="s"))
            .sort_values("date", ascending=True)
            .set_index("date")
            .drop(columns=["timestamp"])
            .rename(columns={"value_classification": "classification"})
        )


    @mo.cache
    def fetch_all_data() -> pd.DataFrame:
        # 获取比特币历史价格
        btcusd = yf.download(
            tickers="BTC-USD",
            ignore_tz=True,
            auto_adjust=True,
            progress=False,
            multi_level_index=False,
        )
        print(f"Downloaded BTCUSD, last:{btcusd.index.max():%Y-%m-%d}")

        # 获取贪婪和恐慌指数
        fgi = fetch_fear_greed_index(10 * 365)
        print(f"Downloaded fear greed index, last:{fgi.index.max():%Y-%m-%d}")

        # 合并数据
        df = (
            fgi.join(btcusd["Close"], how="left")
            .rename(columns={"Close": "btcusd", "value": "fgi"})
            .dropna()
        )

        return df
    return fetch_all_data, fetch_fear_greed_index


@app.cell
def _(fetch_all_data):
    df = fetch_all_data()
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(Metric, go, lowpass_filter, np, pd):
    class FearGreedIndex(Metric):
        """恐慌和贪婪指数指标"""

        @property
        def name(self) -> str:
            return "Fear and Greed index"

        @property
        def description(self) -> str:
            return "通过恐慌和贪婪指数衡量比特币的市场情绪，识别潜在的市场顶部和底部。"

        def __init__(
            self,
            data: pd.DataFrame,
            price_col: str = "btcusd",
            fgi_col: str = "fgi",
            smooth_period: int = 10,
            extreme_greed_threshold: float = 80.0,
            extreme_fear_threshold: float = 20.0,
        ) -> None:
            """
            初始化 FearGreedIndex 指标类

            Args:
                data: 包含数据的 DataFrame
                price_col: DataFrame 中表示比特币价格列的名称
                fgi_col: DataFrame 中表示恐慌贪婪指数列的名称
            """
            self.price_col = price_col
            self.fgi_col = fgi_col
            self.smooth_period = smooth_period
            self.extreme_greed_threshold = extreme_greed_threshold
            self.extreme_fear_threshold = extreme_fear_threshold
            super().__init__(data)

        def _validate_data(self) -> None:
            for col in [self.price_col, self.fgi_col]:
                if col not in self.data.columns:
                    raise ValueError(
                        f"Input dataframe is missing required column: {col}"
                    )

        def generate_signals(self) -> None:
            self.signals = self.data.copy()
            self.signals["smooth_fgi"] = lowpass_filter(
                self.signals[self.fgi_col], self.smooth_period
            )
            signals = np.where(
                self.signals["smooth_fgi"] >= self.extreme_greed_threshold, 1, 0
            )
            signals = np.where(
                self.signals["smooth_fgi"] <= self.extreme_fear_threshold,
                -1,
                signals,
            )
            self.signals["signal"] = signals

        def _add_indicator_traces(self, fig: go.Figure) -> None:
            # 添加原始指标
            fig.add_trace(
                go.Scatter(
                    x=self.signals.index,
                    y=self.signals[self.fgi_col],
                    name="Fear greed index",
                    line=dict(color="#add8e6", width=1.5),
                    opacity=0.5,
                ),
                row=2,
                col=1,
            )

            # 添加移动平滑曲线
            fig.add_trace(
                go.Scatter(
                    x=self.signals.index,
                    y=self.signals["smooth_fgi"],
                    name=f"Smoothed index({self.smooth_period}-days)",
                    line=dict(color="royalblue", width=2),
                ),
                row=2,
                col=1,
            )

            # 添加水平曲线表示极值区域
            levels = [
                (self.extreme_fear_threshold, "extreme fear"),
                (50, "neutral"),
                (self.extreme_greed_threshold, "extreme greed"),
            ]
            for level, text in levels:
                fig.add_hline(
                    level,
                    row=2,
                    col=1,
                    line_dash="dot",
                    line_color="grey",
                    line_width=2,
                    annotation_text=text,
                    annotation_position="top left",
                )

            # 更新第二行 y 轴设置
            fig.update_yaxes(
                row=2,
                col=1,
                title="Index(0-100)",
                gridcolor="#E5E5E5",
                title_font=dict(size=14),
            )


    class ConsecutiveGreedDays(Metric):
        """连续贪婪天数指标"""

        @property
        def name(self) -> str:
            return "Consecutive Greed Days"

        @property
        def description(self) -> str:
            return "统计市场处于贪婪或极度贪婪状态的天数，基于恐慌贪婪指数"

        def __init__(
            self,
            data: pd.DataFrame,
            price_col: str = "btcusd",
            sentiment_col: str = "classification",
            extreme_level: int = 40,
        ) -> None:
            """
            初始化 ConsecutiveGreedDays 指标类

            Args:
                data: 包含数据的 DataFrame
                price_col: DataFrame 中表示比特币价格列的名称
                sentiment_col: DataFrame 中表示市场情绪状态的名称
            """
            self.price_col = price_col
            self.sentiment_col = sentiment_col
            self.extreme_level = extreme_level
            super().__init__(data)

        def _validate_data(self) -> None:
            for col in [self.price_col, self.sentiment_col]:
                if col not in self.data.columns:
                    raise ValueError(
                        f"Input dataframe is missing required column: {col}"
                    )

        def generate_signals(self) -> None:
            self.signals = self.data.copy()
            greed_days = np.zeros(len(self.signals), int)

            for i in range(1, len(greed_days)):
                if self.signals[self.sentiment_col].iloc[i] in [
                    "Greed",
                    "Extreme Greed",
                ]:
                    greed_days[i] = greed_days[i - 1] + 1

            self.signals["greed_days"] = greed_days
            self.signals["signal"] = np.where(
                self.signals["greed_days"] >= self.extreme_level, 1, 0
            )

        def _add_indicator_traces(self, fig: go.Figure) -> None:
            # 添加连续贪婪天数曲线
            fig.add_trace(
                go.Scatter(
                    x=self.signals.index,
                    y=self.signals["greed_days"],
                    name="Greed Days",
                    line=dict(color="royalblue", width=2),
                    hovertemplate="<b>Date</b>: %{x}<br>"
                    + "<b>Days in Greed</b>: %{y:.0f}<br><extra></extra>",
                ),
                row=2,
                col=1,
            )

            # 添加水平线表示连续贪婪天数的极端水平
            reference_levels = [40, 60, 80, 100]
            for level in reference_levels:
                fig.add_hline(
                    y=level,
                    row=2,
                    col=1,
                    line_dash="dot",
                    line_color="#34495e",
                    line_width=1,
                    annotation_text=f"{level} days",
                    annotation_position="right",
                )

            # 更新 y 轴设置
            fig.update_yaxes(
                row=2,
                col=1,
                title="Days",
                gridcolor="#E5E5E5",
                title_font=dict(size=14),
            )
    return ConsecutiveGreedDays, FearGreedIndex


@app.cell
def _(ConsecutiveGreedDays, df):
    # metric = FearGreedIndex(
    #     df,
    #     price_col="btcusd",
    #     fgi_col="fgi",
    #     smooth_period=10,
    #     extreme_greed_threshold=80,
    #     extreme_fear_threshold=20,
    # )
    # metric.generate_signals()
    # metric.signals

    metric = ConsecutiveGreedDays(
        df, price_col="btcusd", sentiment_col="classification", extreme_level=40
    )
    metric.generate_signals()
    metric.signals
    return (metric,)


@app.cell
def _(metric):
    fig = metric.generate_chart()
    fig.show()
    return (fig,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
