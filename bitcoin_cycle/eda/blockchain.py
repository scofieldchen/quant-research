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
    return Metric, go, lowpass_filter, np, pd, yf


@app.cell
def _(mo, pd, yf):
    @mo.cache
    def get_all_data() -> pd.DataFrame:
        # 获取比特币历史价格
        btcusd = yf.download(
            tickers="BTC-USD",
            ignore_tz=True,
            auto_adjust=True,
            progress=False,
            multi_level_index=False,
        )

        # 区块链指标数据
        filepath = "/users/scofield/quant-research/bitcoin_cycle/data/sth_mvrv.csv"
        metric = pd.read_csv(filepath, index_col="datetime", parse_dates=True)

        # 合并数据
        df = (
            pd.concat([metric, btcusd["Close"]], join="outer", axis=1)
            .rename(columns={"Close": "btcusd"})
            .ffill()
            .dropna()
        )

        return df
    return (get_all_data,)


@app.cell
def _(get_all_data):
    data = get_all_data()
    data
    return (data,)


@app.cell(hide_code=True)
def _(Metric, go, lowpass_filter, np, pd):
    class STHRealizedPrice(Metric):
        """短期持有者的实现价格"""

        @property
        def name(self) -> str:
            return "Short-Term Holder Realized Price"

        @property
        def description(self) -> str:
            pass

        def __init__(
            self,
            data: pd.DataFrame,
            price_col: str = "btcusd",
            sth_rp_col: str = "sth_realized_price",
            smooth_period: int = 7,
            rolling_period: int = 200,
            upper_band_percentile: float = 0.99,
            lower_band_percentile: float = 0.01,
        ) -> None:
            """
            初始化 STHRealizedPrice 指标类

            Args:
                data: 包含价格和 STH 实现价格列的 DataFrame。
                price_col: DataFrame 中比特币价格列的名称。
                sth_rp_col: DataFrame 中 STH 实现价格列的名称。
                smooth_period: 平滑窗口。
                rolling_period: 计算滚动百分位数通道的窗口。
                upper_band_percentile: 通道上轨百分位数。
                lower_band_percentile: 通道下轨百分位数。
            """
            self.price_col = price_col
            self.sth_rp_col = sth_rp_col
            self.smooth_period = smooth_period
            self.rolling_period = rolling_period
            self.upper_band_percentile = upper_band_percentile
            self.lower_band_percentile = lower_band_percentile
            super().__init__(data)

        def _validate_data(self) -> None:
            for col in [self.price_col, self.sth_rp_col]:
                if col not in self.data.columns:
                    raise ValueError(
                        f"Input dataframe is missing required column: {col}"
                    )

        def generate_signals(self) -> None:
            self.signals = self.data.copy()

            self.signals["diff"] = (
                self.signals[self.price_col] - self.signals[self.sth_rp_col]
            )
            self.signals["smooth_diff"] = lowpass_filter(
                self.signals["diff"], self.smooth_period
            )
            self.signals["upper_band"] = (
                self.signals["smooth_diff"]
                .rolling(self.rolling_period)
                .quantile(self.upper_band_percentile)
            )
            self.signals["lower_band"] = (
                self.signals["smooth_diff"]
                .rolling(self.rolling_period)
                .quantile(self.lower_band_percentile)
            )

            signals = np.where(
                self.signals["smooth_diff"] >= self.signals["upper_band"], 1, 0
            )
            signals = np.where(
                self.signals["smooth_diff"] <= self.signals["lower_band"],
                -1,
                signals,
            )
            self.signals["signal"] = signals

        def _add_indicator_traces(self, fig: go.Figure) -> None:
            # 在价格图表添加原始指标
            fig.add_trace(
                go.Scatter(
                    x=self.signals.index,
                    y=self.signals[self.sth_rp_col],
                    name="STH Realized Price",
                    line=dict(color="royalblue", width=2),
                    hoverinfo="x+y",
                ),
                row=1,
                col=1,
            )

            # 添加价格偏离曲线
            fig.add_trace(
                go.Scatter(
                    x=self.signals.index,
                    y=self.signals["diff"],
                    name="Deviation",
                    line=dict(color="#add8e6", width=1.5),
                    opacity=0.5,
                    hoverinfo="x+y",
                ),
                row=2,
                col=1,
            )

            # 添加移动平滑曲线
            fig.add_trace(
                go.Scatter(
                    x=self.signals.index,
                    y=self.signals["smooth_diff"],
                    name="Smooth Deviation",
                    line=dict(color="royalblue", width=2),
                    hoverinfo="x+y",
                ),
                row=2,
                col=1,
            )

            # 添加百分位数通道
            fig.add_trace(
                go.Scatter(
                    x=self.signals.index,
                    y=self.signals["upper_band"],
                    line=dict(color="grey", width=1, dash="dot"),
                    showlegend=False,
                    hoverinfo="skip",
                    mode="lines",
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=self.signals.index,
                    y=self.signals["lower_band"],
                    line=dict(color="grey", width=1, dash="dot"),
                    showlegend=False,
                    hoverinfo="skip",
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(128, 128, 128, 0.1)",
                ),
                row=2,
                col=1,
            )

            # 更新 y 轴设置
            fig.update_yaxes(
                row=2,
                col=1,
                title="Deviation",
                title_font=dict(size=14),
                gridcolor="#e0e0e0",
            )
    return


@app.cell(hide_code=True)
def _(Metric, go, lowpass_filter, np, pd):
    class STHSOPR(Metric):
        """短期持有者支出产出比率(STH-SOPR)指标"""

        @property
        def name(self) -> str:
            return "Short-Term Holder SOPR"

        @property
        def description(self) -> str:
            pass

        def __init__(
            self,
            data: pd.DataFrame,
            price_col: str = "btcusd",
            sopr_col: str = "sth_sopr",
            smooth_period: int = 7,
            rolling_period: int = 200,
            upper_band_percentile: float = 0.95,
            lower_band_percentile: float = 0.05,
        ) -> None:
            """
            初始化 STHSOPR 指标类

            Args:
                data: 包含 STH-SOPR 数据的 DataFrame
                price_col: DataFrame 中表示比特币价格列的名称
                sopr_col: DataFrame 中 STH-SOPR 列的名称
                smooth_period: 移动平滑窗口
                rolling_period: 计算滚动百分位数的窗口
                upper_band_percentile: 计算通道上轨的百分位数
                lower_band_percentile: 计算通道下轨的百分位数
            """
            self.price_col = price_col
            self.sopr_col = sopr_col
            self.smooth_period = smooth_period
            self.rolling_period = rolling_period
            self.upper_band_percentile = upper_band_percentile
            self.lower_band_percentile = lower_band_percentile
            super().__init__(data)

        def _validate_data(self) -> None:
            for col in [self.price_col, self.sopr_col]:
                if col not in self.data.columns:
                    raise ValueError(
                        f"Input dataframe is missing required column: {col}"
                    )

        def generate_signals(self) -> None:
            self.signals = self.data.copy()

            self.signals["smooth_sopr"] = lowpass_filter(
                self.signals[self.sopr_col], self.smooth_period
            )
            self.signals["upper_band"] = (
                self.signals["smooth_sopr"]
                .rolling(self.rolling_period)
                .quantile(self.upper_band_percentile)
            )
            self.signals["lower_band"] = (
                self.signals["smooth_sopr"]
                .rolling(self.rolling_period)
                .quantile(self.lower_band_percentile)
            )

            signals = np.where(
                self.signals["smooth_sopr"] >= self.signals["upper_band"], 1, 0
            )
            signals = np.where(
                self.signals["smooth_sopr"] <= self.signals["lower_band"],
                -1,
                signals,
            )
            self.signals["signal"] = signals

        def _add_indicator_traces(self, fig: go.Figure) -> None:
            # 添加价格偏离曲线
            fig.add_trace(
                go.Scatter(
                    x=self.signals.index,
                    y=self.signals[self.sopr_col],
                    name="STH SOPR",
                    line=dict(color="#add8e6", width=1.5),
                    opacity=0.5,
                    hoverinfo="x+y",
                ),
                row=2,
                col=1,
            )

            # 添加移动平滑曲线
            fig.add_trace(
                go.Scatter(
                    x=self.signals.index,
                    y=self.signals["smooth_sopr"],
                    name="Smoothed STH SOPR",
                    line=dict(color="royalblue", width=2),
                    hoverinfo="x+y",
                ),
                row=2,
                col=1,
            )

            # 添加百分位数通道
            fig.add_trace(
                go.Scatter(
                    x=self.signals.index,
                    y=self.signals["upper_band"],
                    line=dict(color="grey", width=1, dash="dot"),
                    showlegend=False,
                    hoverinfo="skip",
                    mode="lines",
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=self.signals.index,
                    y=self.signals["lower_band"],
                    line=dict(color="grey", width=1, dash="dot"),
                    showlegend=False,
                    hoverinfo="skip",
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(128, 128, 128, 0.1)",
                ),
                row=2,
                col=1,
            )

            # 更新 y 轴设置
            fig.update_yaxes(
                row=2,
                col=1,
                title="Ratio",
                title_font=dict(size=14),
                gridcolor="#e0e0e0",
            )
    return


@app.cell(hide_code=True)
def _(Metric, go, lowpass_filter, np, pd):
    class STHNUPL(Metric):
        """短期持有者未实现盈亏比率(STH-NUPL)指标"""

        @property
        def name(self) -> str:
            return "Short-Term Holder NUPL"

        @property
        def description(self) -> str:
            pass

        def __init__(
            self,
            data: pd.DataFrame,
            price_col: str = "btcusd",
            nupl_col: str = "sth_nupl",
            smooth_period: int = 7,
            rolling_period: int = 200,
            upper_band_percentile: float = 0.95,
            lower_band_percentile: float = 0.05,
        ) -> None:
            """
            初始化 STHNUPL 指标类

            Args:
                data: 包含 STH-NUPL 数据的 DataFrame
                price_col: DataFrame 中表示比特币价格列的名称
                nupl_col: DataFrame 中 STH-NUPL 列的名称
                smooth_period: 平滑窗口。
                rolling_period: 计算滚动百分位数通道的窗口。
                upper_band_percentile: 通道上轨百分位数。
                lower_band_percentile: 通道下轨百分位数。
            """
            self.price_col = price_col
            self.nupl_col = nupl_col
            self.smooth_period = smooth_period
            self.rolling_period = rolling_period
            self.upper_band_percentile = upper_band_percentile
            self.lower_band_percentile = lower_band_percentile
            super().__init__(data)

        def _validate_data(self) -> None:
            for col in [self.price_col, self.nupl_col]:
                if col not in self.data.columns:
                    raise ValueError(
                        f"Input dataframe is missing required column: {col}"
                    )

        def generate_signals(self) -> None:
            self.signals = self.data.copy()

            self.signals["smooth_nupl"] = lowpass_filter(
                self.signals[self.nupl_col], self.smooth_period
            )
            self.signals["upper_band"] = (
                self.signals["smooth_nupl"]
                .rolling(self.rolling_period)
                .quantile(self.upper_band_percentile)
            )
            self.signals["lower_band"] = (
                self.signals["smooth_nupl"]
                .rolling(self.rolling_period)
                .quantile(self.lower_band_percentile)
            )

            signals = np.where(
                self.signals["smooth_nupl"] >= self.signals["upper_band"], 1, 0
            )
            signals = np.where(
                self.signals["smooth_nupl"] <= self.signals["lower_band"],
                -1,
                signals,
            )
            self.signals["signal"] = signals

        def _add_indicator_traces(self, fig: go.Figure) -> None:
            # 添加原始指标曲线
            fig.add_trace(
                go.Scatter(
                    x=self.signals.index,
                    y=self.signals[self.nupl_col],
                    name="STH-NUPL",
                    line=dict(color="#add8e6", width=1.5),
                    opacity=0.5,
                    hoverinfo="x+y",
                ),
                row=2,
                col=1,
            )

            # 添加移动平滑曲线
            fig.add_trace(
                go.Scatter(
                    x=self.signals.index,
                    y=self.signals["smooth_nupl"],
                    name="Smooth STH-NUPL",
                    line=dict(color="royalblue", width=2),
                    hoverinfo="x+y",
                ),
                row=2,
                col=1,
            )

            # 添加百分位数通道
            fig.add_trace(
                go.Scatter(
                    x=self.signals.index,
                    y=self.signals["upper_band"],
                    line=dict(color="grey", width=1, dash="dot"),
                    showlegend=False,
                    hoverinfo="skip",
                    mode="lines",
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=self.signals.index,
                    y=self.signals["lower_band"],
                    line=dict(color="grey", width=1, dash="dot"),
                    showlegend=False,
                    hoverinfo="skip",
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(128, 128, 128, 0.1)",
                ),
                row=2,
                col=1,
            )

            # 更新 y 轴设置
            fig.update_yaxes(
                row=2,
                col=1,
                title="USD",
                title_font=dict(size=14),
                gridcolor="#e0e0e0",
            )
    return


@app.cell(hide_code=True)
def _(Metric, go, lowpass_filter, np, pd):
    class STHMVRV(Metric):
        """短期持有者市值实现值比率(STH-MVRV)指标"""

        @property
        def name(self) -> str:
            return "Short-Term Holder MVRV"

        @property
        def description(self) -> str:
            pass

        def __init__(
            self,
            data: pd.DataFrame,
            price_col: str = "btcusd",
            mvrv_col: str = "sth_mvrv",
            smooth_period: int = 7,
            rolling_period: int = 200,
            upper_band_percentile: float = 0.95,
            lower_band_percentile: float = 0.05,
        ) -> None:
            """
            初始化 STHMVRV 指标类

            Args:
                data: 包含 STH-MVRV 数据的 DataFrame
                price_col: DataFrame 中表示比特币价格列的名称
                mvrv_col: DataFrame 中 STH-MVRV 列的名称
                smooth_period: 平滑窗口。
                rolling_period: 计算滚动百分位数通道的窗口。
                upper_band_percentile: 通道上轨百分位数。
                lower_band_percentile: 通道下轨百分位数。
            """
            self.price_col = price_col
            self.mvrv_col = mvrv_col
            self.smooth_period = smooth_period
            self.rolling_period = rolling_period
            self.upper_band_percentile = upper_band_percentile
            self.lower_band_percentile = lower_band_percentile
            super().__init__(data)

        def _validate_data(self) -> None:
            for col in [self.price_col, self.mvrv_col]:
                if col not in self.data.columns:
                    raise ValueError(
                        f"Input dataframe is missing required column: {col}"
                    )

        def generate_signals(self) -> None:
            self.signals = self.data.copy()

            self.signals["smooth_mvrv"] = lowpass_filter(
                self.signals[self.mvrv_col], self.smooth_period
            )
            self.signals["upper_band"] = (
                self.signals["smooth_mvrv"]
                .rolling(self.rolling_period)
                .quantile(self.upper_band_percentile)
            )
            self.signals["lower_band"] = (
                self.signals["smooth_mvrv"]
                .rolling(self.rolling_period)
                .quantile(self.lower_band_percentile)
            )

            signals = np.where(
                self.signals["smooth_mvrv"] >= self.signals["upper_band"], 1, 0
            )
            signals = np.where(
                self.signals["smooth_mvrv"] <= self.signals["lower_band"],
                -1,
                signals,
            )
            self.signals["signal"] = signals

        def _add_indicator_traces(self, fig: go.Figure) -> None:
            # 添加原始指标曲线
            fig.add_trace(
                go.Scatter(
                    x=self.signals.index,
                    y=self.signals[self.mvrv_col],
                    name="STH-MVRV",
                    line=dict(color="#add8e6", width=1.5),
                    opacity=0.5,
                    hoverinfo="x+y",
                ),
                row=2,
                col=1,
            )

            # 添加移动平滑曲线
            fig.add_trace(
                go.Scatter(
                    x=self.signals.index,
                    y=self.signals["smooth_mvrv"],
                    name="Smooth STH-MVRV",
                    line=dict(color="royalblue", width=2),
                    hoverinfo="x+y",
                ),
                row=2,
                col=1,
            )

            # 添加百分位数通道
            fig.add_trace(
                go.Scatter(
                    x=self.signals.index,
                    y=self.signals["upper_band"],
                    line=dict(color="grey", width=1, dash="dot"),
                    showlegend=False,
                    hoverinfo="skip",
                    mode="lines",
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=self.signals.index,
                    y=self.signals["lower_band"],
                    line=dict(color="grey", width=1, dash="dot"),
                    showlegend=False,
                    hoverinfo="skip",
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(128, 128, 128, 0.1)",
                ),
                row=2,
                col=1,
            )

            # 更新 y 轴设置
            fig.update_yaxes(
                row=2,
                col=1,
                title="USD",
                title_font=dict(size=14),
                gridcolor="#e0e0e0",
            )
    return (STHMVRV,)


@app.cell
def _(STHMVRV, data):
    # metric = STHRealizedPrice(
    #     data,
    #     rolling_period=200,
    #     upper_band_percentile=0.99,
    #     lower_band_percentile=0.01,
    # )
    # metric.generate_signals()
    # fig = metric.generate_chart()

    # metric = STHSOPR(data)
    # metric.generate_signals()
    # fig = metric.generate_chart()

    # metric = STHNUPL(data, upper_band_percentile=0.95, lower_band_percentile=0.05)
    # metric.generate_signals()
    # fig = metric.generate_chart()

    metric = STHMVRV(data, upper_band_percentile=0.95, lower_band_percentile=0.05)
    metric.generate_signals()
    fig = metric.generate_chart()

    fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
