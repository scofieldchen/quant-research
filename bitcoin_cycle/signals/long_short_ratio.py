import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .base import Metric
from .indicators import lowpass_filter


class LongShortRatioAccount(Metric):
    """顶级交易员的多空比例（账户）"""

    @property
    def name(self) -> str:
        return "Toptrader Long Short Ratio(Account)"

    @property
    def description(self) -> str:
        pass

    def __init__(
        self,
        data: pd.DataFrame,
        price_col: str = "btcusd",
        metric_col: str = "toptrader_long_short_ratio_account",
        smooth_period: int = 10,
        rolling_period: int = 200,
        lower_band_percentile: float = 0.05,
        upper_band_percentile: float = 0.95,
    ) -> None:
        self.price_col = price_col
        self.metric_col = metric_col
        self.smooth_period = smooth_period
        self.rolling_period = rolling_period
        self.lower_band_percentile = lower_band_percentile
        self.upper_band_percentile = upper_band_percentile
        super().__init__(data)

    def _validate_data(self) -> None:
        for col in [self.price_col, self.metric_col]:
            if col not in self.data.columns:
                raise ValueError(f"Input dataframe is missing required column: {col}")

    def generate_signals(self) -> None:
        self.signals = self.data.copy()

        self.signals["smooth_ratio"] = lowpass_filter(
            self.signals[self.metric_col], self.smooth_period
        )
        self.signals["upper_band"] = (
            self.signals["smooth_ratio"]
            .rolling(self.rolling_period)
            .quantile(self.upper_band_percentile)
        )
        self.signals["lower_band"] = (
            self.signals["smooth_ratio"]
            .rolling(self.rolling_period)
            .quantile(self.lower_band_percentile)
        )

        signals = np.where(
            self.signals["smooth_ratio"] <= self.signals["lower_band"], 1, 0
        )
        signals = np.where(
            self.signals["smooth_ratio"] >= self.signals["upper_band"],
            -1,
            signals,
        )
        self.signals["signal"] = signals

    def _add_indicator_traces(self, fig: go.Figure) -> None:
        # 添加多空比例曲线
        fig.add_trace(
            go.Scatter(
                x=self.signals.index,
                y=self.signals[self.metric_col],
                name="Raw Ratio",
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
                y=self.signals["smooth_ratio"],
                name="Smooth Ratio",
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
                fillcolor="rgba(128, 128, 128, 0.1)",  # 填充颜色和透明度 (浅灰色)
            ),
            row=2,
            col=1,
        )

        # 更新 y 轴设置
        fig.update_yaxes(
            row=2,
            col=1,
            autorange="reversed",
            title="Ratio(reversed)",
            title_font=dict(size=14),
            gridcolor="#e0e0e0",
        )


class LongShortRatioPosition(Metric):
    """顶级交易员的多空比例（仓位价值）"""

    @property
    def name(self) -> str:
        return "Toptrader Long Short Ratio(Position)"

    @property
    def description(self) -> str:
        pass

    def __init__(
        self,
        data: pd.DataFrame,
        price_col: str = "btcusd",
        metric_col: str = "toptrader_long_short_ratio_position",
        smooth_period: int = 10,
        rolling_period: int = 200,
        lower_band_percentile: float = 0.05,
        upper_band_percentile: float = 0.95,
    ) -> None:
        self.price_col = price_col
        self.metric_col = metric_col
        self.smooth_period = smooth_period
        self.rolling_period = rolling_period
        self.lower_band_percentile = lower_band_percentile
        self.upper_band_percentile = upper_band_percentile
        super().__init__(data)

    def _validate_data(self) -> None:
        for col in [self.price_col, self.metric_col]:
            if col not in self.data.columns:
                raise ValueError(f"Input dataframe is missing required column: {col}")

    def generate_signals(self) -> None:
        self.signals = self.data.copy()

        self.signals["smooth_ratio"] = lowpass_filter(
            self.signals[self.metric_col], self.smooth_period
        )

        self.signals["upper_band"] = (
            self.signals["smooth_ratio"]
            .rolling(self.rolling_period)
            .quantile(self.upper_band_percentile)
        )
        self.signals["lower_band"] = (
            self.signals["smooth_ratio"]
            .rolling(self.rolling_period)
            .quantile(self.lower_band_percentile)
        )

        signals = np.where(
            self.signals["smooth_ratio"] >= self.signals["upper_band"], 1, 0
        )
        signals = np.where(
            self.signals["smooth_ratio"] <= self.signals["lower_band"],
            -1,
            signals,
        )
        self.signals["signal"] = signals

    def _add_indicator_traces(self, fig: go.Figure) -> None:
        # 添加多空比例曲线
        fig.add_trace(
            go.Scatter(
                x=self.signals.index,
                y=self.signals[self.metric_col],
                name="Raw Ratio",
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
                y=self.signals["smooth_ratio"],
                name="Smooth Ratio",
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
                fillcolor="rgba(128, 128, 128, 0.1)",  # 填充颜色和透明度 (浅灰色)
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


class LongShortRatio(Metric):
    """所有交易员的多空比例（账户）"""

    @property
    def name(self) -> str:
        return "Long Short Ratio"

    @property
    def description(self) -> str:
        pass

    def __init__(
        self,
        data: pd.DataFrame,
        price_col: str = "btcusd",
        metric_col: str = "long_short_ratio",
        smooth_period: int = 10,
        rolling_period: int = 200,
        lower_band_percentile: float = 0.05,
        upper_band_percentile: float = 0.95,
    ) -> None:
        self.price_col = price_col
        self.metric_col = metric_col
        self.smooth_period = smooth_period
        self.rolling_period = rolling_period
        self.lower_band_percentile = lower_band_percentile
        self.upper_band_percentile = upper_band_percentile
        super().__init__(data)

    def _validate_data(self) -> None:
        for col in [self.price_col, self.metric_col]:
            if col not in self.data.columns:
                raise ValueError(f"Input dataframe is missing required column: {col}")

    def generate_signals(self) -> None:
        self.signals = self.data.copy()

        self.signals["smooth_ratio"] = lowpass_filter(
            self.signals[self.metric_col], self.smooth_period
        )
        self.signals["upper_band"] = (
            self.signals["smooth_ratio"]
            .rolling(self.rolling_period)
            .quantile(self.upper_band_percentile)
        )
        self.signals["lower_band"] = (
            self.signals["smooth_ratio"]
            .rolling(self.rolling_period)
            .quantile(self.lower_band_percentile)
        )

        signals = np.where(
            self.signals["smooth_ratio"] <= self.signals["lower_band"], 1, 0
        )
        signals = np.where(
            self.signals["smooth_ratio"] >= self.signals["upper_band"],
            -1,
            signals,
        )
        self.signals["signal"] = signals

    def _add_indicator_traces(self, fig: go.Figure) -> None:
        # 添加多空比例曲线
        fig.add_trace(
            go.Scatter(
                x=self.signals.index,
                y=self.signals[self.metric_col],
                name="Raw Ratio",
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
                y=self.signals["smooth_ratio"],
                name="Smooth Ratio",
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
                fillcolor="rgba(128, 128, 128, 0.1)",  # 填充颜色和透明度 (浅灰色)
            ),
            row=2,
            col=1,
        )

        # 更新 y 轴设置
        fig.update_yaxes(
            row=2,
            col=1,
            autorange="reversed",
            title="Ratio(reversed)",
            title_font=dict(size=14),
            gridcolor="#e0e0e0",
        )
