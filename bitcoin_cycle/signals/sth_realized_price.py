import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .base import Metric
from .indicators import lowpass_filter


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
                raise ValueError(f"Input dataframe is missing required column: {col}")

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
