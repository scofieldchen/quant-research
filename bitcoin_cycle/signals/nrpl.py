import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .base import Metric
from .indicators import lowpass_filter


class NRPL(Metric):
    """净实现盈亏比率(NRPL)指标"""

    @property
    def name(self) -> str:
        return "Net Realized Profit Loss"

    @property
    def description(self) -> str:
        pass

    def __init__(
        self,
        data: pd.DataFrame,
        price_col: str = "btcusd",
        nrpl_col: str = "nrpl",
        smooth_period: int = 7,
        rolling_period: int = 200,
        upper_band_percentile: float = 0.95,
        lower_band_percentile: float = 0.05,
    ) -> None:
        """
        初始化 NRPL 指标类

        Args:
            data: 包含 NRPL 数据的 DataFrame
            price_col: DataFrame 中表示比特币价格列的名称
            nrpl_col: DataFrame 中 NRPL 列的名称
            smooth_period: 平滑窗口。
            rolling_period: 计算滚动百分位数通道的窗口。
            upper_band_percentile: 通道上轨百分位数。
            lower_band_percentile: 通道下轨百分位数。
        """
        self.price_col = price_col
        self.nrpl_col = nrpl_col
        self.smooth_period = smooth_period
        self.rolling_period = rolling_period
        self.upper_band_percentile = upper_band_percentile
        self.lower_band_percentile = lower_band_percentile
        super().__init__(data)

    def _validate_data(self) -> None:
        for col in [self.price_col, self.nrpl_col]:
            if col not in self.data.columns:
                raise ValueError(f"Input dataframe is missing required column: {col}")

    def generate_signals(self) -> None:
        self.signals = self.data.copy()

        self.signals["smooth_nrpl"] = lowpass_filter(
            self.signals[self.nrpl_col], self.smooth_period
        )
        self.signals["upper_band"] = (
            self.signals["smooth_nrpl"]
            .rolling(self.rolling_period)
            .quantile(self.upper_band_percentile)
        )
        self.signals["lower_band"] = (
            self.signals["smooth_nrpl"]
            .rolling(self.rolling_period)
            .quantile(self.lower_band_percentile)
        )

        signals = np.where(
            self.signals["smooth_nrpl"] >= self.signals["upper_band"], 1, 0
        )
        signals = np.where(
            self.signals["smooth_nrpl"] <= self.signals["lower_band"],
            -1,
            signals,
        )
        self.signals["signal"] = signals

    def _add_indicator_traces(self, fig: go.Figure) -> None:
        # 添加原始指标曲线
        fig.add_trace(
            go.Scatter(
                x=self.signals.index,
                y=self.signals[self.nrpl_col],
                name="NRPL",
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
                y=self.signals["smooth_nrpl"],
                name="Smooth NRPL",
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
