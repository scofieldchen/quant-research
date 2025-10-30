import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .base import Metric
from .indicators import lowpass_filter


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
        metric_cols: str = "fgi",
        smooth_period: int = 10,
        extreme_greed_threshold: float = 80.0,
        extreme_fear_threshold: float = 20.0,
    ) -> None:
        """
        初始化 FearGreedIndex 指标类

        Args:
            data: 包含数据的 DataFrame
            price_col: DataFrame 中表示比特币价格列的名称
            metric_cols: DataFrame 中表示恐慌贪婪指数列的名称
        """
        self.smooth_period = smooth_period
        self.extreme_greed_threshold = extreme_greed_threshold
        self.extreme_fear_threshold = extreme_fear_threshold
        super().__init__(data, price_col, metric_cols)

    def generate_signals(self) -> None:
        self.signals = self.data.copy()
        metric_col = self.metric_cols[0]
        self.signals["smooth_fgi"] = lowpass_filter(
            self.signals[metric_col], self.smooth_period
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
        metric_col = self.metric_cols[0]
        fig.add_trace(
            go.Scatter(
                x=self.signals.index,
                y=self.signals[metric_col],
                name="Fear greed index",
                line=dict(color="lightblue", width=1.5),
                opacity=0.5,
                hovertemplate="<b>Date</b>: %{x}<br>"
                + "<b>Value</b>: %{y:.1f}<br><extra></extra>",
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
                hovertemplate="<b>Date</b>: %{x}<br>"
                + "<b>Value</b>: %{y:.1f}<br><extra></extra>",
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
        metric_cols: str = "sentiment",
        extreme_level: int = 40,
    ) -> None:
        """
        初始化 ConsecutiveGreedDays 指标类

        Args:
            data: 包含数据的 DataFrame
            price_col: DataFrame 中表示比特币价格列的名称
            metric_cols: DataFrame 中表示市场情绪状态的名称
        """
        self.extreme_level = extreme_level
        super().__init__(data, price_col, metric_cols)

    def generate_signals(self) -> None:
        self.signals = self.data.copy()
        greed_days = np.zeros(len(self.signals), int)

        metric_col = self.metric_cols[0]
        for i in range(1, len(greed_days)):
            if self.signals[metric_col].iloc[i] in ["Greed", "Extreme Greed"]:
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
