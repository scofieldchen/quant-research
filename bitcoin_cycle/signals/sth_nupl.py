import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .base import Metric
from .indicators import fisher_transform


class STHNUPL(Metric):
    """短期持有者未实现盈亏比率(STH-NUPL)指标"""

    @property
    def name(self) -> str:
        return "Short-Term Holder NUPL"

    @property
    def description(self) -> str:
        return "使用 Fisher 变换分析短期持有者未实现盈亏比率(STH-NUPL)，识别潜在的市场顶部和底部。"

    def __init__(
        self,
        data: pd.DataFrame,
        price_col: str = "btcusd",
        nupl_col: str = "sth_nupl",
        smooth_period: int = 10,
        fisher_period: int = 200,
        threshold: float = 2.0,
    ) -> None:
        """
        初始化 STHNUPL 指标类

        Args:
            data: 包含 STH-NUPL 数据的 DataFrame
            price_col: DataFrame 中表示比特币价格列的名称
            nupl_col: DataFrame 中 STH-NUPL 列的名称
            smooth_period: 移动平均平滑周期
            fisher_period: Fisher 变换的计算周期
            threshold: 计算峰值/谷值信号的阈值
        """
        self.price_col = price_col
        self.nupl_col = nupl_col
        self.smooth_period = smooth_period
        self.fisher_period = fisher_period
        self.threshold = threshold
        super().__init__(data)

    def _validate_data(self) -> None:
        for col in [self.price_col, self.nupl_col]:
            if col not in self.data.columns:
                raise ValueError(f"Input dataframe is missing required column: {col}")

    def generate_signals(self) -> None:
        self.signals = self.data.copy()

        # 计算平滑的 NUPL
        smoothed = (
            self.signals[self.nupl_col]
            .rolling(self.smooth_period, min_periods=1)
            .mean()
        )

        # 计算 Fisher 变换
        normalized = fisher_transform(smoothed, self.fisher_period)
        self.signals["normalized_nupl"] = normalized

        # 生成信号
        signals = np.where(normalized > self.threshold, 1, 0)
        signals = np.where(normalized < -self.threshold, -1, signals)
        self.signals["signal"] = signals

    def _add_indicator_traces(self, fig: go.Figure) -> None:
        # 添加 NUPL 曲线
        fig.add_trace(
            go.Scatter(
                x=self.signals.index,
                y=self.signals["normalized_nupl"],
                name="Normalized STH-NUPL",
            ),
            row=2,
            col=1,
        )

        # 添加阈值线
        for level in [-self.threshold, self.threshold]:
            fig.add_hline(
                y=level,
                row=2,
                col=1,
                line_dash="dash",
                line_color="grey",
                line_width=0.8,
            )
