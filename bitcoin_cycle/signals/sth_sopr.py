import talib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .base import Metric


class STHSOPR(Metric):
    """短期持有者支出产出比率(STH-SOPR)指标"""

    @property
    def name(self) -> str:
        return "Short-Term Holder SOPR"

    @property
    def description(self) -> str:
        return (
            "使用布林带分析短期持有者支出产出比率(STH-SOPR)，识别潜在的市场顶部和底部。"
        )

    def __init__(
        self,
        data: pd.DataFrame,
        price_col: str = "btcusd",
        sopr_col: str = "sth_sopr",
        bband_period: int = 200,
        bband_upper_std: float = 2.0,
        bband_lower_std: float = 1.5,
    ) -> None:
        """
        初始化 STHSOPR 指标类

        Args:
            data: 包含 STH-SOPR 数据的 DataFrame
            price_col: DataFrame 中表示比特币价格列的名称
            sopr_col: DataFrame 中 STH-SOPR 列的名称
            bband_period: 布林带计算周期
            bband_upper_std: 上轨标准差乘数
            bband_lower_std: 下轨标准差乘数
        """
        self.price_col = price_col
        self.sopr_col = sopr_col
        self.bband_period = bband_period
        self.bband_upper_std = bband_upper_std
        self.bband_lower_std = bband_lower_std
        super().__init__(data)

    def _validate_data(self) -> None:
        for col in [self.price_col, self.sopr_col]:
            if col not in self.data.columns:
                raise ValueError(f"Input dataframe is missing required column: {col}")

    def generate_signals(self) -> None:
        self.signals = self.data.copy()

        # 计算布林带
        bband_upper, _, bband_lower = talib.BBANDS(
            self.signals[self.sopr_col],
            self.bband_period,
            self.bband_upper_std,
            self.bband_lower_std,
        )

        self.signals["upper_band"] = bband_upper
        self.signals["lower_band"] = bband_lower

        # 生成信号
        signals = np.where(self.signals[self.sopr_col] > bband_upper, 1, 0)
        signals = np.where(self.signals[self.sopr_col] < bband_lower, -1, signals)
        self.signals["signal"] = signals

    def _add_indicator_traces(self, fig: go.Figure) -> None:
        # 添加 SOPR 曲线
        fig.add_trace(
            go.Scatter(
                x=self.signals.index,
                y=self.signals[self.sopr_col],
                name="STH-SOPR",
            ),
            row=2,
            col=1,
        )

        # 添加布林带
        fig.add_trace(
            go.Scatter(
                x=self.signals.index,
                y=self.signals["upper_band"],
                line=dict(color="gray", dash="dash"),
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=self.signals.index,
                y=self.signals["lower_band"],
                line=dict(color="gray", dash="dash"),
            ),
            row=2,
            col=1,
        )
