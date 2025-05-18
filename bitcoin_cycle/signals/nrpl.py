import talib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .base import Metric


class NRPL(Metric):
    """净实现盈亏比率(NRPL)指标"""

    @property
    def name(self) -> str:
        return "Net Realized Profit Loss"

    @property
    def description(self) -> str:
        return "使用布林带分析净实现盈亏(NRPL)，识别潜在的市场顶部和底部。"

    def __init__(
        self,
        data: pd.DataFrame,
        price_col: str = "btcusd",
        nrpl_col: str = "nrpl",
        bband_period: int = 200,
        bband_upper_std: float = 2.0,
        bband_lower_std: float = 2.0,
    ) -> None:
        """
        初始化 NRPL 指标类

        Args:
            data: 包含 NRPL 数据的 DataFrame
            price_col: DataFrame 中表示比特币价格列的名称
            nrpl_col: DataFrame 中 NRPL 列的名称
            bband_period: 布林带计算周期
            bband_upper_std: 上轨标准差乘数
            bband_lower_std: 下轨标准差乘数
        """
        self.price_col = price_col
        self.nrpl_col = nrpl_col
        self.bband_period = bband_period
        self.bband_upper_std = bband_upper_std
        self.bband_lower_std = bband_lower_std
        super().__init__(data)

    def _validate_data(self) -> None:
        for col in [self.price_col, self.nrpl_col]:
            if col not in self.data.columns:
                raise ValueError(f"Input dataframe is missing required column: {col}")

    def generate_signals(self) -> None:
        self.signals = self.data.copy()

        # 计算布林带
        bband_upper, _, bband_lower = talib.BBANDS(
            self.signals[self.nrpl_col],
            self.bband_period,
            self.bband_upper_std,
            self.bband_lower_std,
        )

        self.signals["upper_band"] = bband_upper
        self.signals["lower_band"] = bband_lower

        # 生成信号
        signals = np.where(self.signals[self.nrpl_col] > bband_upper, 1, 0)
        signals = np.where(self.signals[self.nrpl_col] < bband_lower, -1, signals)
        self.signals["signal"] = signals

    def _add_indicator_traces(self, fig: go.Figure) -> None:
        # 添加 NRPL 曲线
        fig.add_trace(
            go.Scatter(
                x=self.signals.index,
                y=self.signals[self.nrpl_col],
                name="NRPL",
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
