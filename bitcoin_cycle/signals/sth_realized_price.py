import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .base import Metric
from .indicators import fisher_transform


class STHRealizedPrice(Metric):
    """短期持有者的实现价格"""

    @property
    def name(self) -> str:
        return "Short-Term Holder Realized Price"

    @property
    def description(self) -> str:
        return (
            "比较比特币价格与其短期持有者实现价格，以识别潜在的市场顶部和底部。",
            "使用 Fisher 变换对价格差异进行归一化处理。",
        )

    def __init__(
        self,
        data: pd.DataFrame,
        price_col: str = "btcusd",
        sth_rp_col: str = "sth_realized_price",
        period: int = 200,
        threshold: float = 2.0,
    ) -> None:
        """
        初始化 STHRealizedPrice 指标类

        Args:
            data: 包含价格和 STH 实现价格列的 DataFrame。
            price_col: DataFrame 中比特币价格列的名称。
            sth_rp_col: DataFrame 中 STH 实现价格列的名称。
            period: 费舍尔转换的回溯期。
            threshold: 计算峰值/谷值信号的费舍尔转换阈值。
        """
        self.price_col = price_col
        self.sth_rp_col = sth_rp_col
        self.period = period
        self.threshold = threshold
        super().__init__(data, chart_rows=2)

    def _validate_data(self) -> None:
        for col in [self.price_col, self.sth_rp_col]:
            if col not in self.data.columns:
                raise ValueError(f"Input dataframe is missing required column: {col}")

    def generate_signals(self) -> None:
        self.signals = self.data.copy()
        diff = self.signals[self.price_col] - self.signals[self.sth_rp_col]
        normalized_diff = fisher_transform(diff, self.period)
        signals = np.where(normalized_diff > self.threshold, 1, 0)
        signals = np.where(normalized_diff < -self.threshold, -1, signals)
        self.signals["normalized_diff"] = normalized_diff
        self.signals["signal"] = signals

    def _add_indicator_traces(self, fig: go.Figure) -> None:
        # 在第一行（价格图表）添加原始指标
        fig.add_trace(
            go.Scatter(
                x=self.signals.index,
                y=self.signals[self.sth_rp_col],
                name="STH Realized Price",
            ),
            row=1,
            col=1,
        )

        # 第二行添加标准化指标
        fig.add_trace(
            go.Scatter(
                x=self.signals.index,
                y=self.signals["normalized_diff"],
                name="Normalized Price Diff",
            ),
            row=2,
            col=1,
        )

        for level in [-self.threshold, self.threshold]:
            fig.add_hline(
                y=level,
                row=2,
                col=1,
                line_dash="dash",
                line_color="grey",
                line_width=0.8,
            )
