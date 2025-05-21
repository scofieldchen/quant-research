from typing import List, Union

import pandas as pd
import plotly.graph_objects as go

from .base import Metric
from .utils import add_percentile_bands, calculate_percentile_bands


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
        metric_cols: Union[str, List[str]] = "sth_realized_price",
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
            metric_cols: DataFrame 中 STH 实现价格列的名称。
            smooth_period: 平滑窗口。
            rolling_period: 计算滚动百分位数通道的窗口。
            upper_band_percentile: 通道上轨百分位数。
            lower_band_percentile: 通道下轨百分位数。
        """
        self.smooth_period = smooth_period
        self.rolling_period = rolling_period
        self.upper_band_percentile = upper_band_percentile
        self.lower_band_percentile = lower_band_percentile
        super().__init__(data, price_col, metric_cols)

    def generate_signals(self) -> None:
        data = self.data.copy()

        # 计算价格与实现价格的差值
        metric_col = self.metric_cols[0]  # 使用第一个指标列
        data["diff"] = data[self.price_col] - data[metric_col]

        # 使用工具函数计算百分位数通道
        self.signals = calculate_percentile_bands(
            data=data,
            input_col="diff",
            smooth_period=self.smooth_period,
            rolling_period=self.rolling_period,
            upper_band_percentile=self.upper_band_percentile,
            lower_band_percentile=self.lower_band_percentile,
        )

        # 在价格图表中添加实现价格曲线
        self.signals[metric_col] = data[metric_col]

    def _add_indicator_traces(self, fig: go.Figure) -> None:
        # 在价格图表添加原始指标
        metric_col = self.metric_cols[0]
        fig.add_trace(
            go.Scatter(
                x=self.signals.index,
                y=self.signals[metric_col],
                name="STH Realized Price",
                line=dict(color="royalblue", width=2),
                hovertemplate="<b>Date</b>: %{x}<br>"
                + "<b>Value</b>: $%{y:,.0f}<br><extra></extra>",
            ),
            row=1,
            col=1,
        )

        # 添加价格偏离指标和百分位数通道
        add_percentile_bands(
            fig=fig,
            data=self.signals,
            metric_col="diff",
            smooth_metric_col="smooth_diff",
            yaxis_title="Price - STH Realized Price",
            metric_name="Deviation",
            smooth_metric_name="Smooth Deviation",
        )
