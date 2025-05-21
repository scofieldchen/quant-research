from typing import List, Union

import pandas as pd
import plotly.graph_objects as go

from .base import Metric
from .utils import add_percentile_bands, calculate_percentile_bands


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
        metric_cols: Union[str, List[str]] = "nrpl",
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
            metric_cols: DataFrame 中 NRPL 列的名称
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

        # 使用工具函数计算百分位数通道
        metric_col = self.metric_cols[0]  # 使用第一个指标列
        self.signals = calculate_percentile_bands(
            data=data,
            input_col=metric_col,
            smooth_period=self.smooth_period,
            rolling_period=self.rolling_period,
            upper_band_percentile=self.upper_band_percentile,
            lower_band_percentile=self.lower_band_percentile,
        )

    def _add_indicator_traces(self, fig: go.Figure) -> None:
        # 添加指标和百分位数通道
        metric_col = self.metric_cols[0]
        add_percentile_bands(
            fig=fig,
            data=self.signals,
            metric_col=metric_col,
            smooth_metric_col="smooth_" + metric_col,
            yaxis_title="NRPL",
            metric_name="NRPL",
            smooth_metric_name="Smooth NRPL",
        )
