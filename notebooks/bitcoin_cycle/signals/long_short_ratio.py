from typing import List, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .base import Metric
from .indicators import lowpass_filter
from .utils import add_percentile_bands, calculate_percentile_bands


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
        metric_cols: Union[str, List[str]] = "toptrader_long_short_ratio_account",
        smooth_period: int = 10,
        rolling_period: int = 200,
        lower_band_percentile: float = 0.05,
        upper_band_percentile: float = 0.95,
    ) -> None:
        """
        初始化顶级交易员多空比例指标类

        Args:
            data: 包含多空比例数据的 DataFrame
            price_col: DataFrame 中表示比特币价格列的名称
            metric_cols: DataFrame 中多空比例列的名称
            smooth_period: 移动平滑窗口
            rolling_period: 计算滚动百分位数的窗口
            lower_band_percentile: 计算通道下轨的百分位数
            upper_band_percentile: 计算通道上轨的百分位数
        """
        self.smooth_period = smooth_period
        self.rolling_period = rolling_period
        self.lower_band_percentile = lower_band_percentile
        self.upper_band_percentile = upper_band_percentile
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

        # 反转信号，因为较低的比率表示看涨，较高的比率表示看跌
        self.signals["signal"] = -self.signals["signal"]

    def _add_indicator_traces(self, fig: go.Figure) -> None:
        # 添加指标和百分位数通道
        metric_col = self.metric_cols[0]
        add_percentile_bands(
            fig=fig,
            data=self.signals,
            metric_col=metric_col,
            smooth_metric_col="smooth_" + metric_col,
            yaxis_title="Ratio(reversed)",
            metric_name="Raw Ratio",
            smooth_metric_name="Smooth Ratio",
        )

        # 设置y轴反转
        fig.update_yaxes(
            row=2,
            col=1,
            autorange="reversed",
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
        metric_cols: Union[str, List[str]] = "toptrader_long_short_ratio_position",
        smooth_period: int = 10,
        rolling_period: int = 200,
        lower_band_percentile: float = 0.05,
        upper_band_percentile: float = 0.95,
    ) -> None:
        """
        初始化顶级交易员多空比例（仓位）指标类

        Args:
            data: 包含多空比例数据的 DataFrame
            price_col: DataFrame 中表示比特币价格列的名称
            metric_cols: DataFrame 中多空比例列的名称
            smooth_period: 移动平滑窗口
            rolling_period: 计算滚动百分位数的窗口
            lower_band_percentile: 计算通道下轨的百分位数
            upper_band_percentile: 计算通道上轨的百分位数
        """
        self.smooth_period = smooth_period
        self.rolling_period = rolling_period
        self.lower_band_percentile = lower_band_percentile
        self.upper_band_percentile = upper_band_percentile
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
            yaxis_title="Ratio",
            metric_name="Raw Ratio",
            smooth_metric_name="Smooth Ratio",
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
        metric_cols: Union[str, List[str]] = "long_short_ratio",
        smooth_period: int = 10,
        rolling_period: int = 200,
        lower_band_percentile: float = 0.05,
        upper_band_percentile: float = 0.95,
    ) -> None:
        """
        初始化所有交易员多空比例指标类

        Args:
            data: 包含多空比例数据的 DataFrame
            price_col: DataFrame 中表示比特币价格列的名称
            metric_cols: DataFrame 中多空比例列的名称
            smooth_period: 移动平滑窗口
            rolling_period: 计算滚动百分位数的窗口
            lower_band_percentile: 计算通道下轨的百分位数
            upper_band_percentile: 计算通道上轨的百分位数
        """
        self.smooth_period = smooth_period
        self.rolling_period = rolling_period
        self.lower_band_percentile = lower_band_percentile
        self.upper_band_percentile = upper_band_percentile
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

        # 反转信号，因为较低的比率表示看涨，较高的比率表示看跌
        self.signals["signal"] = -self.signals["signal"]

    def _add_indicator_traces(self, fig: go.Figure) -> None:
        # 添加指标和百分位数通道
        metric_col = self.metric_cols[0]
        add_percentile_bands(
            fig=fig,
            data=self.signals,
            metric_col=metric_col,
            smooth_metric_col="smooth_" + metric_col,
            yaxis_title="Ratio(reversed)",
            metric_name="Raw Ratio",
            smooth_metric_name="Smooth Ratio",
        )

        # 设置y轴反转
        fig.update_yaxes(
            row=2,
            col=1,
            autorange="reversed",
        )
