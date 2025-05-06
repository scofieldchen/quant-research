from abc import ABC, abstractmethod
from typing import Any, List, Type

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import talib
from plotly.subplots import make_subplots
from utils import find_trend_periods, fisher_transform


class Metric(ABC):
    """代表指标的抽象基类"""

    def __init__(self, data: pd.DataFrame, **kwargs: Any) -> None:
        """初始化指标"""
        self.data = data
        self._validate_data()
        self.signals: pd.DataFrame | None = None

    @property
    @abstractmethod
    def name(self) -> str:
        """指标的通用名称"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """指标的简短描述"""
        pass

    @abstractmethod
    def _validate_data(self) -> None:
        """验证输入数据是否有效，由子类实现"""
        pass

    @abstractmethod
    def generate_signals(self) -> None:
        """计算交易信号，由子类实现"""
        pass

    @abstractmethod
    def generate_chart(self) -> go.Figure:
        """数据可视化，由子类实现"""
        pass


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
        super().__init__(data)

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

    def generate_chart(self) -> go.Figure:
        if self.signals is None:
            self.generate_signals()

        # 创建图表
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                "<b>Bitcoin price vs STH Realized price</b>",
                "<b>Normalized price diff</b>",
            ),
            row_heights=[0.7, 0.3],
        )

        # 添加价格曲线
        fig.add_trace(
            go.Scatter(x=self.signals.index, y=self.signals[self.price_col]),
            row=1,
            col=1,
        )

        # 添加指标曲线
        fig.add_trace(
            go.Scatter(x=self.signals.index, y=self.signals[self.sth_rp_col]),
            row=1,
            col=1,
        )

        # 添加极值区域背景
        peak_periods = find_trend_periods(self.signals["signal"] == 1)
        valley_periods = find_trend_periods(self.signals["signal"] == -1)

        for x0, x1 in peak_periods:
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor="#FF6B6B",
                opacity=0.2,
                line_width=0,
                row=1,
                col=1,
            )

        for x0, x1 in valley_periods:
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor="#38A169",
                opacity=0.2,
                line_width=0,
                row=1,
                col=1,
            )

        # 添加标准化指标
        fig.add_trace(
            go.Scatter(x=self.signals.index, y=self.signals["normalized_diff"]),
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

        # 更新图表
        fig.update_layout(
            title=f"</b>{self.name}</b>",
            width=1000,
            height=700,
            template="plotly_white",
            showlegend=False,
        )

        return fig


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

    def generate_chart(self) -> go.Figure:
        if self.signals is None:
            self.generate_signals()

        # 创建图表
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                "<b>Bitcoin price</b>",
                "<b>STH-SOPR with Bollinger Bands</b>",
            ),
            row_heights=[0.7, 0.3],
        )

        # 添加比特币价格曲线
        fig.add_trace(
            go.Scatter(x=self.signals.index, y=self.signals[self.price_col]),
            row=1,
            col=1,
        )

        # 添加 SOPR 曲线
        fig.add_trace(
            go.Scatter(
                x=self.signals.index,
                y=self.signals[self.sopr_col],
                name="STH-SOPR",
                line=dict(color="#1f77b4"),
            ),
            row=2,
            col=1,
        )

        # 添加布林带
        fig.add_trace(
            go.Scatter(
                x=self.signals.index,
                y=self.signals["upper_band"],
                name="Upper Band",
                line=dict(color="gray", dash="dash"),
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=self.signals.index,
                y=self.signals["lower_band"],
                name="Lower Band",
                line=dict(color="gray", dash="dash"),
            ),
            row=2,
            col=1,
        )

        # 添加极值区域背景
        peak_periods = find_trend_periods(self.signals["signal"] == 1)
        valley_periods = find_trend_periods(self.signals["signal"] == -1)

        for x0, x1 in peak_periods:
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor="#FF6B6B",
                opacity=0.2,
                line_width=0,
                row=1,
                col=1,
            )

        for x0, x1 in valley_periods:
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor="#38A169",
                opacity=0.2,
                line_width=0,
                row=1,
                col=1,
            )

        # 更新图表
        fig.update_layout(
            title=f"</b>{self.name}</b>",
            width=1000,
            height=700,
            template="plotly_white",
            showlegend=False,
        )

        return fig


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

    def generate_chart(self) -> go.Figure:
        if self.signals is None:
            self.generate_signals()

        # 创建图表
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                "<b>Bitcoin price</b>",
                "<b>STH-NUPL with Fisher Transform</b>",
            ),
            row_heights=[0.7, 0.3],
        )

        # 添加比特币价格曲线
        fig.add_trace(
            go.Scatter(x=self.signals.index, y=self.signals[self.price_col]),
            row=1,
            col=1,
        )

        # 添加 NUPL 曲线
        fig.add_trace(
            go.Scatter(
                x=self.signals.index,
                y=self.signals["normalized_nupl"],
                name="Normalized STH-NUPL",
                line=dict(color="#1f77b4"),
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

        # 添加极值区域背景
        peak_periods = find_trend_periods(self.signals["signal"] == 1)
        valley_periods = find_trend_periods(self.signals["signal"] == -1)

        for x0, x1 in peak_periods:
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor="#FF6B6B",
                opacity=0.2,
                line_width=0,
                row=1,
                col=1,
            )

        for x0, x1 in valley_periods:
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor="#38A169",
                opacity=0.2,
                line_width=0,
                row=1,
                col=1,
            )

        # 更新图表
        fig.update_layout(
            title=f"</b>{self.name}</b>",
            width=1000,
            height=700,
            template="plotly_white",
            showlegend=False,
        )

        return fig


class STHMVRV(Metric):
    """短期持有者市值实现值比率(STH-MVRV)指标"""

    @property
    def name(self) -> str:
        return "Short-Term Holder MVRV"

    @property
    def description(self) -> str:
        return "使用 Fisher 变换分析短期持有者市值实现值比率(STH-MVRV)，识别潜在的市场顶部和底部。"

    def __init__(
        self,
        data: pd.DataFrame,
        price_col: str = "btcusd",
        mvrv_col: str = "sth_mvrv",
        smooth_period: int = 10,
        fisher_period: int = 200,
        threshold: float = 2.0,
    ) -> None:
        """
        初始化 STHMVRV 指标类

        Args:
            data: 包含 STH-MVRV 数据的 DataFrame
            price_col: DataFrame 中表示比特币价格列的名称
            mvrv_col: DataFrame 中 STH-MVRV 列的名称
            smooth_period: 移动平均平滑周期
            fisher_period: Fisher 变换的计算周期
            threshold: 计算峰值/谷值信号的阈值
        """
        self.price_col = price_col
        self.mvrv_col = mvrv_col
        self.smooth_period = smooth_period
        self.fisher_period = fisher_period
        self.threshold = threshold
        super().__init__(data)

    def _validate_data(self) -> None:
        for col in [self.price_col, self.mvrv_col]:
            if col not in self.data.columns:
                raise ValueError(f"Input dataframe is missing required column: {col}")

    def generate_signals(self) -> None:
        self.signals = self.data.copy()

        # 计算平滑的 MVRV
        smoothed = (
            self.signals[self.mvrv_col]
            .rolling(self.smooth_period, min_periods=1)
            .mean()
        )

        # 计算 Fisher 变换
        normalized = fisher_transform(smoothed, self.fisher_period)
        self.signals["normalized_mvrv"] = normalized

        # 生成信号
        signals = np.where(normalized > self.threshold, 1, 0)
        signals = np.where(normalized < -self.threshold, -1, signals)
        self.signals["signal"] = signals

    def generate_chart(self) -> go.Figure:
        if self.signals is None:
            self.generate_signals()

        # 创建图表
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                "<b>Bitcoin price</b>",
                "<b>STH-MVRV with Fisher Transform</b>",
            ),
            row_heights=[0.7, 0.3],
        )

        # 添加比特币价格曲线
        fig.add_trace(
            go.Scatter(x=self.signals.index, y=self.signals[self.price_col]),
            row=1,
            col=1,
        )

        # 添加 MVRV 曲线
        fig.add_trace(
            go.Scatter(
                x=self.signals.index,
                y=self.signals["normalized_mvrv"],
                name="Normalized STH-MVRV",
                line=dict(color="#1f77b4"),
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

        # 添加极值区域背景
        peak_periods = find_trend_periods(self.signals["signal"] == 1)
        valley_periods = find_trend_periods(self.signals["signal"] == -1)

        for x0, x1 in peak_periods:
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor="#FF6B6B",
                opacity=0.2,
                line_width=0,
                row=1,
                col=1,
            )

        for x0, x1 in valley_periods:
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor="#38A169",
                opacity=0.2,
                line_width=0,
                row=1,
                col=1,
            )

        # 更新图表
        fig.update_layout(
            title=f"</b>{self.name}</b>",
            width=1000,
            height=700,
            template="plotly_white",
            showlegend=False,
        )

        return fig
