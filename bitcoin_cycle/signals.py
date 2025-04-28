from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Type

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import talib
from plotly.subplots import make_subplots
from utils import find_trend_periods, fisher_transform


@dataclass
class ParameterInfo:
    """存储指标参数的元数据"""

    name: str
    description: str
    type: Type
    default: Any


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

    @property
    @abstractmethod
    def parameters_info(self) -> List[ParameterInfo]:
        """定义指标参数"""
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
        return "Short-Term Holder Realized Price Oscillator"

    @property
    def description(self) -> str:
        return (
            "比较比特币价格与其短期持有者实现价格，以识别潜在的市场顶部和底部。",
            "使用 Fisher 变换对价格差异进行归一化处理。",
        )

    @property
    def parameters_info(self) -> List[ParameterInfo]:
        return [
            ParameterInfo(
                name="price_col",
                description="Column name for Bitcoin price.",
                type=str,
                default="btcusd",
            ),
            ParameterInfo(
                name="sth_rp_col",
                description="Column name for STH Realized Price data.",
                type=str,
                default="sth_realized_price",
            ),
            ParameterInfo(
                name="period",
                description="Lookback period for the Fisher Transform.",
                type=int,
                default=200,
            ),
            ParameterInfo(
                name="threshold",
                description="Threshold for Fisher Transform to trigger peak/valley signals.",
                type=float,
                default=2.0,
            ),
        ]

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
