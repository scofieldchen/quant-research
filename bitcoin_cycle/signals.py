import datetime as dt
from abc import ABC, abstractmethod
from typing import List, Tuple

import talib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def fisher_transform(series: pd.Series, period: int = 10) -> pd.Series:
    highest = series.rolling(period, min_periods=1).max()
    lowest = series.rolling(period, min_periods=1).min()
    values = np.zeros(len(series))
    fishers = np.zeros(len(series))

    for i in range(1, len(series)):
        values[i] = (
            0.66
            * (
                (series.iloc[i] - lowest.iloc[i]) / (highest.iloc[i] - lowest.iloc[i])
                - 0.5
            )
            + 0.67 * values[i - 1]
        )
        values[i] = max(min(values[i], 0.999), -0.999)
        fishers[i] = (
            0.5 * np.log((1 + values[i]) / (1 - values[i])) + 0.5 * fishers[i - 1]
        )

    return pd.Series(fishers, index=series.index)


def find_trend_periods(series: pd.Series) -> List[Tuple[dt.datetime, dt.datetime]]:
    """找到连续的1的开始时间和结束时间"""
    periods = []
    start = None

    for i in range(len(series)):
        if series.iloc[i] == 1 and start is None:
            start = series.index[i]
        elif series.iloc[i] == 0 and start is not None:
            end = series.index[i - 1]
            periods.append((start, end))
            start = None

    if start is not None:
        end = series.index[-1]
        periods.append((start, end))

    return periods


class Metric(ABC):
    """代表指标的抽象基类"""

    @abstractmethod
    def generate_signals(self) -> None:
        """生成信号，由子类实现"""
        pass

    @abstractmethod
    def generate_chart(self) -> None:
        """数据可视化，由子类实现"""
        pass


class STHRealizedPrice(Metric):
    """短期持有者的实现价格"""

    def __init__(
        self,
        data: pd.Series,
        btc_prices: pd.Series,
        period: int = 200,
        threshold: float = 2.0,
    ) -> None:
        self.data = data
        self.btc_prices = btc_prices
        self.period = period
        self.threshold = threshold
        self.signals = None

    def generate_signals(self) -> None:
        diff = self.btc_prices - self.data
        normalized_diff = fisher_transform(diff, self.period)
        self.signals = pd.concat(
            {
                "btcusd": self.btc_prices,
                "sth_realized_price": self.data,
                "normalized_diff": normalized_diff,
            },
            axis=1,
        )
        signals = np.where(
            self.signals["normalized_diff"] > self.threshold, "peak", "neutral"
        )
        signals = np.where(
            self.signals["normalized_diff"] < -self.threshold, "valley", signals
        )
        self.signals["signal"] = signals

    def generate_chart(self) -> go.Figure:
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
        )

        # 添加价格曲线
        fig.add_trace(
            go.Scatter(x=self.signals.index, y=self.signals["btcusd"]), row=1, col=1
        )

        # 添加指标曲线
        fig.add_trace(
            go.Scatter(x=self.signals.index, y=self.signals["sth_realized_price"]),
            row=1,
            col=1,
        )

        # 添加极值区域背景
        peak_periods = find_trend_periods(self.signals["signal"] == "peak")
        valley_periods = find_trend_periods(self.signals["signal"] == "valley")

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

        # 标准化指标
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
            title="STH Realized Price",
            width=1000,
            height=800,
            template="plotly_white",
            showlegend=False,
        )

        return fig
