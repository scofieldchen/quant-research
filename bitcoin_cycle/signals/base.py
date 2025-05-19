from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .indicators import find_trend_periods


class Metric(ABC):
    """代表指标的抽象基类"""

    def __init__(
        self,
        data: pd.DataFrame,
        chart_rows: int = 2,
        chart_width: int = 1000,
        chart_height: int = 700,
        **kwargs: Any,
    ) -> None:
        """初始化指标"""
        self.data = data
        self.chart_rows = chart_rows
        self.chart_width = chart_width
        self.chart_height = chart_height
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

    def _create_base_figure(self) -> go.Figure:
        """创建基础分层图表"""
        return make_subplots(
            rows=self.chart_rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
        )

    def _add_price_trace(self, fig: go.Figure) -> None:
        """添加比特币价格曲线"""
        fig.add_trace(
            go.Scatter(
                x=self.signals.index,
                y=self.signals[self.price_col],
                name="BTC/USD",
                line=dict(color="#F7931A", width=2),
                hovertemplate="<b>Date</b>: %{x}<br>"
                + "<b>Price</b>: $%{y:,.0f}<br><extra></extra>",
            ),
            row=1,
            col=1,
        )

    def _add_signal_backgrounds(self, fig: go.Figure) -> None:
        """添加信号背景区域"""
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

    def _update_layout(self, fig: go.Figure) -> None:
        """更新图表布局"""
        fig.update_layout(
            title=dict(
                text=f"<b>{self.name}</b>",
                x=0.5,
                y=0.95,
                font=dict(size=20),
            ),
            width=self.chart_width,
            height=self.chart_height,
            template="plotly_white",
            showlegend=True,
            legend=dict(
                orientation="h",  # 图例水平排列
                yanchor="bottom",  # 图例垂直对其方式
                y=1.02,  # 将图例放置在图表上方
                x=0.5,  # 将图例放置在图表中间
                xanchor="center",  # 图例水平对其方式
            ),
            # xaxis=dict(
            #     rangeselector=dict(
            #         buttons=list(
            #             [
            #                 dict(
            #                     count=3, label="3m", step="month", stepmode="backward"
            #                 ),
            #                 dict(
            #                     count=6, label="6m", step="month", stepmode="backward"
            #                 ),
            #                 dict(count=1, label="1y", step="year", stepmode="backward"),
            #                 dict(count=2, label="2y", step="year", stepmode="backward"),
            #                 dict(count=5, label="5y", step="year", stepmode="backward"),
            #                 dict(step="all"),
            #             ]
            #         )
            #     ),
            #     rangeslider=dict(visible=False),
            #     type="date",
            # ),
        )

        fig.update_yaxes(
            row=1,
            col=1,
            title="Price (USD)",
            type="log",
            gridcolor="#E5E5E5",
            title_font=dict(size=14),
        )

    @abstractmethod
    def _add_indicator_traces(self, fig: go.Figure) -> None:
        """添加指标相关曲线，由子类实现"""
        pass

    def generate_chart(self) -> go.Figure:
        """生成完整的图表"""
        if self.signals is None:
            self.generate_signals()

        fig = self._create_base_figure()
        self._add_price_trace(fig)
        self._add_indicator_traces(fig)
        self._add_signal_backgrounds(fig)
        self._update_layout(fig)

        return fig
