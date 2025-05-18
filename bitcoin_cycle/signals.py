from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import talib
from plotly.subplots import make_subplots

from indicators import find_trend_periods, fisher_transform, lowpass_filter


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
            xaxis=dict(
                rangeselector=dict(
                    buttons=list(
                        [
                            dict(
                                count=3, label="3m", step="month", stepmode="backward"
                            ),
                            dict(
                                count=6, label="6m", step="month", stepmode="backward"
                            ),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(count=2, label="2y", step="year", stepmode="backward"),
                            dict(count=5, label="5y", step="year", stepmode="backward"),
                            dict(step="all"),
                        ]
                    )
                ),
                rangeslider=dict(visible=False),
                type="date",
            ),
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

    def _add_indicator_traces(self, fig: go.Figure) -> None:
        # 添加 NUPL 曲线
        fig.add_trace(
            go.Scatter(
                x=self.signals.index,
                y=self.signals["normalized_nupl"],
                name="Normalized STH-NUPL",
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

    def _add_indicator_traces(self, fig: go.Figure) -> None:
        # 添加 MVRV 曲线
        fig.add_trace(
            go.Scatter(
                x=self.signals.index,
                y=self.signals["normalized_mvrv"],
                name="Normalized STH-MVRV",
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


class FearGreedIndex(Metric):
    """恐慌和贪婪指数指标"""

    @property
    def name(self) -> str:
        return "Fear and Greed index"

    @property
    def description(self) -> str:
        return "通过恐慌和贪婪指数衡量比特币的市场情绪，识别潜在的市场顶部和底部。"

    def __init__(
        self,
        data: pd.DataFrame,
        price_col: str = "btcusd",
        fgi_col: str = "fgi",
        smooth_period: int = 10,
        extreme_greed_threshold: float = 80.0,
        extreme_fear_threshold: float = 20.0,
    ) -> None:
        """
        初始化 FearGreedIndex 指标类

        Args:
            data: 包含数据的 DataFrame
            price_col: DataFrame 中表示比特币价格列的名称
            fgi_col: DataFrame 中表示恐慌贪婪指数列的名称
        """
        self.price_col = price_col
        self.fgi_col = fgi_col
        self.smooth_period = smooth_period
        self.extreme_greed_threshold = extreme_greed_threshold
        self.extreme_fear_threshold = extreme_fear_threshold
        super().__init__(data)

    def _validate_data(self) -> None:
        for col in [self.price_col, self.fgi_col]:
            if col not in self.data.columns:
                raise ValueError(f"Input dataframe is missing required column: {col}")

    def generate_signals(self) -> None:
        self.signals = self.data.copy()
        self.signals["smooth_fgi"] = lowpass_filter(
            self.signals[self.fgi_col], self.smooth_period
        )
        signals = np.where(
            self.signals["smooth_fgi"] >= self.extreme_greed_threshold, 1, 0
        )
        signals = np.where(
            self.signals["smooth_fgi"] <= self.extreme_fear_threshold,
            -1,
            signals,
        )
        self.signals["signal"] = signals

    def _add_indicator_traces(self, fig: go.Figure) -> None:
        # 添加原始指标
        fig.add_trace(
            go.Scatter(
                x=self.signals.index,
                y=self.signals[self.fgi_col],
                name="Fear greed index",
                line=dict(color="#add8e6", width=1.5),
                opacity=0.5,
            ),
            row=2,
            col=1,
        )

        # 添加移动平滑曲线
        fig.add_trace(
            go.Scatter(
                x=self.signals.index,
                y=self.signals["smooth_fgi"],
                name=f"Smoothed index({self.smooth_period}-days)",
                line=dict(color="royalblue", width=2),
            ),
            row=2,
            col=1,
        )

        # 添加水平曲线表示极值区域
        levels = [
            (self.extreme_fear_threshold, "extreme fear"),
            (50, "neutral"),
            (self.extreme_greed_threshold, "extreme greed"),
        ]
        for level, text in levels:
            fig.add_hline(
                level,
                row=2,
                col=1,
                line_dash="dot",
                line_color="grey",
                line_width=2,
                annotation_text=text,
                annotation_position="top left",
            )

        # 更新第二行 y 轴设置
        fig.update_yaxes(
            row=2,
            col=1,
            title="Index(0-100)",
            gridcolor="#E5E5E5",
            title_font=dict(size=14),
        )


class ConsecutiveGreedDays(Metric):
    """连续贪婪天数指标"""

    @property
    def name(self) -> str:
        return "Consecutive Greed Days"

    @property
    def description(self) -> str:
        return "统计市场处于贪婪或极度贪婪状态的天数，基于恐慌贪婪指数"

    def __init__(
        self,
        data: pd.DataFrame,
        price_col: str = "btcusd",
        sentiment_col: str = "sentiment",
        extreme_level: int = 40,
    ) -> None:
        """
        初始化 ConsecutiveGreedDays 指标类

        Args:
            data: 包含数据的 DataFrame
            price_col: DataFrame 中表示比特币价格列的名称
            sentiment_col: DataFrame 中表示市场情绪状态的名称
        """
        self.price_col = price_col
        self.sentiment_col = sentiment_col
        self.extreme_level = extreme_level
        super().__init__(data)

    def _validate_data(self) -> None:
        for col in [self.price_col, self.sentiment_col]:
            if col not in self.data.columns:
                raise ValueError(f"Input dataframe is missing required column: {col}")

    def generate_signals(self) -> None:
        self.signals = self.data.copy()
        greed_days = np.zeros(len(self.signals), int)

        for i in range(1, len(greed_days)):
            if self.signals[self.sentiment_col].iloc[i] in [
                "Greed",
                "Extreme Greed",
            ]:
                greed_days[i] = greed_days[i - 1] + 1

        self.signals["greed_days"] = greed_days
        self.signals["signal"] = np.where(
            self.signals["greed_days"] >= self.extreme_level, 1, 0
        )

    def _add_indicator_traces(self, fig: go.Figure) -> None:
        # 添加连续贪婪天数曲线
        fig.add_trace(
            go.Scatter(
                x=self.signals.index,
                y=self.signals["greed_days"],
                name="Greed Days",
                line=dict(color="royalblue", width=2),
                hovertemplate="<b>Date</b>: %{x}<br>"
                + "<b>Days in Greed</b>: %{y:.0f}<br><extra></extra>",
            ),
            row=2,
            col=1,
        )

        # 添加水平线表示连续贪婪天数的极端水平
        reference_levels = [40, 60, 80, 100]
        for level in reference_levels:
            fig.add_hline(
                y=level,
                row=2,
                col=1,
                line_dash="dot",
                line_color="#34495e",
                line_width=1,
                annotation_text=f"{level} days",
                annotation_position="right",
            )

        # 更新 y 轴设置
        fig.update_yaxes(
            row=2,
            col=1,
            title="Days",
            gridcolor="#E5E5E5",
            title_font=dict(size=14),
        )
