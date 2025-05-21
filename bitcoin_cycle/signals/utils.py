import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .indicators import lowpass_filter


def calculate_percentile_bands(
    data: pd.DataFrame,
    input_col: str,
    smooth_period: int = 7,
    rolling_period: int = 200,
    upper_band_percentile: float = 0.95,
    lower_band_percentile: float = 0.05,
) -> pd.DataFrame:
    """计算百分位数通道，作为常用分析方法"""
    # 先对指标进行移动平滑
    smooth_input_col = "smooth_" + input_col
    if smooth_period >= 2:  # 需要至少2个值才能计算
        data[smooth_input_col] = lowpass_filter(data[input_col], smooth_period)
    else:
        data[smooth_input_col] = data[input_col]

    # 计算百分位数通道
    data["upper_band"] = (
        data[smooth_input_col].rolling(rolling_period).quantile(upper_band_percentile)
    )
    data["lower_band"] = (
        data[smooth_input_col].rolling(rolling_period).quantile(lower_band_percentile)
    )

    # 生成信号
    signals = np.where(data[smooth_input_col] >= data["upper_band"], 1, 0)
    signals = np.where(
        data[smooth_input_col] <= data["lower_band"],
        -1,
        signals,
    )
    data["signal"] = signals

    return data


def add_percentile_bands(
    fig: go.Figure,
    data: pd.DataFrame,
    metric_col: str,
    smooth_metric_col: str = None,
    upper_band_col: str = "upper_band",
    lower_band_col: str = "lower_band",
    yaxis_title: str = "",
    metric_name: str = "",
    smooth_metric_name: str = "",
) -> go.Figure:
    """向副图添加指标和百分位数通道"""
    # 添加指标曲线和移动平滑
    if metric_col and smooth_metric_col:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[metric_col],
                name=metric_name if metric_name else metric_col,
                line=dict(color="lightblue", width=1.5),
                opacity=0.5,
                hovertemplate="<b>Date</b>: %{x}<br>"
                + "<b>Value</b>: %{y:.4f}<br><extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[smooth_metric_col],
                name=smooth_metric_name if smooth_metric_name else smooth_metric_col,
                line=dict(color="royalblue", width=2),
                hovertemplate="<b>Date</b>: %{x}<br>"
                + "<b>Value</b>: %{y:.4f}<br><extra></extra>",
            ),
            row=2,
            col=1,
        )
    elif smooth_metric_col is None:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[metric_col],
                name=metric_name if metric_name else metric_col,
                line=dict(color="royalblue", width=2),
                hovertemplate="<b>Date</b>: %{x}<br>"
                + "<b>Value</b>: %{y:.4f}<br><extra></extra>",
            ),
            row=2,
            col=1,
        )

    # 添加百分位数通道
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data[upper_band_col],
            line=dict(color="grey", width=1, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
            mode="lines",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data[lower_band_col],
            line=dict(color="grey", width=1, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(128, 128, 128, 0.1)",
        ),
        row=2,
        col=1,
    )

    # 更新 y 轴设置
    fig.update_yaxes(
        row=2,
        col=1,
        title=yaxis_title,
        title_font=dict(size=14),
        gridcolor="#e0e0e0",
    )
