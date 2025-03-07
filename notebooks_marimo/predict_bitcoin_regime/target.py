import marimo

__generated_with = "0.11.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import talib
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from typing import List, Tuple, Dict
    return Dict, List, Tuple, go, make_subplots, np, pd, talib


@app.cell
def _(Dict, List, Tuple, go, make_subplots, np, pd, talib):
    def lowpass_filter(x: pd.Series, period: int = 10) -> pd.Series:
        """
        低通滤波器

        Args:
            x (pd.Series): 时间序列输入，通常是价格或其它指标
            period (int): 截断窗口, 压制频率低于该窗口的高频波动

        Returns:
            pd.Series，输入的移动平滑
        """
        a = 2.0 / (1 + period)

        out = np.zeros(len(x))
        out[0] = x.iloc[0]
        out[1] = x.iloc[1]

        for i in range(2, len(x)):
            out[i] = (
                (a - 0.25 * a * a) * x.iloc[i]
                + 0.5 * a * a * x.iloc[i - 1]
                - (a - 0.75 * a * a) * x.iloc[i - 2]
                + (2.0 - 2.0 * a) * out[i - 1]
                - (1.0 - a) * (1.0 - a) * out[i - 2]
            )

        return pd.Series(out, index=x.index)


    def calculate_market_state(
        df: pd.DataFrame,
        lowpass_period: int = 200,
        atr_multiplier: float = 1.0
    ) -> pd.Series:
        """计算市场状态，1代表上涨，-1代表下跌，0代表震荡。

        Args:
            df: 包含高开低收价格的时间序列数据框，索引应为时间序列。
            lowpass_period: 低通滤波器的回溯期，默认为200。
            atr_multiplier: ATR乘数，用于计算波动率通道的宽度，默认为2.0。

        Returns:
            Series，包含每个时间点的市场状态，1代表上涨，-1代表下跌，0代表震荡。
        """

        close = df['close']

        # 计算趋势线
        trend = lowpass_filter(close, lowpass_period)

        # 根据ATR构建波动性通道
        atr = talib.ATR(df['high'], df['low'], close, timeperiod=14)
        upper_band = trend + atr * atr_multiplier
        lower_band = trend - atr * atr_multiplier

        # 市场状态判定规则
        market_state = pd.Series(np.zeros(len(close), dtype=int), index=close.index)
        market_state[(trend.diff() > 0) & (close > upper_band)] = 1
        market_state[(trend.diff() < 0) & (close < lower_band)] = -1

        return market_state


    def identify_market_state_periods(
        market_state: pd.Series
    ) -> Dict[int, List[Tuple[pd.Timestamp, pd.Timestamp]]]:
        """识别市场状态为1或-1的时期。

        Args:
            market_state: 包含市场状态的时间序列，1代表上涨，-1代表下跌，0代表震荡。

        Returns:
            字典，key为市场状态（1或-1），value为包含开始和结束时间的时间段列表。
        """
        periods = {1: [], -1: []}
        for state in [1, -1]:
            start = None
            for i, value in market_state.items():
                if value == state and start is None:
                    start = i
                elif value != state and start is not None:
                    periods[state].append((start, i))
                    start = None
            if start is not None:
                periods[state].append((start, market_state.index[-1]))  # 处理最后一个周期

        return periods


    def visualize_market_state(
        df: pd.DataFrame,
        market_state: pd.Series,
        periods: Dict[int, List[Tuple[pd.Timestamp, pd.Timestamp]]]
    ) -> None:
        """可视化市场状态和价格走势。

        使用Plotly创建交互式图表，显示价格曲线，并用不同颜色高亮显示上涨和下跌时期。

        Args:
            df: 包含价格的时间序列数据框，索引应为时间序列，至少包含'close'列。
            market_state: 包含市场状态的时间序列，1代表上涨，-1代表下跌，0代表震荡。
            periods: 包含开始和结束时间的时间段列表，由identify_market_state_periods函数生成。
        """
        fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

        # 添加价格曲线
        fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Close Price'), row=1, col=1)

        # 添加高亮矩形
        for state, color in [(1, 'lime'), (-1, 'red')]:
            for start, end in periods[state]:
                fig.add_vrect(x0=start, x1=end, fillcolor=color, opacity=0.2, line_width=0)

        # 更新布局
        fig.update_layout(
            title='Market State Visualization',
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False,
            template='plotly_white'  # 使用白色背景
        )

        fig.show()
    return (
        calculate_market_state,
        identify_market_state_periods,
        lowpass_filter,
        visualize_market_state,
    )


@app.cell
def _(pd):
    file_path = "~/quant-research/data/yahoo/Bitcoin.csv"
    btcusd = pd.read_csv(file_path, index_col=0, parse_dates=True)
    btcusd = btcusd.drop(columns="Adj Close")
    btcusd.columns = [x.lower() for x in btcusd.columns]
    btcusd.index.name = "date"
    btcusd
    return btcusd, file_path


@app.cell
def _(mo):
    lowpass_period = mo.ui.number(start=100, stop=500, step=10, value=200, label="Lowpass period")
    atr_multiplier = mo.ui.number(start=0.5, stop=3.0, step=0.1, value=1.5, label="ATR multiplier")

    mo.vstack([lowpass_period, atr_multiplier])
    return atr_multiplier, lowpass_period


@app.cell
def _(
    atr_multiplier,
    btcusd,
    calculate_market_state,
    identify_market_state_periods,
    lowpass_period,
    visualize_market_state,
):
    # 使用 .copy() 避免修改原始数据
    states = calculate_market_state(btcusd.copy(), int(lowpass_period.value), atr_multiplier.value)
    state_periods = identify_market_state_periods(states.copy())
    visualize_market_state(btcusd.copy(), states.copy(), state_periods.copy())
    return state_periods, states


@app.cell
def _(states):
    # 将states向上移动一位，目标是预测未来一期的市场结构
    # features(t) -> state(t+1)
    target = states.copy()
    target.name = "target"
    target.shift(-1).to_csv("target.csv", index=True)
    return (target,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
