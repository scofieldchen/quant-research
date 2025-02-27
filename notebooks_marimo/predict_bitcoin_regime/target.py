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


@app.cell(hide_code=True)
def _(Dict, List, Tuple, go, make_subplots, np, pd, talib):
    def calculate_market_state(
        df: pd.DataFrame,
        n1: int = 5,
        n2: int = 20,
        atr_multiplier: float = 2.0
    ) -> pd.Series:
        """计算市场状态，1代表上涨，-1代表下跌，0代表震荡。

        基于移动平均、ATR波动率通道和动量指标来判断市场状态。
        当价格突破波动率通道且短期均线高于长期均线，并且动量大于1时，判断为上涨；
        当价格跌破波动率通道且短期均线低于长期均线，并且动量小于1时，判断为下跌；
        其他情况判断为震荡。

        Args:
            df: 包含高开低收价格的时间序列数据框，索引应为时间序列。
            n1: 短期移动平均线周期，默认为5。
            n2: 长期移动平均线周期，默认为20。
            atr_multiplier: ATR乘数，用于计算波动率通道的宽度，默认为2.0。

        Returns:
            Series，包含每个时间点的市场状态，1代表上涨，-1代表下跌，0代表震荡。
        """

        close = df['close']

        # 计算移动平均
        fast_ma = talib.EMA(close, timeperiod=n1)
        slow_ma = talib.EMA(close, timeperiod=n2)

        # 根据ATR构建波动性通道
        atr = talib.ATR(df['high'], df['low'], close, timeperiod=n2)
        upper_band = slow_ma + atr * atr_multiplier
        lower_band = slow_ma - atr * atr_multiplier

        # 计算短期动量
        momentum = (close - close.shift(n1)) / close.shift(n1)

        # 市场状态判定规则
        market_state = pd.Series(np.zeros(len(close), dtype=int), index=close.index)
        market_state[(close > upper_band) & (fast_ma > slow_ma) & (momentum > 0.01)] = 1
        market_state[(close < lower_band) & (fast_ma < slow_ma) & (momentum < -0.01)] = -1

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
    n1 = mo.ui.number(start=5, stop=50, step=1, value=20, label="Fast MA period")
    n2 = mo.ui.number(start=50, stop=300, step=10, value=100, label="Slow MA period")
    atr_multiplier = mo.ui.number(start=0.5, stop=3.0, step=0.1, value=1.5, label="ATR multiplier")

    mo.hstack([n1, n2, atr_multiplier])
    return atr_multiplier, n1, n2


@app.cell
def _(
    atr_multiplier,
    btcusd,
    calculate_market_state,
    identify_market_state_periods,
    mo,
    n1,
    n2,
    visualize_market_state,
):
    if n1.value < n2.value:
        # 使用 .copy() 避免修改原始数据
        states = calculate_market_state(btcusd.copy(), int(n1.value), int(n2.value), atr_multiplier.value)
        state_periods = identify_market_state_periods(states.copy())
        visualize_market_state(btcusd.copy(), states.copy(), state_periods.copy())
    else:
        mo.md("## Error: Fast MA period must be less than Slow MA period.")
    return state_periods, states


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
