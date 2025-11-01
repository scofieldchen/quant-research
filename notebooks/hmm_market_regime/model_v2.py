import marimo

__generated_with = "0.17.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    from pathlib import Path

    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    import plotly.io as pio
    from sklearn.preprocessing import StandardScaler
    from hmmlearn import hmm

    pio.templates.default = "simple_white"
    return Path, StandardScaler, go, hmm, mo, pd


@app.cell
def _(mo):
    file_selector = mo.ui.file(
        filetypes=[".csv"], kind="area", label="Select training file"
    )
    file_selector
    return (file_selector,)


@app.cell
def _(Path, StandardScaler, file_selector, hmm, mo, pd):
    mo.stop(not file_selector.value)

    def train_hmm(
        features: pd.DataFrame,
        states: int = 3,
        random_state: int = 123,
        verbose: bool = False,
    ) -> tuple[hmm.GaussianHMM, StandardScaler]:
        """训练隐马尔可夫模型"""
        # 特征标准化
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # 创建模型对象
        model = hmm.GaussianHMM(
            n_components=states,
            covariance_type="full",  # 特征相关程度高，应使用完全协方差矩阵，否则模型可能不收敛
            n_iter=2000,
            tol=1e-5,
            random_state=random_state,
            verbose=verbose,
        )

        # 训练模型
        model.fit(features_scaled)

        return model, scaler


    # 准备特征
    features_path = Path(file_selector.value[0].name)
    wfo_num = int(features_path.stem.split("_")[-1])
    features = pd.read_csv(features_path, index_col="Date", parse_dates=True)

    # 训练模型
    model, scaler = train_hmm(features, states=4)
    return features, model, scaler


@app.cell
def _(StandardScaler, features, file_selector, hmm, mo, model, pd, scaler):
    mo.stop(not file_selector.value)


    # 解读状态均值
    def get_state_means(
        model: hmm.GaussianHMM, scaler: StandardScaler, feature_names: list[str]
    ) -> pd.DataFrame:
        # model.means_ 是标准化后的均值，为了直观理解，我们需要将其“逆标准化”回原始尺度
        original_means = scaler.inverse_transform(model.means_)
        means_df = pd.DataFrame(original_means, columns=feature_names)
        return means_df


    state_means = get_state_means(model, scaler, list(features.columns))
    state_means
    return (get_state_means,)


@app.cell
def _(
    StandardScaler,
    features,
    file_selector,
    get_state_means,
    hmm,
    mo,
    model,
    scaler,
):
    mo.stop(not file_selector.value)


    # 自动识别状态
    def map_states_to_labels(
        model: hmm.GaussianHMM, scaler: StandardScaler, feature_names: list[str]
    ) -> dict[int, str]:
        """
        自动将HMM的状态ID映射到预定义的标签。
        该函数的核心是基于每个状态的特征均值来识别其市场意义。

        Args:
            model: 训练好的 hmm.GaussianHMM 对象。
            scaler: 用于逆标准化的 sklearn.StandardScaler 对象。
            feature_names (list): 特征名称列表，顺序必须与训练时一致。

        Returns:
            dict: 一个将状态ID映射到标签的字典, e.g., {0: 'Bull', 1: 'Risk', 2: 'Range'}.
        """
        # 获取状态均值
        state_means = get_state_means(model, scaler, feature_names)

        # 定义状态标签
        labels = [
            "bullish",  # 上涨趋势
            "bearish",  # 下跌趋势
            "range_high_vola",  # 宽幅震荡（高波动）
            "range_low_vola",  # 窄幅震荡（低波动）
        ]

        # 根据预定义的规则来识别状态
        bullish_state = state_means["RisingProb"].idxmax()
        bearish_state = state_means["RisingProb"].idxmin()

        means_2 = state_means.drop(index=[bullish_state, bearish_state])
        low_vola_state = means_2["ATR"].idxmin()
        high_vola_state = means_2["ATR"].idxmax()

        # 构建状态id到标签的字典
        state_mapping = {
            bullish_state: "bullish",
            bearish_state: "bearish",
            low_vola_state: "range_low_vola",
            high_vola_state: "range_high_vola",
        }

        return state_mapping


    state_labels = map_states_to_labels(model, scaler, list(features.columns))
    state_labels
    return (map_states_to_labels,)


@app.cell
def _(
    StandardScaler,
    file_selector,
    hmm,
    map_states_to_labels,
    mo,
    model,
    pd,
    scaler,
):
    mo.stop(not file_selector.value)


    def predict_state(
        model: hmm.GaussianHMM,
        scaler: StandardScaler,
        latest_features_df: pd.DataFrame,
        labels: dict[int, str],
    ) -> pd.DataFrame:
        """
        输入训练好的模型、scaler和包含最新特征的DataFrame，预测潜状态。

        Args:
            model: 训练好的 GaussianHMM 对象。
            scaler: 训练好的 StandardScaler 对象。
            latest_features_df (pd.DataFrame): 包含最新K线特征的DataFrame，
                                            列序必须与训练时一致。
            labels: 状态id到自定义标签的映射表

        Returns:
            pd.DataFrame, 包含预测状态，以及每个状态的概率
        """
        # 对新数据进行标准化（不需要fit）
        features_scaled = scaler.transform(latest_features_df)

        # 预测状态
        states = model.predict(features_scaled)
        states_series = pd.Series(states, index=latest_features_df.index)
        states_series.name = "state"
        states_series = states_series.map(labels)

        # 预测状态概率
        state_probs = model.predict_proba(features_scaled)
        state_probs_df = pd.DataFrame(
            state_probs,
            index=latest_features_df.index,
            columns=list(range(model.n_components)),
        )
        state_probs_df.columns = state_probs_df.columns.map(labels)

        return pd.concat([states_series, state_probs_df], axis=1)


    # 获取样本外数据
    oos_prices = pd.read_csv("oos_prices.csv", index_col="Date", parse_dates=True)
    oos_prices = oos_prices.query("WFO == @wfo_num")

    # 样本外预测
    oos_features = oos_prices[
        ["LogReturn", "RisingProb", "ATR", "Fisher", "Devia"]
    ]
    labels = map_states_to_labels(model, scaler, oos_features.columns)
    predicted_states = predict_state(model, scaler, oos_features, labels)

    mo.ui.table(
        predicted_states,
        selection=None,
        show_column_summaries=False,
        format_mapping={
            "bullish": "{:.1%}",
            "bearish": "{:.1%}",
            "range_low_vola": "{:.1%}",
            "range_high_vola": "{:.1%}",
        },
    )
    return oos_prices, predicted_states


@app.cell
def _(file_selector, go, mo, oos_prices, predicted_states):
    mo.stop(not file_selector.value)

    # 创建蜡烛图
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=oos_prices.index,
                open=oos_prices["Open"],
                high=oos_prices["High"],
                low=oos_prices["Low"],
                close=oos_prices["Close"],
                name="BTCUSDT",
            )
        ]
    )

    # 定义状态颜色
    state_colors = {
        "bullish": "rgba(0, 204, 0, 0.9)",  # 浅绿色
        "bearish": "rgba(204, 0, 0, 0.9)",  # 浅红色
        "range_low_vola": "rgba(0, 0, 255, 0.9)",  # 浅蓝色
        "range_high_vola": "rgba(204, 204, 0, 0.9)",  # 浅黄色
    }

    # 获取状态序列
    states = predicted_states["state"]

    # 识别连续相同状态的区间
    current_state = None
    start_idx = None

    for i, (date, state) in enumerate(states.items()):
        if state != current_state:
            # 如果不是第一个状态，先为前一个状态区间添加背景
            if current_state is not None and start_idx is not None:
                fig.add_vrect(
                    x0=states.index[start_idx],
                    x1=date,
                    fillcolor=state_colors[current_state],
                    layer="below",
                    line_width=0,
                )
            # 开始新的状态区间
            current_state = state
            start_idx = i

    # 处理最后一个状态区间
    if current_state is not None and start_idx is not None:
        fig.add_vrect(
            x0=states.index[start_idx],
            x1=states.index[-1],
            fillcolor=state_colors[current_state],
            layer="below",
            line_width=0,
        )

    # 更新布局
    fig.update_layout(
        title="BTC/USDT Market Regime",
        width=1000,
        height=600,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        yaxis_title="USDT",
    )

    fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
