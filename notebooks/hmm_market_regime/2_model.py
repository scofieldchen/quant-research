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
    features = pd.read_csv(features_path, index_col="date", parse_dates=True)

    # 训练模型
    model, scaler = train_hmm(features, states=5)
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

        # 按照 rising_prob 对状态进行排序
        sorted_states = state_means["rising_prob"].sort_values().index

        # 识别两端的最强趋势
        bear_trend_id = sorted_states[0]  # rising_prob 最小
        bull_trend_id = sorted_states[-1]  # rising_prob 最大

        # 在剩余的状态中，根据atr_ratio区分 "Pause" 和 "Chop"
        remaining_ids = sorted_states[1:-1]
        remaining_means = state_means.loc[remaining_ids]
        remaining_means

        # 找到波动性最高/最低的那个
        high_vol_chop_id = remaining_means["atr_ratio"].idxmax()
        low_vol_id1 = remaining_means[
            "atr_ratio"
        ].idxmin()  # 可能是 Bull_Pause 或 Bear_Pause

        # 剩余的最后一个
        remaining_ids_set = set(remaining_ids) - {high_vol_chop_id, low_vol_id1}
        low_vol_id2 = list(remaining_ids_set)[0]

        # 根据 low_vol_id1和id2的 rising_prob 来确定哪个是 Bull_Pause, 哪个是 Bear_Pause
        if (
            state_means.loc[low_vol_id1, "rising_prob"]
            > state_means.loc[low_vol_id2, "rising_prob"]
        ):
            bull_pause_id = low_vol_id1
            bear_pause_id = low_vol_id2
        else:
            bull_pause_id = low_vol_id2
            bear_pause_id = low_vol_id1

        # 构建映射
        state_mapping = {
            bull_trend_id: "bull_trend",
            bear_trend_id: "bear_trend",
            bull_pause_id: "bull_pause",
            bear_pause_id: "bear_pause",
            high_vol_chop_id: "high_vol_chop",
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
    oos_prices = pd.read_csv(
        "BTCUSDT_oos_prices.csv", index_col="date", parse_dates=True
    )
    oos_prices = oos_prices.query("wfo == @wfo_num")

    # 样本外预测
    oos_features = oos_prices[
        ["log_return", "rising_prob", "atr_ratio", "fisher", "devia"]
    ]
    labels = map_states_to_labels(model, scaler, oos_features.columns)
    predicted_states = predict_state(model, scaler, oos_features, labels)

    mo.ui.table(
        predicted_states,
        selection=None,
        show_column_summaries=False,
        show_data_types=False,
        format_mapping={
            "bull_trend": "{:.1%}",
            "bear_trend": "{:.1%}",
            "bull_pause": "{:.1%}",
            "bear_pause": "{:.1%}",
            "high_vol_chop": "{:.1%}",
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
                open=oos_prices["open"],
                high=oos_prices["high"],
                low=oos_prices["low"],
                close=oos_prices["close"],
                name="BTCUSDT",
            )
        ]
    )

    # 定义5种市场状态对应的颜色
    state_colors = {
        "bull_trend": "rgba(34, 139, 34, 0.7)",  # 深绿色 - 强势上涨
        "bear_trend": "rgba(220, 20, 60, 0.7)",  # 深红色 - 强势下跌
        "bull_pause": "rgba(144, 238, 144, 0.5)",  # 浅绿色 - 上涨暂停/整理
        "bear_pause": "rgba(255, 182, 193, 0.5)",  # 浅红色 - 下跌暂停/整理
        "high_vol_chop": "rgba(65, 105, 225, 0.5)",  # 蓝色 - 高波动震荡
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


if __name__ == "__main__":
    app.run()
