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
    return Path, StandardScaler, go, hmm, mo, np, pd


@app.cell
def _(mo):
    file_selector = mo.ui.file(
        filetypes=[".csv"], kind="area", label="Select training file"
    )
    run_btn = mo.ui.run_button(label="Run model")

    mo.vstack([file_selector, run_btn])
    return file_selector, run_btn


@app.cell
def _(
    Path,
    analyze_states,
    file_selector,
    map_states_to_labels,
    mo,
    pd,
    run_btn,
    train_hmm,
):
    mo.stop(not run_btn.value)

    # 准备特征
    features_path = Path(file_selector.value[0].name)
    wfo_num = int(features_path.stem.split("_")[-1])
    features = pd.read_csv(features_path, index_col="Date", parse_dates=True)

    # 训练模型
    model, scaler = train_hmm(features, states=3)

    # 解读状态
    feature_names = list(features.columns)
    analyze_states(model, scaler, feature_names)

    # 自动识别状态
    regime = map_states_to_labels(model, scaler, feature_names)
    print("\n自动识别市场状态：")
    print(regime)
    return model, scaler


@app.cell
def _(map_states_to_labels, mo, model, pd, predict_state, run_btn, scaler):
    mo.stop(not run_btn.value)

    # 获取样本外数据
    oos_prices = pd.read_csv("oos_prices.csv", index_col="Date", parse_dates=True)
    oos_prices = oos_prices.query("WFO == @wfo_num")

    # 样本外预测
    oos_features = oos_prices[
        ["LogReturn", "RisingProb", "ATR", "Fisher", "Devia"]
    ]
    labels = map_states_to_labels(model, scaler, oos_features.columns)
    predicted_states = predict_state(model, scaler, oos_features, labels)

    mo.ui.table(predicted_states, selection=None, show_column_summaries=False)
    return oos_prices, predicted_states


@app.cell
def _(go, mo, oos_prices, predicted_states, run_btn):
    mo.stop(not run_btn.value)

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
        "Bull": "rgba(0, 255, 0, 0.7)",  # 浅绿色
        "Risk": "rgba(255, 0, 0, 0.7)",  # 浅红色
        "Range": "rgba(255, 255, 0, 0.7)",  # 浅黄色
    }

    # 获取状态序列
    states = predicted_states["State"]

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
def _(StandardScaler, hmm, pd):
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
    return (train_hmm,)


@app.cell
def _(StandardScaler, hmm, np, pd):
    def analyze_states(
        model: hmm.GaussianHMM, scaler: StandardScaler, feature_names: list[str]
    ) -> None:
        """分析和解读状态，赋予状态经济意义"""

        # 1. 解读状态均值 (Means)
        # model.means_ 是标准化后的均值，为了直观理解，我们需要将其“逆标准化”回原始尺度
        original_means = scaler.inverse_transform(model.means_)

        # 将结果整理成DataFrame，方便查看
        means_df = pd.DataFrame(original_means, columns=feature_names)
        print("--- 1. 各状态下的特征均值 (原始尺度) ---")
        print(means_df)
        print("\n")

        # 2. 解读状态协方差/波动性 (Covariances)
        # 我们主要关心每个状态下特征自身的方差（波动性），即协方差矩阵的对角线元素
        print("--- 2. 各状态下的特征波动性 (方差) ---")
        for i in range(model.n_components):
            # 提取对角线元素（方差）
            # 注意：这里的方差是在标准化尺度上的，可以直接比较大小
            variances = np.diag(model.covars_[i])
            print(f"状态 {i} 的方差 (标准化尺度):")
            # 将方差与特征名对应起来
            variance_series = pd.Series(variances, index=feature_names)
            print(variance_series)
            print("-" * 20)
        print("\n")

        # 3. 解读状态转移矩阵 (Transition Matrix)
        # transmat_[i, j] 表示从状态 i 转换到状态 j 的概率
        transmat_df = pd.DataFrame(
            model.transmat_,
            index=[f"从状态 {i}" for i in range(model.n_components)],
            columns=[f"到状态 {j}" for j in range(model.n_components)],
        )

        print("--- 3. 状态转移概率矩阵 ---")
        print(transmat_df)
        print("\n")

        # 附加步骤：检查每个状态的持续性
        # 状态的持续性概率 = 状态i转移到状态i的概率（对角线元素）
        # 持续性 = 1 / (1 - P_ii) ，表示平均停留的期数
        print("--- 4. 各状态的平均持续期数 ---")
        for i in range(model.n_components):
            persistence_prob = model.transmat_[i, i]
            avg_duration = 1 / (1 - persistence_prob)
            print(f"状态 {i} 的平均持续期数: {avg_duration:.2f} 根K线")
    return (analyze_states,)


@app.cell
def _(StandardScaler, hmm, pd):
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

        # 1. 获取原始尺度的均值，这是识别的基础
        original_means = scaler.inverse_transform(model.means_)
        means_df = pd.DataFrame(original_means, columns=feature_names)

        # 2. 定义状态标签
        # 这是你根据解读结果定义的规则。这个例子基于我们之前的讨论。
        # 'Bull': 强势上涨
        # 'Risk': 高波动风险 (宽幅震荡/恐慌下跌)
        # 'Range': 窄幅盘整
        labels = ["Bull", "Risk", "Range"]

        # 3. 创建规则来匹配状态和标签
        # 这里的关键是找到每个状态最“独特”的特征作为识别的“锚点”。

        # 规则1: 'Risk' 状态的 'atr' 均值最高
        # argmax()会返回最大值所在的索引（即状态ID）
        risk_state_id = means_df["ATR"].idxmax()

        # 规则2: 'Bull' 状态的 'rising_prob' 均值最高
        bull_state_id = means_df["RisingProb"].idxmax()

        # 规则3: 剩下的那个就是 'Range' 状态
        # 使用集合运算找到剩余的ID
        all_state_ids = set(range(model.n_components))
        identified_ids = {risk_state_id, bull_state_id}

        # 健壮性检查：确保'Risk'和'Bull'没有被识别为同一个状态
        if len(identified_ids) != 2:
            # 如果出现这种情况，说明模型可能没有清晰地分离出我们预期的状态
            # 需要更复杂的规则或人工干预
            raise ValueError(
                "Failed to uniquely identify 'Risk' and 'Bull' states."
                "The model structure might be ambiguous. Please check model parameters."
            )

        range_state_id = list(all_state_ids - identified_ids)[0]

        # 4. 构建并返回最终的映射字典
        state_mapping = {
            bull_state_id: "Bull",
            risk_state_id: "Risk",
            range_state_id: "Range",
        }

        # 最终检查，确保所有状态都被映射
        if len(state_mapping) != model.n_components:
            raise ValueError(
                "The number of mapped states does not match the model's components."
            )

        return state_mapping
    return (map_states_to_labels,)


@app.cell
def _(StandardScaler, hmm, pd):
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
        """
        # # 1. 确保输入是正确的格式 (例如, 一个2D数组)
        # if latest_features_df.ndim == 1:
        #     features_array = latest_features_df.values.reshape(1, -1)
        # else:
        #     # 如果你希望一次性预测多个点，可以传入多行
        #     features_array = latest_features_df.values

        # 2. 对新数据进行标准化
        features_scaled = scaler.transform(latest_features_df)

        # 3. 使用模型进行预测
        states = model.predict(features_scaled)
        states_series = pd.Series(states, index=latest_features_df.index)
        states_series.name = "State"
        states_series = states_series.map(labels)

        state_probs = model.predict_proba(features_scaled)
        state_probs_df = pd.DataFrame(
            state_probs,
            index=latest_features_df.index,
            columns=list(range(model.n_components)),
        )
        state_probs_df.columns = state_probs_df.columns.map(labels)

        return pd.concat([states_series, state_probs_df], axis=1)
    return (predict_state,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
