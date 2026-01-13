import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from hmmlearn import hmm
    from sklearn.preprocessing import StandardScaler

    data_dir = Path("data/")
    return StandardScaler, data_dir, hmm, pd


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
def _(StandardScaler, hmm, pd):
    def get_state_means(
        model: hmm.GaussianHMM, scaler: StandardScaler, feature_names: list[str]
    ) -> pd.DataFrame:
        # model.means_ 是标准化后的均值，为了直观理解，我们需要将其“逆标准化”回原始尺度
        original_means = scaler.inverse_transform(model.means_)
        means_df = pd.DataFrame(original_means, columns=feature_names)
        return means_df
    return (get_state_means,)


@app.cell
def _(StandardScaler, get_state_means, hmm):
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
    return (predict_state,)


@app.cell
def _(data_dir, map_states_to_labels, pd, predict_state, train_hmm):
    # 获取所有训练文件
    training_files = [data_dir / f"BTCUSDT_train_{x:d}.csv" for x in range(1, 10)]

    # 加载样本外价格数据
    oos_prices = pd.read_csv(
        data_dir / "BTCUSDT_oos_prices.csv", index_col="date", parse_dates=True
    )

    all_predictions = []

    for features_path in training_files:
        wfo_num = int(features_path.stem.split("_")[-1])
        features = pd.read_csv(features_path, index_col="date", parse_dates=True)

        # 训练模型
        model, scaler = train_hmm(features, states=5)

        # 获取对应wfo的oos数据
        oos_prices_wfo = oos_prices.query("wfo == @wfo_num")

        # 样本外特征
        oos_features = oos_prices_wfo[
            ["log_return", "rising_prob", "atr_ratio", "fisher", "devia"]
        ]

        # 映射状态标签
        labels = map_states_to_labels(model, scaler, oos_features.columns)

        # 预测
        predicted_states = predict_state(model, scaler, oos_features, labels)
        predicted_states["wfo"] = wfo_num

        all_predictions.append(predicted_states)

    # 合并所有预测
    combined_predictions = pd.concat(all_predictions, axis=0)

    # 清洗：删除重复行
    combined_predictions = combined_predictions.drop_duplicates()

    # 导出到CSV
    output_path = data_dir / "hmm_predictions_oos.csv"
    combined_predictions.to_csv(output_path)

    print(f"Predictions exported to {output_path}")
    return (combined_predictions,)


@app.cell
def _(combined_predictions):
    combined_predictions.round(3)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
