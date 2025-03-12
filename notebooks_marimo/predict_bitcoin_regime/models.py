import marimo

__generated_with = "0.11.17"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    from typing import List, Tuple, Any, Optional, Dict
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import (
        StandardScaler,
        PowerTransformer,
        MinMaxScaler,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
    )

    plt.style.use("ggplot")
    return (
        Any,
        Dict,
        List,
        LogisticRegression,
        MinMaxScaler,
        Optional,
        PowerTransformer,
        StandardScaler,
        Tuple,
        accuracy_score,
        f1_score,
        mo,
        np,
        pd,
        plt,
        precision_score,
        recall_score,
        sns,
    )


@app.cell
def _(
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    accuracy_score,
    f1_score,
    pd,
    precision_score,
    recall_score,
):
    def walk_forward_splits(
        total_bars: int, train_bars: int, test_bars: int
    ) -> List[Tuple[int, int, int]]:
        """创建 Walk Forward Analysis (WFA) 交叉验证划分。

        根据给定的总柱数、训练柱数和测试柱数，生成一系列训练集和测试集的索引，
        用于时间序列数据的交叉验证。WFA 是一种常用的时间序列交叉验证方法，
        它模拟了模型在实际应用中的滚动预测过程。

        Args:
            total_bars: 数据集的总柱数。
            train_bars: 训练窗口的大小（柱数）。
            test_bars: 测试窗口的大小（柱数）。

        Returns:
            一个包含多个元组的列表，每个元组代表一个交叉验证的划分，
            包含训练集的起始索引、训练集的结束索引和测试集的结束索引。
            例如：[(train_start_index_1, train_end_index_1, test_end_index_1),
                  (train_start_index_2, train_end_index_2, test_end_index_2),
                  ...]

        Raises:
            ValueError: 如果总柱数小于等于0，或者训练柱数或测试柱数小于等于0，
                        或者训练柱数加上测试柱数大于总柱数，则抛出异常。
        """

        if total_bars <= 0:
            raise ValueError("total_bars must be greater than 0")

        if train_bars <= 0:
            raise ValueError("train_bars must be greater than 0")

        if test_bars <= 0:
            raise ValueError("test_bars must be greater than 0")

        if train_bars + test_bars > total_bars:
            raise ValueError(
                "train_bars + test_bars must be less than or equal to total_bars"
            )

        splits: List[Tuple[int, int, int]] = []
        train_start_index = 0  # 第一个训练集的初始索引

        while train_start_index + train_bars + test_bars <= total_bars:
            train_end_index = train_start_index + train_bars
            test_end_index = train_end_index + test_bars

            splits.append((train_start_index, train_end_index, test_end_index))

            train_start_index += test_bars  # 训练集向前滚动一个测试窗口

        return splits


    def walk_forward_validate(
        X: pd.DataFrame,
        y: pd.Series,
        model: Any,
        train_bars: int,
        test_bars: int,
        scaler: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        使用 Walk Forward Analysis (WFA) 验证模型，并记录评估指标。

        循环进行时间序列数据的 Walk Forward 划分，在每个训练集上训练模型，
        并在相应的测试集上生成预测，并计算和记录测试集的评估指标。

        Args:
            X: 特征矩阵，pandas DataFrame 格式，索引为日期。
            y: 目标变量，pandas Series 格式，索引为日期。
            model: 待训练和评估的机器学习模型，需要有 fit 和 predict 方法。
            train_bars: 训练窗口的柱子树。
            test_bars: 测试窗口的柱子数。
            scaler: 可选的标准化器，实现了 fit_transform 和 transform 方法。
                    如果为 None，则不进行标准化。

        Returns:
            一个列表，列表的每个元素是一个字典，记录了每个 WFA 循环的详细信息，
            包括循环次数、训练集和测试集的起始和结束日期、预测结果以及评估指标。
            例如:
            [
                {
                    "cycle": 1,
                    "train_start_date": "2023-01-01",
                    "train_end_date": "2023-01-30",
                    "train_days": 30,
                    "test_start_date": "2023-01-31",
                    "test_end_date": "2023-02-06",
                    "test_days": 7,
                    "accuracy": 0.85,
                    "precision": 0.80,
                    "recall": 0.90,
                    "f1_score": 0.85
                },
                ...
            ]
        """

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if not isinstance(y, pd.Series):
            raise TypeError("y must be a pandas Series")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")

        # 滚动划分训练集和测试集
        splits = walk_forward_splits(len(X), train_bars, test_bars)

        # 存储所有训练和测试的数据
        results: List[Dict[str, Any]] = []

        for i, (train_start, train_end, test_end) in enumerate(splits, 1):
            # 提取训练集和测试集
            X_train = X.iloc[train_start:train_end]
            y_train = y.iloc[train_start:train_end]
            X_test = X.iloc[train_end:test_end]
            y_test = y.iloc[train_end:test_end]

            # 应用标准化 (如果 scaler 不为 None)
            if scaler is not None:
                X_train_transformed = scaler.fit_transform(X_train)
                X_test_transformed = scaler.transform(X_test)
            else:
                X_train_transformed = X_train.copy()
                X_test_transformed = X_test.copy()

            # 训练模型
            model.fit(X_train_transformed, y_train)

            # 生成预测
            y_pred = model.predict(X_test_transformed)

            # 计算评估指标
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(
                y_test, y_pred, average="weighted", zero_division=0
            )
            recall = recall_score(
                y_test, y_pred, average="weighted", zero_division=0
            )
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            results.append(
                {
                    "cycle": i,
                    "train_start_date": X_train.index.min().strftime("%Y-%m-%d"),
                    "train_end_date": X_train.index.max().strftime("%Y-%m-%d"),
                    "train_days": len(X_train),
                    "test_start_date": X_test.index.min().strftime("%Y-%m-%d"),
                    "test_end_date": X_test.index.max().strftime("%Y-%m-%d"),
                    "test_days": len(X_test),
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                }
            )

        return results
    return walk_forward_splits, walk_forward_validate


@app.cell
def _(LogisticRegression, MinMaxScaler, pd, walk_forward_validate):
    # 准备特征和目标变量
    features = pd.read_csv("features.csv", index_col="date", parse_dates=True)
    start_date = "2023-01-01"
    end_date = "2024-12-31"
    features = features.loc[start_date:end_date]
    X = features.drop(columns="target")
    y = features["target"]

    # 模型
    model = LogisticRegression()

    # 标准化器
    scaler = MinMaxScaler()

    # WFA参数
    train_bars = 365
    test_bars = 30

    # 滚动训练模型
    results = walk_forward_validate(X, y, model, train_bars, test_bars, scaler)
    return (
        X,
        end_date,
        features,
        model,
        results,
        scaler,
        start_date,
        test_bars,
        train_bars,
        y,
    )


@app.cell
def _(pd, results):
    results_df = pd.DataFrame(results)
    results_df
    return (results_df,)


@app.cell
def _(results_df):
    print(f"Average accuracy: {results_df['accuracy'].mean():.1%}")
    print(f"Average precision: {results_df['precision'].mean():.1%}")
    print(f"Average recall: {results_df['recall'].mean():.1%}")
    print(f"Average f1-score: {results_df['f1_score'].mean():.1%}")
    return


@app.cell
def _(plt, results_df, sns):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(results_df, x="cycle", y="precision", ax=ax)
    return ax, fig


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
