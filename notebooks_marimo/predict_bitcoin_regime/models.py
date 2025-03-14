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
    from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from lightgbm import LGBMClassifier
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        classification_report,
    )

    plt.style.use("ggplot")
    return (
        Any,
        Dict,
        GridSearchCV,
        LGBMClassifier,
        List,
        LogisticRegression,
        MinMaxScaler,
        Optional,
        RandomForestClassifier,
        SVC,
        TimeSeriesSplit,
        Tuple,
        accuracy_score,
        classification_report,
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
    GridSearchCV,
    List,
    Optional,
    TimeSeriesSplit,
    accuracy_score,
    f1_score,
    pd,
    precision_score,
    recall_score,
):
    def walk_forward_splits(
        data_length: int, train_bars: int, test_bars: int
    ) -> List[tuple[int, int, int]]:
        """
        生成用于时间序列数据 Walk Forward 验证的训练集和测试集分割点。

        Args:
            data_length: 数据总长度。
            train_bars: 训练窗口的大小。
            test_bars: 测试窗口的大小。

        Returns:
            一个列表，包含训练集和测试集的起始和结束索引。
            每个元组的格式为 (train_start, train_end, test_end)。
        """
        if train_bars + test_bars > data_length:
            raise ValueError(
                "train_bars + test_bars must be less than or equal to total_bars"
            )

        splits = []
        train_start = 0
        while train_start + train_bars + test_bars <= data_length:
            train_end = train_start + train_bars
            test_end = train_end + test_bars
            splits.append((train_start, train_end, test_end))
            train_start += test_bars

        return splits


    def calculate_metrics(
        y_true: pd.Series, y_pred: pd.Series
    ) -> Dict[str, float]:
        """
        计算分类模型的评估指标。

        Args:
            y_true: 真实的目标变量值。
            y_pred: 预测的目标变量值。

        Returns:
            包含准确率、精确率、召回率和 F1 分数的字典。
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }


    def train_and_predict(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model: Any,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        inner_cv_splits: int = 3,
        scoring: str = "accuracy",
    ) -> Dict[str, Any]:
        """
        训练模型并使用其进行预测，可以选择进行超参数优化。

        Args:
            X_train: 训练特征矩阵。
            y_train: 训练目标变量。
            X_test: 测试特征矩阵。
            y_test: 测试目标变量。
            model: 待训练和评估的机器学习模型。
            param_grid: 可选的超参数网格，用于 GridSearchCV。如果为 None，则不进行超参数优化。
            inner_cv_splits: 交叉验证的折数，用于训练窗口的超参数优化。
            scoring: 评估指标，用于 GridSearchCV。

        Returns:
            一个字典，包含预测精度和测试集的预测结果。
        """
        if param_grid is not None:
            tscv = TimeSeriesSplit(n_splits=inner_cv_splits)
            grid_search = GridSearchCV(
                model,  # 分类器
                param_grid,  # 搜索参数网格
                scoring=scoring,  # 评估指标
                cv=tscv,  # 交叉验证折数
                refit=True,  # 找到最优参数后重新对整个训练集拟合模型
                n_jobs=-1,  # 使用全部cpu
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            # best_params = grid_search.best_params_
        else:
            best_model = model
            best_model.fit(X_train, y_train)
            # best_params = {}

        y_pred = best_model.predict(X_test)

        metrics = calculate_metrics(y_test, y_pred)

        results = {
            "train_start_date": X_train.index.min().strftime("%Y-%m-%d"),
            "train_end_date": X_train.index.max().strftime("%Y-%m-%d"),
            "train_days": len(X_train),
            "test_start_date": X_test.index.min().strftime("%Y-%m-%d"),
            "test_end_date": X_test.index.max().strftime("%Y-%m-%d"),
            "test_days": len(X_test),
            "y_test": y_test,
            "y_pred": pd.Series(y_pred, index=y_test.index),
        }
        results.update(metrics)

        return results


    def walk_forward_validate(
        X: pd.DataFrame,
        y: pd.Series,
        train_bars: int,
        test_bars: int,
        model: Any,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        scaler: Optional[Any] = None,
        inner_cv_splits: int = 3,
        scoring: str = "accuracy",
    ) -> List[Dict[str, Any]]:
        """使用 Walk Forward Analysis (WFA) 验证模型，并进行超参数优化（可选）。

        循环进行时间序列数据的 Walk Forward 划分，在每个训练集上进行模型训练和预测，
        并在相应的测试集上生成预测，并计算和记录测试集的评估指标。
        如果提供了 param_grid，则使用 GridSearchCV 进行超参数优化。

        Args:
            X: 特征矩阵，pandas DataFrame 格式，索引为日期。
            y: 目标变量，pandas Series 格式，索引为日期。
            train_bars: 训练窗口的柱子数。
            test_bars: 测试窗口的柱子数。
            model: 待训练和评估的机器学习模型，需要有 fit 和 predict 方法。
            param_grid: 可选的超参数网格，用于 GridSearchCV。如果为 None，则不进行超参数优化。
            scaler: 可选的标准化器，实现了 fit_transform 和 transform 方法。
                    如果为 None，则不进行标准化。
            inner_cv_splits: 交叉验证的折数，用于训练窗口的超参数优化。
            scoring: 评估指标，用于 GridSearchCV。

        Returns:
            一个列表，列表的每个元素是一个字典，记录了每个 WFA 循环的详细信息，
            包括循环次数、训练集和测试集的起始和结束日期、预测结果以及评估指标，
            以及最佳超参数（如果进行了超参数优化）。
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

        for train_start, train_end, test_end in splits:
            # 提取训练集和测试集
            X_train = X.iloc[train_start:train_end]
            y_train = y.iloc[train_start:train_end]
            X_test = X.iloc[train_end:test_end]
            y_test = y.iloc[train_end:test_end]

            # 应用标准化 (如果 scaler 不为 None)
            if scaler is not None:
                X_train_transformed = pd.DataFrame(
                    scaler.fit_transform(X_train),
                    index=X_train.index,
                    columns=X_train.columns,
                )
                X_test_transformed = pd.DataFrame(
                    scaler.transform(X_test),
                    index=X_test.index,
                    columns=X_test.columns,
                )
            else:
                X_train_transformed = X_train.copy()
                X_test_transformed = X_test.copy()

            # 训练模型并进行预测
            res = train_and_predict(
                X_train_transformed,
                y_train,
                X_test_transformed,
                y_test,
                model,
                param_grid,
                inner_cv_splits,
                scoring,
            )

            results.append(res)

        return results
    return (
        calculate_metrics,
        train_and_predict,
        walk_forward_splits,
        walk_forward_validate,
    )


@app.cell
def _(pd):
    # 特征和目标变量
    features = pd.read_csv("features.csv", index_col="date", parse_dates=True)
    start_date = "2020-01-01"
    end_date = "2024-12-31"
    features = features.loc[start_date:end_date]
    X = features.drop(columns="target")
    y = features["target"]
    return X, end_date, features, start_date, y


@app.cell
def _(mo):
    mo.md("## 测试单个模型")
    return


@app.cell
def _():
    # # 参数
    # train_bars = 365
    # test_bars = 30

    # # 模型
    # model = LogisticRegression()

    # # 标准化器
    # scaler = MinMaxScaler()

    # # 评估模型
    # results = walk_forward_validate(
    #     X=X,
    #     y=y,
    #     train_bars=train_bars,
    #     test_bars=test_bars,
    #     model=model,
    #     scaler=scaler
    # )
    # results_df = pd.DataFrame(results).drop(columns=["y_test", "y_pred"])
    # print(results_df)

    # # 评估预测精度
    # y_test = pd.concat((x["y_test"] for x in results))
    # y_pred = pd.concat((x["y_pred"] for x in results))
    # metrics = calculate_metrics(y_test, y_pred)
    # for k,v in metrics.items():
    #     print(f"{k}: {v:.1%}")
    return


@app.cell
def _(mo):
    mo.md("## 对比不同模型")
    return


@app.cell
def _():
    # # 参数
    # train_bars = 365*3
    # test_bars = 30
    # random_state = 123

    # # 模型
    # models = [
    #     ("Logistic Regresion", LogisticRegression(random_state=random_state)),
    #     ("SVM", SVC(random_state=random_state)),
    #     ("Random Forest", RandomForestClassifier(random_state=random_state)),
    #     ("LightGBM", LGBMClassifier(random_state=random_state))
    # ]

    # # 标准化器
    # scaler = MinMaxScaler()

    # # 评估模型
    # model_comparisons = []
    # for model_name, model_instance in models:
    #     results = walk_forward_validate(
    #         X=X,
    #         y=y,
    #         train_bars=train_bars,
    #         test_bars=test_bars,
    #         model=model_instance,
    #         scaler=scaler
    #     )
    #     y_test = pd.concat((x["y_test"] for x in results))
    #     y_pred = pd.concat((x["y_pred"] for x in results))
    #     metrics = calculate_metrics(y_test, y_pred)
    #     metrics.update({"model": model_name})
    #     model_comparisons.append(metrics)

    # model_comparisions_df = pd.DataFrame(model_comparisons)
    return


@app.cell
def _(model_comparisions_df):
    model_comparisions_df
    return


@app.cell
def _(mo):
    mo.md("## 优化超参数")
    return


@app.cell
def _(
    LGBMClassifier,
    MinMaxScaler,
    X,
    calculate_metrics,
    pd,
    walk_forward_validate,
    y,
):
    # 参数
    train_bars = 365 * 3
    test_bars = 30
    random_state = 123

    # 模型
    model = LGBMClassifier(random_state=random_state)

    # 超参数扫描区间
    param_grid = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.1],
        # "num_leaves": [20, 31, 40],
        # "max_depth": [-1, 6, 8],
        # "min_child_samples": [20, 30],
        # "subsample": [0.8, 1.0],
        # "colsample_bytree": [0.8, 1.0],
        # "reg_alpha": [0.0, 0.1],
        # "reg_lambda": [0.0, 0.1],
    }

    # 标准化器
    scaler = MinMaxScaler()

    # 评估模型
    results = walk_forward_validate(
        X=X,
        y=y,
        train_bars=train_bars,
        test_bars=test_bars,
        model=model,
        scaler=scaler,
        param_grid=param_grid,
    )

    # 评估预测精度
    y_test = pd.concat((x["y_test"] for x in results))
    y_pred = pd.concat((x["y_pred"] for x in results))
    metrics = calculate_metrics(y_test, y_pred)
    for k, v in metrics.items():
        print(f"{k}: {v:.1%}")
    return (
        k,
        metrics,
        model,
        param_grid,
        random_state,
        results,
        scaler,
        test_bars,
        train_bars,
        v,
        y_pred,
        y_test,
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
