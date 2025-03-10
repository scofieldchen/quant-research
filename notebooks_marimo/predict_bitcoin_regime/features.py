import marimo

__generated_with = "0.11.17"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from typing import List, Dict, Any, Optional, Tuple, Union
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import talib
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.stats import kruskal
    from numpy.typing import NDArray

    plt.style.use("ggplot")
    return (
        Any,
        Dict,
        List,
        NDArray,
        Optional,
        Tuple,
        Union,
        go,
        kruskal,
        make_subplots,
        np,
        pd,
        plt,
        sns,
        talib,
    )


@app.cell
def _(mo):
    mo.md("## 计算特征")
    return


@app.cell(hide_code=True)
def _(np, pd, talib):
    def calculate_fractal_dimension(data: pd.Series, period: int) -> pd.Series:
        """
        计算分形维数（Fractal Dimension）指标

        分形维数常用于衡量市场结构：
        - 指标 > 1.4：市场处于震荡行情
        - 指标 < 1.4: 市场处于趋势行情

        Args:
            data (pd.Series): 输入时间序列，通常是资产价格
            period (int): 指标回溯期，如果是奇数会自动转化为偶数
        """
        if period < 2:
            raise ValueError("period must be greater than 1")
        if len(data) < period:
            raise ValueError("data length must be >= period")

        def _fractal_dimension(data: np.ndarray) -> float:
            # 反转数据，索引0表示最新数据
            series = data[::-1]

            # 计算半周期
            period = len(data)
            period2 = period // 2

            n1 = (max(series[0:period2]) - min(series[0:period2])) / period2
            n2 = (
                max(series[period2:period]) - min(series[period2:period])
            ) / period2
            n3 = (max(series[0:period]) - min(series[0:period])) / period

            if n1 + n2 <= 0 or n3 <= 0:
                return 1.0

            return (np.log(n1 + n2) - np.log(n3)) / np.log(2)

        # 确保period是偶数
        # 如果period是奇数，将其减1，如果period是偶数，保持不变
        period = period & ~1

        return data.rolling(window=period, min_periods=period).apply(
            _fractal_dimension, raw=True
        )


    def calculate_fisher_transform(
        series: pd.Series, period: int = 10
    ) -> pd.Series:
        """
        计算费舍尔转换指标

        Args:
            series (pd.Series): 输入时间序列，通常是资产价格
            period (int): 指标回溯期
        """
        highest = series.rolling(period, min_periods=1).max()
        lowest = series.rolling(period, min_periods=1).min()
        values = np.zeros(len(series))
        fishers = np.zeros(len(series))

        for i in range(1, len(series)):
            values[i] = (
                0.66
                * (
                    (series.iloc[i] - lowest.iloc[i])
                    / (highest.iloc[i] - lowest.iloc[i])
                    - 0.5
                )
                + 0.67 * values[i - 1]
            )
            values[i] = max(min(values[i], 0.999), -0.999)
            fishers[i] = (
                0.5 * np.log((1 + values[i]) / (1 - values[i]))
                + 0.5 * fishers[i - 1]
            )

        return pd.Series(fishers, index=series.index)


    def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有指定的特征，并将结果保存为 pandas 数据框。

        Args:
            df: 包含 'open', 'high', 'low', 'close', 'volume' 字段的 pandas 数据框，索引是时间序列。

        Returns:
            包含所有计算出的特征的 pandas 数据框，索引与输入数据框相同。

        Raises:
            ValueError: 如果输入数据框缺少必要的字段。
        """
        # 检查输入字段是否合法
        required_columns = ["open", "high", "low", "close", "volume"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"输入数据框缺少字段: {col}")

        # 初始化结果数据框
        features_df = pd.DataFrame(index=df.index)

        # 趋势指标
        ema_10 = talib.EMA(df["close"], timeperiod=10)
        ema_50 = talib.EMA(df["close"], timeperiod=50)
        ema_200 = talib.EMA(df["close"], timeperiod=200)

        features_df["EMA_10_DIFF"] = ema_10.diff()
        features_df["EMA_50_DIFF"] = ema_50.diff()
        features_df["EMA_200_DIFF"] = ema_200.diff()

        features_df["TREND_RATIO_10_50"] = (ema_10 - ema_50) / ema_50
        features_df["TREND_RATIO_10_200"] = (ema_10 - ema_200) / ema_200
        features_df["TREND_RATIO_50_200"] = (ema_50 - ema_200) / ema_200

        # 动量指标
        features_df["RSI"] = talib.RSI(df["close"], timeperiod=14)
        _, _, macdhist = talib.MACD(
            df["close"], fastperiod=12, slowperiod=26, signalperiod=9
        )
        features_df["MACD_HIST"] = macdhist

        # 周期指标（费舍尔转换）
        features_df["FT"] = calculate_fisher_transform(df["close"], 10)

        # 波动性指标
        atr_20 = talib.ATR(df["high"], df["low"], df["close"], timeperiod=20)
        atr_100 = talib.ATR(df["high"], df["low"], df["close"], timeperiod=100)
        atr_ratio = atr_20 / atr_100

        features_df["ATR_20"] = atr_20
        features_df["ATR_100"] = atr_100
        features_df["ATR_RATIO"] = atr_ratio

        bb_up, bb_mid, bb_low = talib.BBANDS(df["close"], timeperiod=20)
        features_df["BB_WIDTH"] = bb_width = (bb_up - bb_low) / bb_mid
        features_df["BB_POSITION"] = (df["close"] - bb_low) / (bb_up - bb_low)

        # 成交量指标
        volume_ma_20 = talib.MA(df["volume"], timeperiod=20)
        volume_ma_100 = talib.MA(df["volume"], timeperiod=100)
        volume_ratio = volume_ma_20 / volume_ma_100

        features_df["VOLUMA_20_DIFF"] = volume_ma_20.diff()
        features_df["VOLUME_100_DIFF"] = volume_ma_100.diff()
        features_df["VOLUME_RATIO"] = volume_ratio

        # 市场结构(分形维数)
        features_df["FD"] = calculate_fractal_dimension(df["close"], 100)

        return features_df.dropna()
    return (
        calculate_features,
        calculate_fisher_transform,
        calculate_fractal_dimension,
    )


@app.cell
def _(calculate_features, pd):
    # 历史价格
    file_path = "~/quant-research/data/yahoo/Bitcoin.csv"
    btcusd = pd.read_csv(file_path, index_col=0, parse_dates=True)
    btcusd = btcusd.drop(columns="Adj Close")
    btcusd.columns = [x.lower() for x in btcusd.columns]
    btcusd.index.name = "date"

    # 目标变量
    target = pd.read_csv("./target.csv", index_col="date", parse_dates=True)

    # 计算特征
    features = calculate_features(btcusd)

    # 合并特征和目标变量
    df = features.join(target, on="date", how="left")

    # 删除缺失值
    df = df.dropna()

    df
    return btcusd, df, features, file_path, target


@app.cell
def _(mo):
    mo.md("""## 研究特征分布""")
    return


@app.cell(hide_code=True)
def _(Any, Dict, Tuple, pd, plt, sns):
    def calculate_descriptive_statistics(series: pd.Series) -> Dict[str, Any]:
        """
        计算 pandas 序列的详细基础统计量。

        Args:
            series: 代表特征的 pandas 序列，索引是时间序列索引。

        Returns:
            包含以下统计量的字典：
                'count': 样本数量
                'mean': 均值
                'std': 标准差
                'min': 最小值
                '25%': 第一四分位数
                '50%': 中位数
                '75%': 第三四分位数
                'max': 最大值
                'skew': 偏度系数
                'kurtosis': 峰度系数
        """
        statistics = {
            "count": series.count(),
            "mean": series.mean(),
            "std": series.std(),
            "min": series.min(),
            "25%": series.quantile(0.25),
            "50%": series.quantile(0.50),
            "75%": series.quantile(0.75),
            "max": series.max(),
            "skew": series.skew(),
            "kurtosis": series.kurtosis(),
        }

        return statistics


    def visualize_distribution(data: pd.Series, name: str) -> Tuple[plt.Figure, plt.Axes]:
        """
        创建直方图/密度图来分析特征的分布

        Args:
            data: 输入特征
            name: 特征名称
        """
        fig, ax = plt.subplots(figsize=(8, 5))

        sns.histplot(
            data.values,
            bins=30,
            kde=True,
            ax=ax,
            color="skyblue",
            edgecolor="black",
            alpha=0.7,
        )
        ax.set_title(f"Distribution of {name}")
        ax.set_xlabel(name)
        ax.set_ylabel("Frequency")

        plt.tight_layout()

        return fig, ax
    return calculate_descriptive_statistics, visualize_distribution


@app.cell
def _(features, mo, pd):
    feature_name = mo.ui.dropdown.from_series(pd.Series(features.columns), value=features.columns[0])
    return (feature_name,)


@app.cell
def _(
    calculate_descriptive_statistics,
    feature_name,
    features,
    mo,
    visualize_distribution,
):
    feature_values = features[feature_name.value]
    feature_stats = calculate_descriptive_statistics(feature_values)
    feature_dist_fig, feature_dist_ax = visualize_distribution(feature_values, feature_name.value)

    mo.vstack([feature_name, mo.md("---"), mo.hstack([feature_stats, feature_dist_ax], widths=[1,3])])
    return feature_dist_ax, feature_dist_fig, feature_stats, feature_values


@app.cell
def _(mo):
    mo.md(r"""## 研究特征之间的关系""")
    return


@app.cell(hide_code=True)
def _(List, go, np, pd):
    def plot_correlation_heatmap(
        features: pd.DataFrame, selected_features: List[str] = None
    ) -> go.Figure:
        """
        计算相关系数矩阵并创建交互式热力图进行可视化（使用 plotly），只显示下三角，并添加数字。

        Args:
            features: 包含特征的数据框，索引是时间序列。
            selected_features: 可选参数，指定要显示在热力图中的特征列表。
                               如果为 None，则默认显示所有特征。
        """

        if selected_features is None:
            selected_features = features.columns.tolist()

        # 确保选择的特征存在于数据框中
        valid_features = [
            feature for feature in selected_features if feature in features.columns
        ]
        if len(valid_features) != len(selected_features):
            print("警告：部分选择的特征不存在于数据框中，已自动忽略。")
        selected_features = valid_features

        # 计算相关系数矩阵
        corr_matrix = features[selected_features].corr()

        # 创建下三角 mask
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        corr_matrix_masked = corr_matrix.mask(mask)

        # 创建热力图
        fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix_masked.values,
                x=corr_matrix_masked.columns,
                y=corr_matrix_masked.index,
                colorscale="RdBu_r",  # 蓝色表示负相关，红色表示正相关
                zmin=-1,  # 设置颜色范围
                zmax=1,
                hovertemplate="Feature X: %{x}<br>Feature Y: %{y}<br>Correlation: %{z}<extra></extra>",  # 自定义悬停提示
            )
        )

        # 添加数字
        annotations = []
        for i, row in enumerate(corr_matrix_masked.values):
            for j, value in enumerate(row):
                if not np.isnan(value):
                    annotations.append(
                        dict(
                            x=corr_matrix_masked.columns[j],
                            y=corr_matrix_masked.index[i],
                            text=str(round(value, 2)),
                            showarrow=False,
                            font=dict(color="black")
                            if -0.5 < value < 0.5
                            else dict(color="white"),  # 根据数值大小调整字体颜色
                            xref="x",  # 引用x轴
                            yref="y",  # 引用y轴
                        )
                    )

        fig.update_layout(
            template="plotly_white",
            title="Interactive Correlation Heatmap of Selected Features (Lower Triangle)",
            xaxis_title="Features",
            yaxis_title="Features",
            height=1000,  # 调整图形高度
            width=1200,  # 调整图形宽度
            margin=dict(
                l=200, r=200, b=150, t=150
            ),  # 调整边距，留出更多空间显示标签
            annotations=annotations,  # 添加数字
        )

        return fig
    return (plot_correlation_heatmap,)


@app.cell
def _(features, mo):
    selected_features = mo.ui.multiselect(
        options=features.columns,
        value=features.columns,
    )
    return (selected_features,)


@app.cell
def _(features, mo, plot_correlation_heatmap, selected_features):
    mo.vstack(
        [
            mo.md("选择特征"),
            selected_features,
            mo.md("---"),
            plot_correlation_heatmap(features, selected_features.value),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""## 研究特征与目标变量的关系""")
    return


@app.cell
def _(mo):
    mo.md("""### 1. 数据可视化""")
    return


@app.cell(hide_code=True)
def _(Tuple, pd, plt, sns):
    def plot_single_violin(
        df: pd.DataFrame, feature: str, target_column: str = "target"
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        为指定的特征，按照目标变量的不同类别，绘制单个小提琴图。

        此函数用于展示单个特征在目标变量不同类别下的分布情况，
        帮助理解该特征与目标变量之间的关系。

        Args:
            df: 包含特征和目标变量的 Pandas DataFrame。
            feature: 需要绘制小提琴图的特征名。
            target_column: 目标变量的列名，默认为 'target'。

        Returns:
            包含图表 (plt.Figure) 和坐标轴对象 (plt.Axes) 的元组。

        Raises:
            ValueError: 如果 DataFrame 中缺少指定的特征或目标变量列。
        """

        if feature not in df.columns:
            raise ValueError(f"DataFrame 中缺少特征列: {feature}")

        if target_column not in df.columns:
            raise ValueError(f"DataFrame 中缺少目标变量列: {target_column}")

        fig, ax = plt.subplots(figsize=(8, 5))

        # 先绘制小提琴图
        sns.violinplot(
            x=target_column,
            y=feature,
            hue=target_column,
            data=df,
            ax=ax,
            palette="Set2",
            alpha=0.8,
            inner=None,  # 不显示内部箱体
            width=0.8,  # 控制小提琴图的宽度
            linewidth=1.2,  # 增加轮廓线宽度
            legend=False,
        )

        # 使用自定义箱线图来替换内部箱体
        sns.boxplot(
            x=target_column,
            y=feature,
            hue=target_column,
            data=df,
            ax=ax,
            width=0.3,  # 控制箱体宽度，使其窄于小提琴图
            palette="Set2",
            boxprops={'alpha': 0.7, 'zorder': 2},  # 设置箱体属性，确保在小提琴图上方
            medianprops={'color': 'red', 'linewidth': 2},  # 突出显示中位数线
            whiskerprops={'linewidth': 1.0, 'color': 'black'},  # 控制须线宽度
            capprops={'linewidth': 1.0, 'color': 'black'},  # 上下边界线宽度
            showfliers=False,  # 不显示异常值
            legend=False,
        )

        ax.set_title(f"Violin Plot of {feature} by {target_column}")
        ax.set_xlabel(target_column)
        ax.set_ylabel(feature)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()

        return fig, ax
    return (plot_single_violin,)


@app.cell
def _(features, mo, pd):
    feature_name_2 = mo.ui.dropdown.from_series(pd.Series(features.columns), value=features.columns[0])
    return (feature_name_2,)


@app.cell
def _(df, feature_name_2, mo, plot_single_violin):
    fig_violin, ax_violin = plot_single_violin(df.copy(), feature_name_2.value)

    mo.vstack([
        feature_name_2,
        mo.md("---"),
        fig_violin
    ])
    return ax_violin, fig_violin


@app.cell
def _(mo):
    mo.md("""### 2. 统计检验""")
    return


@app.cell(hide_code=True)
def _(Any, Dict, NDArray, kruskal, np):
    def perform_kruskal_wallis(feature: NDArray, target: NDArray) -> Dict[str, Any]:
        """
        对指定的特征和目标变量数组，执行 Kruskal-Wallis 检验。

        Kruskal-Wallis 检验用于检验不同类别下特征的中位数是否相同。
        此函数返回检验的统计量和 p 值。

        Args:
            feature: 包含特征数据的数组。
            target: 包含目标变量数据的数组。

        Returns:
            包含检验结果的字典，包括：
            - statistic: Kruskal-Wallis 统计量
            - pvalue: p 值

        Raises:
            ValueError: 如果特征数组和目标变量数组的长度不一致。
        """
        # 转换为 numpy 数组，方便处理
        feature = np.asarray(feature)
        target = np.asarray(target)

        # 确保特征数组和目标变量数组的长度一致
        if len(feature) != len(target):
            raise ValueError("特征数组和目标变量数组的长度必须一致")

        # 获取目标变量的唯一类别
        categories = np.unique(target)

        # 将每个类别下特征的数据提取到列表中
        data = [feature[target == category] for category in categories]

        # 执行 Kruskal-Wallis 检验
        statistic, pvalue = kruskal(*data)

        return {'statistic': statistic, 'pvalue': pvalue}
    return (perform_kruskal_wallis,)


@app.cell
def _(df, perform_kruskal_wallis):
    alpha = 0.05
    target_array = df["target"].values
    for col in df.drop(columns="target").columns:
        feature_array = df[col].values
        results = perform_kruskal_wallis(feature_array, target_array)
        pvalue = results["pvalue"]
        if pvalue < alpha:
            print(f"{col}: p-value({pvalue:.2%}) < {alpha:.1%}, 拒绝原假设，即不同类别下特征的中位数存在显著差异")
        else:
            print(f"{col}: p-value({pvalue:.2%}) > {alpha:.1%}, 接受原假设，即不同类别下特征的中位数没有显著差异")
        
    return alpha, col, feature_array, pvalue, results, target_array


@app.cell
def _(mo):
    mo.md("### 3. 特征重要性")
    return


@app.cell
def _():
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    return RandomForestClassifier, accuracy_score, train_test_split


@app.cell
def _(
    RandomForestClassifier,
    accuracy_score,
    df,
    pd,
    plt,
    sns,
    train_test_split,
):
    test_size = 0.2
    random_state = 123

    # 划分特征和目标变量
    X = df.drop(columns="target")
    y = df["target"]

    # 按照时间顺序划分训练集和检验集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False, random_state=random_state
    )

    # 随机森林分类器
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)

    # 训练模型
    model.fit(X_train, y_train)

    # 评估模型
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"模型在测试集上的预测精度为: {accuracy:.1%}")

    # 获取特征重要性
    importances = model.feature_importances_
    feature_importances = pd.DataFrame(
        {"feature": X.columns, "importance": importances}
    ).sort_values("importance", ascending=False)

    # 可视化特征重要性
    plt.figure(figsize=(8, 6))
    sns.barplot(x="importance", y="feature", data=feature_importances, orient="h")
    plt.title("Feature importance")
    plt.xticks(fontsize=8, rotation=45)
    plt.show()
    return (
        X,
        X_test,
        X_train,
        accuracy,
        feature_importances,
        importances,
        model,
        random_state,
        test_size,
        y,
        y_pred,
        y_test,
        y_train,
    )


@app.cell
def _(mo):
    mo.md("## 保存数据")
    return


@app.cell
def _(df):
    df.to_csv("features.csv", index=True)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
