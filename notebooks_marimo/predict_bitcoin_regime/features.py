import marimo

__generated_with = "0.11.10"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from typing import List, Dict, Any, Optional, Tuple
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import talib
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    plt.style.use("ggplot")
    return (
        Any,
        Dict,
        List,
        Optional,
        Tuple,
        go,
        make_subplots,
        np,
        pd,
        plt,
        sns,
        talib,
    )


@app.cell
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

        # 滞后特征
        features_df["LAG_1_CLOSE_DIFF"] = df["close"].shift(1).diff()
        features_df["LAG_5_CLOSE_DIFF"] = df["close"].shift(5).diff()

        return features_df.dropna()
    return (
        calculate_features,
        calculate_fisher_transform,
        calculate_fractal_dimension,
    )


@app.cell
def _(mo):
    mo.md(r"""## 计算特征""")
    return


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


@app.cell
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
def _(List, go, np, pd, plt, sns):
    def plot_correlation_heatmap(features: pd.DataFrame, selected_features: List[str] = None) -> None:
        """
        计算相关系数矩阵并创建热力图进行可视化。

        Args:
            features: 包含特征的数据框，索引是时间序列。
            selected_features: 可选参数，指定要显示在热力图中的特征列表。
                               如果为 None，则默认显示所有特征。
        """

        if selected_features is None:
            selected_features = features.columns.tolist()

        # 确保选择的特征存在于数据框中
        valid_features = [feature for feature in selected_features if feature in features.columns]
        if len(valid_features) != len(selected_features):
            print("警告：部分选择的特征不存在于数据框中，已自动忽略。")
        selected_features = valid_features

        # 计算相关系数矩阵
        corr_matrix = features[selected_features].corr()

        # 创建热力图
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) # 创建上三角mask
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=.5, mask=mask, cbar=True)
        plt.title("Correlation Heatmap of Selected Features")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()


    def plot_correlation_heatmap_interactive(features: pd.DataFrame, selected_features: List[str] = None) -> None:
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
        valid_features = [feature for feature in selected_features if feature in features.columns]
        if len(valid_features) != len(selected_features):
            print("警告：部分选择的特征不存在于数据框中，已自动忽略。")
        selected_features = valid_features

        # 计算相关系数矩阵
        corr_matrix = features[selected_features].corr()

        # 创建下三角 mask
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        corr_matrix_masked = corr_matrix.mask(mask)

        # 创建热力图
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix_masked.values,
            x=corr_matrix_masked.columns,
            y=corr_matrix_masked.index,
            colorscale="RdBu_r",  # 蓝色表示负相关，红色表示正相关
            zmin=-1,  # 设置颜色范围
            zmax=1,
            hovertemplate="Feature X: %{x}<br>Feature Y: %{y}<br>Correlation: %{z}<extra></extra>"  # 自定义悬停提示
        ))

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
                            font=dict(color='black') if -0.5 < value < 0.5 else dict(color='white'), # 根据数值大小调整字体颜色
                            xref="x", # 引用x轴
                            yref="y"  # 引用y轴
                        )
                    )

        fig.update_layout(
            template="plotly_white",
            title="Interactive Correlation Heatmap of Selected Features (Lower Triangle)",
            xaxis_title="Features",
            yaxis_title="Features",
            height=1000,  # 调整图形高度
            width=1200,  # 调整图形宽度
            margin=dict(l=200, r=200, b=150, t=150),  # 调整边距，留出更多空间显示标签
            annotations=annotations # 添加数字
        )

        fig.show()
    return plot_correlation_heatmap, plot_correlation_heatmap_interactive


@app.cell(hide_code=True)
def _(features, mo):
    selected_features = mo.ui.multiselect(
        options=features.columns,
        value=features.columns,
        label="选择特征"
    )

    selected_features
    return (selected_features,)


@app.cell(hide_code=True)
def _(features, plot_correlation_heatmap_interactive, selected_features):
    # plot_correlation_heatmap(features, selected_features.value)
    plot_correlation_heatmap_interactive(features, selected_features.value)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
