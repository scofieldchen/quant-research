import marimo

__generated_with = "0.11.10"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from typing import List, Dict, Any, Optional
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import talib
    from sklearn.preprocessing import StandardScaler
    return Any, Dict, List, Optional, StandardScaler, np, pd, plt, sns, talib


@app.cell
def _(pd, talib):
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
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"输入数据框缺少字段: {col}")

        # 初始化结果数据框
        features_df = pd.DataFrame(index=df.index)

        # 趋势指标
        for period in [10, 50, 200]:
            ema = talib.EMA(df['close'], timeperiod=period)
            ema_diff = ema.diff()
            feature_name = f'EMA_{period}_DIFF'
            features_df[feature_name] = ema_diff

            ema_diff_pct = (df['close'] - ema) / ema
            feature_name = f'EMA_{period}_DIFF_PCT'
            features_df[feature_name] = ema_diff_pct

        # 趋势指标，均线之间的差异
        ema_10 = talib.EMA(df['close'], timeperiod=10)
        ema_50 = talib.EMA(df['close'], timeperiod=50)
        ema_200 = talib.EMA(df['close'], timeperiod=200)
        trend_diff_ratio_10_50 = (ema_10 - ema_50) / ema_50
        trend_diff_ratio_10_200 = (ema_10 - ema_200) / ema_200
        trend_diff_ratio_50_200 = (ema_50 - ema_200) / ema_200
        features_df['TREND_DIFF_RATIO_10_50'] = trend_diff_ratio_10_50
        features_df['TREND_DIFF_RATIO_10_200'] = trend_diff_ratio_10_200
        features_df['TREND_DIFF_RATIO_50_200'] = trend_diff_ratio_50_200

        # 动量指标
        for period in [14, 28]:
            rsi = talib.RSI(df['close'], timeperiod=period)
            feature_name = f'RSI_{period}'
            features_df[feature_name] = rsi

        macd, macdsignal, macdhist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        features_df['MACD'] = macd
        features_df['MACD_SIGNAL'] = macdsignal
        features_df['MACD_HIST'] = macdhist

        # 周期指标，费舍尔转换

        # 波动性指标
        for period in [20, 100]:
            atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=period)
            feature_name = f'ATR_{period}'
            features_df[feature_name] = atr

            if period != 20:
                continue
        
            atr_20 = talib.ATR(df['high'], df['low'], df['close'], timeperiod=20)
            atr_100 = talib.ATR(df['high'], df['low'], df['close'], timeperiod=100)
            atr_ratio = atr_20 / atr_100
            feature_name = 'ATR_RATIO'
            features_df[feature_name] = atr_ratio

        bb_up, bb_mid, bb_low = talib.BBANDS(df['close'], timeperiod=20)
        bb_width = (bb_up - bb_low) / bb_mid
        feature_name = 'BB_WIDTH'
        features_df[feature_name] = bb_width

        bb_position = (df['close'] - bb_low) / (bb_up - bb_low)
        feature_name = 'BB_POSITION'
        features_df[feature_name] = bb_position

        # 成交量指标
        for period in [20, 100]:
            volume_ma = talib.MA(df['volume'], timeperiod=period)
            volume_ma_diff = volume_ma.diff()
            feature_name = f'VOLUME_MA_{period}_DIFF'
            features_df[feature_name] = volume_ma_diff

        volume_ma_20 = talib.MA(df['volume'], timeperiod=20)
        volume_ma_100 = talib.MA(df['volume'], timeperiod=100)
        volume_ratio = volume_ma_20 / volume_ma_100
        feature_name = 'VOLUME_RATIO'
        features_df[feature_name] = volume_ratio

        # 量价指标
        obv = talib.OBV(df['close'], df['volume'])
        obv_diff = obv.diff()
        feature_name = 'OBV_DIFF'
        features_df[feature_name] = obv_diff

        # 滞后特征
        lag_1_close = df['close'].shift(1)
        lag_1_close_diff = lag_1_close.diff()
        feature_name = 'LAG_1_CLOSE_DIFF'
        features_df[feature_name] = lag_1_close_diff

        lag_5_close = df['close'].shift(5)
        lag_5_close_diff = lag_5_close.diff()
        feature_name = 'LAG_5_CLOSE_DIFF'
        features_df[feature_name] = lag_5_close_diff

        return features_df.dropna()
    return (calculate_features,)


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
def _(btcusd, calculate_features):
    features = calculate_features(btcusd)
    features
    return (features,)


@app.cell
def _(List, features, np, pd, plt, sns):
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

    plot_correlation_heatmap(features)
    return (plot_correlation_heatmap,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
