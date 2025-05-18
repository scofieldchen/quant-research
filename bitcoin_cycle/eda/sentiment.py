import marimo

__generated_with = "0.11.31"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import sys
    import datetime as dt
    from pathlib import Path

    sys.path.insert(0, "/users/scofield/quant-research/bitcoin_cycle")

    import talib
    import yfinance as yf
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    from signals import Metric
    from indicators import lowpass_filter, fisher_transform
    from binance import HistoricalFutureMetricsDownloader

    yf.set_config(
        proxy={"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}
    )
    return (
        HistoricalFutureMetricsDownloader,
        Metric,
        Path,
        dt,
        fisher_transform,
        go,
        lowpass_filter,
        np,
        pd,
        sys,
        talib,
        yf,
    )


@app.cell
def _(mo):
    mo.md("""## 数据""")
    return


@app.cell
def _(HistoricalFutureMetricsDownloader, Path, dt, mo, pd, yf):
    def load_lsr(data_dir: Path) -> pd.DataFrame:
        """从本地 csv 加载历史原始数据"""
        csv_files = sorted(data_dir.glob("*.csv"))

        if not csv_files:
            raise Exception(f"No data found in the directory: {str(data_dir)}")

        # 过滤2022年以前的文件，因为其不包含多空比例数据
        csv_files_2023_and_later = [
            file for file in csv_files if file.stem >= "20230101"
        ]

        return pd.concat(pd.read_csv(f) for f in csv_files_2023_and_later)


    def process_lsr(df: pd.DataFrame) -> pd.DataFrame:
        """处理数据，转化为日频多空比例"""
        drop_columns = [
            "sum_open_interest",
            "sum_open_interest_value",
            "sum_taker_long_short_vol_ratio",
        ]
        rename_columns = {
            "count_toptrader_long_short_ratio": "toptrader_long_short_ratio_account",
            "sum_toptrader_long_short_ratio": "toptrader_long_short_ratio_position",
            "count_long_short_ratio": "long_short_ratio",
        }

        df_processed = (
            df.drop(columns=drop_columns)
            .assign(datetime=lambda x: pd.to_datetime(x["datetime"]))
            .set_index("datetime")
            .shift(-1)  # 将时间序列向左移动一位，才能正确重采样
            .dropna()
            .resample("D")  # 重采样为日频数据
            .last()  # 因为情绪指标是市场快照，所以使用当天最后一个值
            .rename(columns=rename_columns)  # 使用更简洁的名称
        )

        return df_processed


    def download_lsr(data_dir: Path) -> None:
        """更新历史数据并整合到单一的日频数据文件"""
        # 获取历史数据的最后一天
        raw_data_dir = data_dir / "metrics" / "BTCUSDT"
        csv_files = sorted(raw_data_dir.glob("*.csv"))
        last_date = dt.datetime.strptime(csv_files[-1].stem, "%Y%m%d")
        last_date = last_date.replace(tzinfo=dt.timezone.utc)

        # 更新数据的日期范围
        start_date = last_date + dt.timedelta(days=1)
        end_date = dt.datetime.now(tz=dt.timezone.utc)
        if start_date > end_date:
            print("Long short ratio is already up to date.")
            return

        # 下载最新数据
        downloader = HistoricalFutureMetricsDownloader(data_dir / "metrics")
        downloader.download("BTCUSDT", start_date, end_date, max_workers=1)

        # 读取和处理数据
        lsr = load_lsr(raw_data_dir)
        daily_lsr = process_lsr(lsr)

        # 删除时间索引的时区信息，跟价格数据保持一致，时区默认为 UTC
        daily_lsr.index = daily_lsr.index.tz_convert(None)

        # 存储数据
        daily_lsr.to_csv(data_dir / "long_short_ratio.csv", index=True)


    @mo.cache
    def get_all_data(data_dir: Path) -> pd.DataFrame:
        # 获取比特币历史价格
        btcusd = yf.download(
            tickers="BTC-USD",
            ignore_tz=True,
            auto_adjust=True,
            progress=False,
            multi_level_index=False,
        )

        # 获取多空比例
        lsr = pd.read_csv(
            data_dir / "long_short_ratio.csv",
            index_col="datetime",
            parse_dates=True,
        )

        # 合并数据
        df = (
            lsr.join(btcusd["Close"], how="left")
            .rename(columns={"Close": "btcusd"})
            .dropna()
        )

        return df
    return download_lsr, get_all_data, load_lsr, process_lsr


@app.cell
def _(Path, get_all_data):
    data_dir = Path("/users/scofield/quant-research/bitcoin_cycle/data")

    # download_lsr(data_dir)
    data = get_all_data(data_dir)
    data
    return data, data_dir


@app.cell
def _(mo):
    mo.md("""## 分析""")
    return


@app.cell
def _(Metric, go, lowpass_filter, np, pd):
    class LongShortRatioAccount(Metric):
        """顶级交易员的多空比例（账户）"""

        @property
        def name(self) -> str:
            return "Toptrader Long Short Ratio(Account)"

        @property
        def description(self) -> str:
            pass

        def __init__(
            self,
            data: pd.DataFrame,
            price_col: str = "btcusd",
            metric_col: str = "toptrader_long_short_ratio_account",
            smooth_period: int = 10,
            rolling_period: int = 200,
            lower_band_percentile: float = 0.05,
            upper_band_percentile: float = 0.95,
        ) -> None:
            self.price_col = price_col
            self.metric_col = metric_col
            self.smooth_period = smooth_period
            self.rolling_period = rolling_period
            self.lower_band_percentile = lower_band_percentile
            self.upper_band_percentile = upper_band_percentile
            super().__init__(data)

        def _validate_data(self) -> None:
            for col in [self.price_col, self.metric_col]:
                if col not in self.data.columns:
                    raise ValueError(
                        f"Input dataframe is missing required column: {col}"
                    )

        def generate_signals(self) -> None:
            self.signals = self.data.copy()

            self.signals["smooth_ratio"] = lowpass_filter(
                self.signals[self.metric_col], self.smooth_period
            )
            self.signals["upper_band"] = (
                self.signals["smooth_ratio"]
                .rolling(self.rolling_period)
                .quantile(self.upper_band_percentile)
            )
            self.signals["lower_band"] = (
                self.signals["smooth_ratio"]
                .rolling(self.rolling_period)
                .quantile(self.lower_band_percentile)
            )

            signals = np.where(
                self.signals["smooth_ratio"] <= self.signals["lower_band"], 1, 0
            )
            signals = np.where(
                self.signals["smooth_ratio"] >= self.signals["upper_band"],
                -1,
                signals,
            )
            self.signals["signal"] = signals

        def _add_indicator_traces(self, fig: go.Figure) -> None:
            # 添加多空比例曲线
            fig.add_trace(
                go.Scatter(
                    x=self.signals.index,
                    y=self.signals[self.metric_col],
                    name="Raw Ratio",
                    line=dict(color="#add8e6", width=1.5),
                    opacity=0.5,
                    hoverinfo="x+y",
                ),
                row=2,
                col=1,
            )

            # 添加移动平滑曲线
            fig.add_trace(
                go.Scatter(
                    x=self.signals.index,
                    y=self.signals["smooth_ratio"],
                    name="Smooth Ratio",
                    line=dict(color="royalblue", width=2),
                    hoverinfo="x+y",
                ),
                row=2,
                col=1,
            )

            # 添加百分位数通道
            fig.add_trace(
                go.Scatter(
                    x=self.signals.index,
                    y=self.signals["upper_band"],
                    line=dict(color="grey", width=1, dash="dot"),
                    showlegend=False,
                    hoverinfo="skip",
                    mode="lines",
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=self.signals.index,
                    y=self.signals["lower_band"],
                    line=dict(color="grey", width=1, dash="dot"),
                    showlegend=False,
                    hoverinfo="skip",
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(128, 128, 128, 0.1)",  # 填充颜色和透明度 (浅灰色)
                ),
                row=2,
                col=1,
            )

            # 更新 y 轴设置
            fig.update_yaxes(
                row=2,
                col=1,
                autorange="reversed",
                title="Ratio(reversed)",
                title_font=dict(size=14),
                gridcolor="#e0e0e0",
            )
    return (LongShortRatioAccount,)


app._unparsable_cell(
    r"""
        class LongShortRatioPosition(Metric):
            \"\"\"顶级交易员的多空比例（仓位价值）\"\"\"
    
            @property
            def name(self) -> str:
                return \"Toptrader Long Short Ratio(Position)\"
    
            @property
            def description(self) -> str:
                pass
    
            def __init__(
                self,
                data: pd.DataFrame,
                price_col: str = \"btcusd\",
                metric_col: str = \"toptrader_long_short_ratio_position\",
                smooth_period: int = 10,
                rolling_period: int = 200,
                lower_band_percentile: float = 0.05,
                upper_band_percentile: float = 0.95,
            ) -> None:
                self.price_col = price_col
                self.metric_col = metric_col
                self.smooth_period = smooth_period
                self.rolling_period = rolling_period
                self.lower_band_percentile = lower_band_percentile
                self.upper_band_percentile = upper_band_percentile
                super().__init__(data)
    
            def _validate_data(self) -> None:
                for col in [self.price_col, self.metric_col]:
                    if col not in self.data.columns:
                        raise ValueError(
                            f\"Input dataframe is missing required column: {col}\"
                        )
    
            def generate_signals(self) -> None:
                self.signals = self.data.copy()
    
                self.signals[\"smooth_ratio\"] = lowpass_filter(
                    self.signals[self.metric_col], self.smooth_period
                )
    
                self.signals[\"upper_band\"] = (
                    self.signals[\"smooth_ratio\"]
                    .rolling(self.rolling_period)
                    .quantile(self.upper_band_percentile)
                )
                self.signals[\"lower_band\"] = (
                    self.signals[\"smooth_ratio\"]
                    .rolling(self.rolling_period)
                    .quantile(self.lower_band_percentile)
                )
    
                signals = np.where(
                    self.signals[\"smooth_ratio\"] >= self.signals[\"upper_band\"], 1, 0
                )
                signals = np.where(
                    self.signals[\"smooth_ratio\"] <= self.signals[\"lower_band\"],
                    -1,
                    signals,
                )
                self.signals[\"signal\"] = signals
    
            def _add_indicator_traces(self, fig: go.Figure) -> None:
                # 添加多空比例曲线
                fig.add_trace(
                    go.Scatter(
                        x=self.signals.index,
                        y=self.signals[self.metric_col],
                        name=\"Raw Ratio\",
                        line=dict(color=\"#add8e6\", width=1.5),
                        opacity=0.5,
                        hoverinfo=\"x+y\",
                    ),
                    row=2,
                    col=1,
                )
    
                # 添加移动平滑曲线
                fig.add_trace(
                    go.Scatter(
                        x=self.signals.index,
                        y=self.signals[\"smooth_ratio\"],
                        name=\"Smooth Ratio\",
                        line=dict(color=\"royalblue\", width=2),
                        hoverinfo=\"x+y\",
                    ),
                    row=2,
                    col=1,
                )
    
                # 添加百分位数通道
                fig.add_trace(
                    go.Scatter(
                        x=self.signals.index,
                        y=self.signals[\"upper_band\"],
                        line=dict(color=\"grey\", width=1, dash=\"dot\"),
                        showlegend=False,
                        hoverinfo=\"skip\",
                        mode=\"lines\",
                    ),
                    row=2,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=self.signals.index,
                        y=self.signals[\"lower_band\"],
                        line=dict(color=\"grey\", width=1, dash=\"dot\"),
                        showlegend=False,
                        hoverinfo=\"skip\",
                        mode=\"lines\",
                        fill=\"tonexty\",
                        fillcolor=\"rgba(128, 128, 128, 0.1)\",  # 填充颜色和透明度 (浅灰色)
                    ),
                    row=2,
                    col=1,
                )
    
                # 更新 y 轴设置
                fig.update_yaxes(
                    row=2,
                    col=1,
                    title=\"Ratio\",
                    title_font=dict(size=14),
                    gridcolor=\"#e0e0e0\",
                )
    """,
    name="_"
)


@app.cell
def _(Metric, go, lowpass_filter, np, pd):
    class LongShortRatio(Metric):
        """所有交易员的多空比例（账户）"""

        @property
        def name(self) -> str:
            return "Long Short Ratio"

        @property
        def description(self) -> str:
            pass

        def __init__(
            self,
            data: pd.DataFrame,
            price_col: str = "btcusd",
            metric_col: str = "long_short_ratio",
            smooth_period: int = 10,
            rolling_period: int = 200,
            lower_band_percentile: float = 0.05,
            upper_band_percentile: float = 0.95,
        ) -> None:
            self.price_col = price_col
            self.metric_col = metric_col
            self.smooth_period = smooth_period
            self.rolling_period = rolling_period
            self.lower_band_percentile = lower_band_percentile
            self.upper_band_percentile = upper_band_percentile
            super().__init__(data)

        def _validate_data(self) -> None:
            for col in [self.price_col, self.metric_col]:
                if col not in self.data.columns:
                    raise ValueError(
                        f"Input dataframe is missing required column: {col}"
                    )

        def generate_signals(self) -> None:
            self.signals = self.data.copy()

            self.signals["smooth_ratio"] = lowpass_filter(
                self.signals[self.metric_col], self.smooth_period
            )
            self.signals["upper_band"] = (
                self.signals["smooth_ratio"]
                .rolling(self.rolling_period)
                .quantile(self.upper_band_percentile)
            )
            self.signals["lower_band"] = (
                self.signals["smooth_ratio"]
                .rolling(self.rolling_period)
                .quantile(self.lower_band_percentile)
            )

            signals = np.where(
                self.signals["smooth_ratio"] <= self.signals["lower_band"], 1, 0
            )
            signals = np.where(
                self.signals["smooth_ratio"] >= self.signals["upper_band"],
                -1,
                signals,
            )
            self.signals["signal"] = signals

        def _add_indicator_traces(self, fig: go.Figure) -> None:
            # 添加多空比例曲线
            fig.add_trace(
                go.Scatter(
                    x=self.signals.index,
                    y=self.signals[self.metric_col],
                    name="Raw Ratio",
                    line=dict(color="#add8e6", width=1.5),
                    opacity=0.5,
                    hoverinfo="x+y",
                ),
                row=2,
                col=1,
            )

            # 添加移动平滑曲线
            fig.add_trace(
                go.Scatter(
                    x=self.signals.index,
                    y=self.signals["smooth_ratio"],
                    name="Smooth Ratio",
                    line=dict(color="royalblue", width=2),
                    hoverinfo="x+y",
                ),
                row=2,
                col=1,
            )

            # 添加百分位数通道
            fig.add_trace(
                go.Scatter(
                    x=self.signals.index,
                    y=self.signals["upper_band"],
                    line=dict(color="grey", width=1, dash="dot"),
                    showlegend=False,
                    hoverinfo="skip",
                    mode="lines",
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=self.signals.index,
                    y=self.signals["lower_band"],
                    line=dict(color="grey", width=1, dash="dot"),
                    showlegend=False,
                    hoverinfo="skip",
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(128, 128, 128, 0.1)",  # 填充颜色和透明度 (浅灰色)
                ),
                row=2,
                col=1,
            )

            # 更新 y 轴设置
            fig.update_yaxes(
                row=2,
                col=1,
                autorange="reversed",
                title="Ratio(reversed)",
                title_font=dict(size=14),
                gridcolor="#e0e0e0",
            )
    return (LongShortRatio,)


@app.cell
def _(LongShortRatio, data):
    # metric = LongShortRatioAccount(
    #     data,
    #     smooth_period=10,
    #     rolling_period=200,
    #     upper_band_percentile=0.95,
    #     lower_band_percentile=0.05,
    # )
    # metric.generate_signals()
    # fig = metric.generate_chart()

    # metric = LongShortRatioPosition(
    #     data,
    #     smooth_period=10,
    #     rolling_period=200,
    #     upper_band_percentile=0.95,
    #     lower_band_percentile=0.05,
    # )
    # metric.generate_signals()
    # fig = metric.generate_chart()

    metric = LongShortRatio(
        data,
        smooth_period=10,
        rolling_period=200,
        upper_band_percentile=0.95,
        lower_band_percentile=0.05,
    )
    metric.generate_signals()
    fig = metric.generate_chart()
    return fig, metric


@app.cell
def _(fig):
    fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
