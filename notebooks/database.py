import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # 查询数据库

    该脚本为辅助脚本，专门用于查询 parqeut 文件数据。
    """)
    return


@app.cell
def _():
    import marimo as mo

    from pathlib import Path

    import duckdb
    import pandas as pd
    return Path, mo, pd


@app.cell
def _(mo):
    mo.md(r"""
    ## 合约交易对
    """)
    return


@app.cell
def _():
    # 查询币安永续合约交易对

    # query = f"""
    # SELECT * FROM 'data/cleaned/binance_tickers_perp.parquet'
    # """

    # tickers_df = duckdb.sql(query).df()

    # mo.ui.table(tickers_df, selection=None, show_column_summaries=False)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## k线
    """)
    return


@app.cell
def _(Path, pd):
    data_dir = Path(
        "/Users/scofield/quant-research/data/cleaned/binance_klines_perp_m1/"
    )


    def load_market_data(
        data_dir: Path, symbol: str | None = None, period: str | None = None
    ) -> pd.DataFrame:
        """从分区目录加载 Parquet 数据。

        使用 pyarrow 引擎自动解析 Hive 分区结构。

        Args:
            data_dir: 数据根目录 (例如 'data/cleaned')。
            symbol: 交易对名称，如 'BTCUSDT'。如果为 None 则加载所有。
            period: 时间周期，如 '2025-11'。如果为 None 则加载所有。

        Returns:
            包含分区信息的 DataFrame。
        """
        if not data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")

        # 构建过滤条件 (Predicate Pushdown)
        # 仅在使用 pyarrow 引擎时有效，能显著减少 IO
        filters = []
        if symbol:
            filters.append(("symbol", "==", symbol))
        if period:
            filters.append(("period", "==", period))

        # 读取数据
        # engine='pyarrow' 是处理分区目录的最佳选择
        df = pd.read_parquet(
            data_dir, engine="pyarrow", filters=filters if filters else None
        )

        return df
    return data_dir, load_market_data


@app.cell
def _(data_dir, load_market_data):
    klines_df = load_market_data(data_dir, symbol="BTCUSDT", period="2026-01")
    klines_df
    return (klines_df,)


@app.cell
def _(klines_df):
    tmp = klines_df.set_index("datetime").loc["2026-01-17"]
    tmp
    return


@app.cell
def _(klines_df):
    klines_df.datetime.dt.date.unique()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
