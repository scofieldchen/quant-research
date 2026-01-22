"""Binance k线数据管道任务模块。"""

import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import duckdb
import pandas as pd
import typer

from src.core.logger import get_logger
from src.pipelines.binance_klines.downloader import (
    AssetType,
    Granularity,
    PartitionInterval,
    fetch_api,
    fetch_historical,
)
from src.pipelines.binance_klines.storage import save_monthly_data, get_last_timestamp

logger = get_logger("binancekline")

# 全局配置
BASE_DATA_PATH = Path("data/cleaned")
TICKERS_FILE = BASE_DATA_PATH / "binance_tickers_perp.parquet"

app = typer.Typer(help="Binance k线数据管道任务")


def get_active_tickers(symbols: list[str] | None = None) -> pd.DataFrame:
    """获取活跃交易对列表。

    Args:
        symbols: 指定交易对列表，如果None则获取所有活跃的

    Returns:
        交易对DataFrame
    """
    df = pd.read_parquet(TICKERS_FILE)
    df["onboard_date"] = pd.to_datetime(df["onboard_date"], utc=True)
    df = df[df["status"] == "TRADING"]
    if symbols:
        symbols_upper = [s.upper() for s in symbols]
        df = df[df["symbol"].isin(symbols_upper)]
    return df


def clean_kline_data(df: pd.DataFrame) -> pd.DataFrame:
    """清洗k线数据，仅保留核心字段。

    Args:
        df: 原始k线DataFrame（索引为datetime）

    Returns:
        清洗后的DataFrame
    """
    # 重置索引，datetime 作为列
    df = df.reset_index().rename(columns={"index": "datetime"})

    # 仅保留核心字段
    df = df[["datetime", "open", "high", "low", "close", "volume"]].copy()

    # 确保类型
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def generate_months(start_date: dt.datetime, end_date: dt.datetime) -> list[str]:
    """生成月份列表，格式YYYY-MM。

    Args:
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        月份字符串列表
    """
    months = []
    current = start_date.replace(day=1)
    while current <= end_date:
        months.append(current.strftime("%Y-%m"))
        # 下一月
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    return months


def download_symbol_month(
    symbol: str, period: str, onboard_date: dt.datetime
) -> pd.DataFrame | None:
    """下载单个交易对的月度数据。

    Args:
        symbol: 交易对
        period: 期间 YYYY-MM
        onboard_date: 上市日期

    Returns:
        k线DataFrame或None
    """
    try:
        # 解析期间
        year, month = map(int, period.split("-"))
        period_start = dt.datetime(year, month, 1, tzinfo=dt.timezone.utc)
        period_end = (period_start + dt.timedelta(days=31)).replace(
            day=1
        ) - dt.timedelta(days=1)

        # 如果期间早于上市日期，跳过
        if period_end < onboard_date:
            logger.info(f"跳过{symbol} {period}，早于上市日期")
            return None

        # 尝试从历史ZIP下载
        date_in_period = period_start + dt.timedelta(days=15)  # 月中日期
        df = fetch_historical(
            symbol,
            Granularity.KLINE_1_MINUTE,
            date_in_period,
            AssetType.PERP,
            PartitionInterval.MONTHLY,
        )

        # 过滤到期间内
        df = df.loc[period_start:period_end]

        # 清洗数据
        df = clean_kline_data(df)

        logger.info(f"下载{symbol} {period}成功，行数：{len(df)}")
        return df

    except Exception as e:
        logger.error(f"下载{symbol} {period}失败: {e}")
        return None


@app.command()
def backfill(
    symbols: str = typer.Option(None, "--symbols", "-s", help="交易对列表，逗号分隔"),
    start_date: str = typer.Option(None, "--start-date", help="开始日期 YYYYMMDD"),
    end_date: str = typer.Option(None, "--end-date", help="结束日期 YYYYMMDD"),
    max_workers: int = typer.Option(5, help="最大线程数"),
):
    """回填历史数据。

    Args:
        symbols: 交易对列表，逗号分隔
        start_date: 开始日期 YYYYMMDD
        end_date: 结束日期 YYYYMMDD
        max_workers: 最大线程数
    """
    # 解析参数
    symbol_list = symbols.split(",") if symbols else None
    start_dt = (
        dt.datetime.strptime(start_date, "%Y%m%d").replace(tzinfo=dt.timezone.utc)
        if start_date
        else None
    )
    end_dt = (
        dt.datetime.strptime(end_date, "%Y%m%d").replace(tzinfo=dt.timezone.utc)
        if end_date
        else dt.datetime.now(dt.timezone.utc)
    )

    # 获取活跃交易对
    tickers_df = get_active_tickers(symbol_list)
    if tickers_df.empty:
        logger.warning("未找到活跃交易对")
        return

    # 收集下载任务
    tasks = []
    for _, row in tickers_df.iterrows():
        symbol = row["symbol"]
        onboard_dt = row["onboard_date"]
        effective_start = max(start_dt, onboard_dt) if start_dt else onboard_dt

        months = generate_months(effective_start, end_dt)
        for period in months:
            tasks.append((symbol, period, onboard_dt))

    logger.info(f"开始回填，共{len(tasks)}个任务，使用{max_workers}线程")

    # 多线程执行
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(download_symbol_month, symbol, period, onboard_dt)
            for symbol, period, onboard_dt in tasks
        ]

        for future in as_completed(futures):
            symbol, period, _ = tasks[futures.index(future)]
            try:
                df = future.result()
                if df is not None and not df.empty:
                    save_monthly_data(df, symbol, period)
            except Exception as e:
                logger.error(f"处理{symbol} {period}时出错: {e}")


def read_existing_month_data(
    symbol: str, period: str, base_path: Path | None = None
) -> pd.DataFrame:
    """读取现有月度数据。

    Args:
        symbol: 交易对
        period: 期间 YYYY-MM
        base_path: 基础路径

    Returns:
        现有DataFrame
    """
    if base_path is None:
        base_path = Path("data/cleaned/binance_klines_perp_m1")

    query = f"SELECT * FROM read_parquet('{base_path}/symbol={symbol}/period={period}/data.parquet')"
    try:
        df = duckdb.sql(query).df()
        return df
    except Exception:
        return pd.DataFrame(
            columns=["datetime", "open", "high", "low", "close", "volume"]
        )


def update_symbol(
    symbol: str,
    onboard_date: dt.datetime | pd.Timestamp,
    start_date: dt.date | None,
    end_date: dt.date,
    proxy: str | None = None,
) -> None:
    """更新单个交易对的最新数据。

    Args:
        symbol: 交易对
        onboard_date: 上市日期
        start_date: 开始日期，如果None则从最后时间戳开始
        end_date: 结束日期
        proxy: 代理，例如：http://127.0.0.1:7890
    """
    last_ts = get_last_timestamp(symbol)
    effective_start = start_date or (last_ts.date() if last_ts else onboard_date.date())
    if not last_ts and not start_date:
        logger.info(f"{symbol} 没有历史数据，跳过更新")
        return

    if effective_start > end_date:
        logger.info(f"{symbol} 数据已是最新")
        return

    logger.info(f"更新{symbol} 从 {effective_start} 到 {end_date}")

    merged_data = {}  # period -> df
    current = effective_start
    while current <= end_date:
        period = current.strftime("%Y-%m")
        if period not in merged_data:
            merged_data[period] = read_existing_month_data(symbol, period)

        # 下载当日数据：优先 ZIP，失败用 API
        df = None
        try:
            # 优先历史 ZIP
            day_dt = dt.datetime.combine(
                current, dt.time.min, tzinfo=dt.timezone.utc
            ) + dt.timedelta(hours=12)
            df = fetch_historical(
                symbol,
                Granularity.KLINE_1_MINUTE,
                day_dt,
                AssetType.PERP,
                PartitionInterval.DAILY,
            )
            logger.info(f"下载{symbol} {day_dt:%Y%m%d}成功(数据仓库)")
        except Exception:
            try:
                # 回退 API
                start_ts = dt.datetime.combine(
                    current, dt.time.min, tzinfo=dt.timezone.utc
                )
                if proxy:
                    df = fetch_api(symbol, start_ts, proxy={"http": proxy, "https": proxy})
                else:
                    df = fetch_api(symbol, start_ts)
                logger.info(f"下载{symbol} {day_dt:%Y%m%d}成功(API)")
            except Exception as e:
                logger.warning(f"下载{symbol} {current} 数据失败: {e}")

        if df is not None and not df.empty:
            df = clean_kline_data(df)
            data_to_merge = merged_data[period]
            if data_to_merge.empty:
                merged_data[period] = df
            else:
                merged_data[period] = pd.concat([data_to_merge, df]).drop_duplicates(
                    subset="datetime", keep="last"
                )

        current += dt.timedelta(days=1)

    # 保存每个月的合并数据
    for period, df in merged_data.items():
        if not df.empty:
            df = df.sort_values("datetime")
            save_monthly_data(df, symbol, period)
            logger.info(f"{symbol} {period} 更新完成")


@app.command()
def update(
    symbols: str = typer.Option(None, "--symbols", "-s", help="交易对列表，逗号分隔"),
    start_date: str = typer.Option(None, "--start-date", help="开始日期 YYYYMMDD"),
    end_date: str = typer.Option(None, "--end-date", help="结束日期 YYYYMMDD"),
    max_workers: int = typer.Option(5, help="最大线程数"),
    proxy: str = typer.Option(None, "--proxy", help="使用代理，例如http://127.0.0.1:7890")
):
    """增量更新最新数据。

    Args:
        symbols: 交易对列表，逗号分隔
        start_date: 开始日期 YYYYMMDD
        end_date: 结束日期 YYYYMMDD
        max_workers: 最大线程数
    """
    symbol_list = symbols.split(",") if symbols else None

    # 解析日期
    start_dt = dt.datetime.strptime(start_date, "%Y%m%d").date() if start_date else None
    end_dt = (
        dt.datetime.strptime(end_date, "%Y%m%d").date() if end_date else dt.date.today()
    )

    # 获取活跃交易对
    tickers_df = get_active_tickers(symbol_list)
    if tickers_df.empty:
        logger.warning("未找到活跃交易对")
        return

    logger.info(f"开始更新，共{len(tickers_df)}个交易对，使用{max_workers}线程")

    # 多线程执行
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for _, row in tickers_df.iterrows():
            symbol = row["symbol"]
            onboard_dt = row["onboard_date"]
            futures.append(
                executor.submit(update_symbol, symbol, onboard_dt, start_dt, end_dt, proxy)
            )

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"更新时出错: {e}")


if __name__ == "__main__":
    app()
