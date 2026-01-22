"""Binance k线数据存储模块。"""

import datetime as dt
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.core.logger import get_logger

logger = get_logger("binancekline")


def save_monthly_data(
    df: pd.DataFrame, symbol: str, period: str, base_path: Path | None = None
) -> None:
    """保存月度k线数据到parquet分区。

    Args:
        df: k线数据，列包括datetime, open, high, low, close, volume
        symbol: 交易对名称，如'BTCUSDT'
        period: 期间，格式'YYYY-MM'
        base_path: 基础路径，默认为data/cleaned/binance_klines_perp_m1
    """
    if base_path is None:
        base_path = Path("data/cleaned/binance_klines_perp_m1")

    # 确保数据类型（数据已清洗）
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["high"] = pd.to_numeric(df["high"], errors="coerce")
    df["low"] = pd.to_numeric(df["low"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    # 创建分区路径
    partition_path = base_path / f"symbol={symbol}" / f"period={period}"
    partition_path.mkdir(parents=True, exist_ok=True)
    parquet_file = partition_path / "data.parquet"

    # 转换为pyarrow表，保存为parquet
    table = pa.Table.from_pandas(df)
    pq.write_table(table, parquet_file)

    logger.info(f"保存{symbol} {period}数据到{parquet_file}")


def get_last_timestamp(
    symbol: str, base_path: Path | None = None
) -> dt.datetime | None:
    """获取指定交易对的最后一条数据时间戳。

    Args:
        symbol: 交易对名称，如'BTCUSDT'
        base_path: 基础路径，默认为data/cleaned/binance_klines_perp_m1

    Returns:
        最后时间戳，如果没有数据则返回None
    """
    if base_path is None:
        base_path = Path("data/cleaned/binance_klines_perp_m1")

    # 使用duckdb查询分区parquet
    query = f"""
    SELECT MAX(datetime) as max_datetime
    FROM read_parquet('{base_path}/symbol={symbol}/**/*.parquet', hive_partitioning=true)
    WHERE symbol = '{symbol}'
    """

    try:
        result = duckdb.sql(query).df()
        if result.empty or result["max_datetime"].isna().all():
            logger.warning(f"未找到{symbol}的数据")
            return None
        last_ts = result["max_datetime"].iloc[0]
        if isinstance(last_ts, pd.Timestamp):
            return last_ts.to_pydatetime()
        else:
            return dt.datetime.fromisoformat(last_ts)
    except Exception as e:
        logger.error(f"查询{symbol}最后时间戳失败: {e}")
        return None
