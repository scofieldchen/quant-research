"""Binance k线数据管道任务模块。"""

import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import duckdb
import pandas as pd
import requests
import typer

from src.core.logger import get_logger
from src.pipelines.binance_klines.downloader import (
    AssetType,
    Granularity,
    PartitionInterval,
    fetch_api,
    fetch_historical,
)
from src.pipelines.binance_klines.exceptions import DataNotFoundError
from src.pipelines.binance_klines.storage import save_monthly_data, get_last_timestamp

logger = get_logger("kline")

# 全局配置
BASE_DATA_PATH = Path("data/cleaned")
TICKERS_FILE = BASE_DATA_PATH / "binance_tickers_perp.parquet"

# 币安合约历史数据仓库的最早有效日期
EARLIEST_REPO_DATE = pd.Timestamp("2020-01-01", tz="UTC")


class BinanceKlinePipeline:
    """Binance 永续合约 K线数据管道类。

    统一管理历史回填和增量更新的调度逻辑。
    """

    def __init__(self, max_workers: int = 5, proxy: str | None = None):
        """初始化管道。

        Args:
            max_workers: 最大并发线程数
            proxy: 代理设置
        """
        self.max_workers = max_workers
        self.proxy = proxy
        self.session = requests.Session()
        if proxy:
            self.session.proxies = {"http": proxy, "https": proxy}

        # 状态记录
        self.failed_tasks: list[dict] = []
        self.missing_data: list[dict] = []
        self.success_count = 0

    def run_backfill(
        self,
        symbols: str | None,
        start_date: str | None,
        end_date: str | None,
    ):
        """历史数据回填。

        Args:
            symbols: 交易对列表，逗号分隔
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD
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

        # 生成任务列表：(symbol, period, onboard_date)
        tasks = []
        for _, row in tickers_df.iterrows():
            symbol = row["symbol"]
            raw_onboard = row["onboard_date"]

            # 强制截断：如果上市日期早于仓库最早日期，修正为仓库最早日期
            valid_onboard = max(raw_onboard, EARLIEST_REPO_DATE)

            effective_start = (
                max(start_dt, valid_onboard) if start_dt else valid_onboard
            )

            months = generate_months(effective_start, end_dt)
            for period in months:
                tasks.append((symbol, period, valid_onboard))

        logger.info(f"开始回填，共{len(tasks)}个任务，使用{self.max_workers}线程")

        # 执行任务
        self._process_tasks(tasks, worker_func=self._download_month_task)

        # 重试失败任务
        if self.failed_tasks:
            logger.info(f"开始重试{len(self.failed_tasks)}个失败任务")
            retry_tasks = [
                (t["symbol"], t["period"], t["onboard_date"]) for t in self.failed_tasks
            ]
            self.failed_tasks.clear()  # 清空重试
            self._process_tasks(retry_tasks, worker_func=self._download_month_task)

        # 生成报告
        self._generate_report("backfill")

    def run_update(
        self,
        symbols: str | None,
        start_date: str | None,
        end_date: str | None,
    ):
        """增量数据更新。

        Args:
            symbols: 交易对列表，逗号分隔
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD
        """
        symbol_list = symbols.split(",") if symbols else None

        # 解析日期（在 _update_symbol_task 中处理）

        # 获取活跃交易对
        tickers_df = get_active_tickers(symbol_list)
        if tickers_df.empty:
            logger.warning("未找到活跃交易对")
            return

        logger.info(
            f"开始更新，共{len(tickers_df)}个交易对，使用{self.max_workers}线程"
        )

        # 生成任务列表：(symbol, onboard_date)
        tasks = [
            (row["symbol"], row["onboard_date"]) for _, row in tickers_df.iterrows()
        ]

        # 执行任务
        self._process_tasks(tasks, worker_func=self._update_symbol_task)

        # 重试失败任务
        if self.failed_tasks:
            logger.info(f"开始重试{len(self.failed_tasks)}个失败任务")
            retry_tasks = [(t["symbol"], t["onboard_date"]) for t in self.failed_tasks]
            self.failed_tasks.clear()
            self._process_tasks(retry_tasks, worker_func=self._update_symbol_task)

        # 生成报告
        self._generate_report("update")

    def _process_tasks(self, tasks, worker_func):
        """通用并发调度器。

        Args:
            tasks: 任务列表
            worker_func: 工作函数
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(worker_func, task) for task in tasks]

            for future in as_completed(futures):
                try:
                    future.result()
                    self.success_count += 1
                except Exception:
                    # 工作函数中抛出的异常已记录，这里只需要计数
                    pass

    def _download_month_task(self, task):
        """下载单个交易对月度数据任务。

        Args:
            task: (symbol, period, onboard_date)

        Raises:
            Exception: 下载失败时抛出
        """
        symbol, period, onboard_date = task

        try:
            # 尝试从历史ZIP下载
            date_in_period = dt.datetime.strptime(period + "-15", "%Y-%m-%d").replace(
                tzinfo=dt.timezone.utc
            )
            df = fetch_historical(
                symbol,
                Granularity.KLINE_1_MINUTE,
                date_in_period,
                AssetType.PERP,
                PartitionInterval.MONTHLY,
                session=self.session,
            )

            # 过滤到期间内
            year, month = map(int, period.split("-"))
            period_start = dt.datetime(year, month, 1, tzinfo=dt.timezone.utc)
            period_end = (period_start + dt.timedelta(days=31)).replace(
                day=1
            ) - dt.timedelta(days=1)
            df = df.loc[period_start:period_end]

            # 清洗数据
            df = clean_kline_data(df)

            # 保存
            save_monthly_data(df, symbol, period)
            logger.info(f"下载{symbol} {period}成功，行数：{len(df)}")

        except DataNotFoundError as e:
            # 数据不存在，不重试
            self.missing_data.append(
                {
                    "symbol": symbol,
                    "period": period,
                    "reason": "data_not_found",
                    "url": e.url,
                }
            )
            logger.info(f"{symbol} {period} 数据不存在，跳过")
        except Exception as e:
            # 网络或其他错误，记录为失败
            self.failed_tasks.append(
                {
                    "symbol": symbol,
                    "period": period,
                    "onboard_date": onboard_date,
                    "error": str(e),
                }
            )
            raise  # 重新抛出让调度器捕获

    def _update_symbol_task(self, task):
        """更新单个交易对的最新数据任务。

        Args:
            task: (symbol, onboard_date)

        Raises:
            Exception: 更新失败时抛出
        """
        symbol, onboard_date = task

        try:
            last_ts = get_last_timestamp(symbol)
            effective_start = (
                last_ts.date()
                if last_ts
                else max(onboard_date.date(), EARLIEST_REPO_DATE.date())
            )
            end_date = dt.date.today()

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
                        session=self.session,
                    )
                except Exception:
                    try:
                        # 回退 API
                        start_ts = dt.datetime.combine(
                            current, dt.time.min, tzinfo=dt.timezone.utc
                        )
                        df = fetch_api(symbol, start_ts)
                    except Exception as e:
                        logger.warning(f"下载{symbol} {current} 数据失败: {e}")

                if df is not None and not df.empty:
                    df = clean_kline_data(df)
                    data_to_merge = merged_data[period]
                    if data_to_merge.empty:
                        merged_data[period] = df
                    else:
                        merged_data[period] = pd.concat(
                            [data_to_merge, df]
                        ).drop_duplicates(subset="datetime", keep="last")

                current += dt.timedelta(days=1)

            # 保存每个月的合并数据
            for period, df in merged_data.items():
                if not df.empty:
                    df = df.sort_values("datetime")
                    save_monthly_data(df, symbol, period)
                    logger.info(f"{symbol} {period} 更新完成")

        except Exception as e:
            # 记录失败
            self.failed_tasks.append(
                {
                    "symbol": symbol,
                    "onboard_date": onboard_date,
                    "error": str(e),
                }
            )
            raise

    def _generate_report(self, operation: str):
        """生成执行报告。

        Args:
            operation: 操作类型 ("backfill" or "update")
        """
        total_tasks = (
            self.success_count + len(self.failed_tasks) + len(self.missing_data)
        )
        report = {
            "operation": operation,
            "timestamp": dt.datetime.now().isoformat(),
            "total_tasks": total_tasks,
            "success_count": self.success_count,
            "failed_count": len(self.failed_tasks),
            "missing_data_count": len(self.missing_data),
            "failed_tasks": self.failed_tasks,
            "missing_data": self.missing_data,
        }

        # 保存为 JSON
        import json

        report_file = Path("data/cleaned") / f"binance_klines_{operation}_report.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"报告已保存到 {report_file}")


app = typer.Typer(help="Binance k线数据管道任务", add_completion=False)


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
    proxy: str = typer.Option(
        None, "--proxy", help="使用代理，例如http://127.0.0.1:7890"
    ),
):
    """回填历史数据"""
    pipeline = BinanceKlinePipeline(max_workers=max_workers, proxy=proxy)
    pipeline.run_backfill(symbols, start_date, end_date)


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
        with duckdb.connect() as con:
            df = con.sql(query).df()
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
        except Exception:
            try:
                # 回退 API
                start_ts = dt.datetime.combine(
                    current, dt.time.min, tzinfo=dt.timezone.utc
                )
                if proxy:
                    df = fetch_api(
                        symbol, start_ts, proxy={"http": proxy, "https": proxy}
                    )
                else:
                    df = fetch_api(symbol, start_ts)
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
    proxy: str = typer.Option(
        None, "--proxy", help="使用代理，例如http://127.0.0.1:7890"
    ),
):
    """增量更新最新数据"""
    pipeline = BinanceKlinePipeline(max_workers=max_workers, proxy=proxy)
    pipeline.run_update(symbols, start_date, end_date)


if __name__ == "__main__":
    app()
