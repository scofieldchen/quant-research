"""Binance k线数据管道任务模块。"""

import json
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import duckdb
import pandas as pd
import requests
import typer

from typing import Sequence
from src.core.logger import get_logger
from src.pipelines.binance_klines.downloader import (
    AssetType,
    Granularity,
    PartitionInterval,
    fetch_api,
    fetch_historical,
)
from src.pipelines.binance_klines.exceptions import (
    DataNotFoundError,
    IncompleteUpdateError,
)
from src.pipelines.binance_klines.models import BackfillTask, KlineTask, UpdateTask
from src.pipelines.binance_klines.storage import save_monthly_data, get_last_timestamp

# 全局配置
BASE_DATA_PATH = Path("data/cleaned")
TICKERS_FILE = BASE_DATA_PATH / "binance_tickers_perp.parquet"
KLINES_ROOT_PATH = BASE_DATA_PATH / "binance_klines_perp_m1"

# 币安合约历史数据仓库的最早有效日期
EARLIEST_REPO_DATE = dt.date(2020, 1, 1)

logger = get_logger("kline")
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
        df = df[df["symbol"].astype(str).isin(symbols_upper)]
    return df


def clean_kline_data(df: pd.DataFrame) -> pd.DataFrame:
    """清洗k线数据，仅保留核心字段。

    Args:
        df: 原始k线DataFrame（索引为datetime）

    Returns:
        清洗后的DataFrame
    """
    cleaned_df = (
        df.loc[:, ["open", "high", "low", "close", "volume"]]
        .reset_index()
        .rename(columns={"index": "datetime"})
    )
    return cleaned_df


def generate_months(start_date: dt.date, end_date: dt.date) -> list[str]:
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


def generate_dates(start_dt: dt.date, end_date: dt.date) -> list[dt.date]:
    """生成日期列表。"""
    dates = []
    curr = start_dt
    while curr <= end_date:
        dates.append(curr)
        curr += dt.timedelta(days=1)
    return dates


def read_existing_month_data(symbol: str, period: str) -> pd.DataFrame:
    """读取现有月度数据。

    Args:
        symbol: 交易对
        period: 期间 YYYY-MM

    Returns:
        现有DataFrame
    """
    query = f"SELECT * FROM read_parquet('{KLINES_ROOT_PATH}/symbol={symbol}/period={period}/data.parquet')"
    try:
        with duckdb.connect() as con:
            df = con.sql(query).df()
        return df
    except Exception:
        return pd.DataFrame(
            columns=["datetime", "open", "high", "low", "close", "volume"]
        )


class BinanceKlinePipeline:
    """Binance 永续合约 K线数据管道类。

    统一管理历史回填和增量更新的调度逻辑。
    """

    def __init__(
        self, max_workers: int = 5, proxy: str | None = None, max_retries: int = 3
    ):
        """初始化管道。

        Args:
            max_workers: 最大并发线程数
            proxy: 代理设置
            max_retries: 最大重试轮数
        """
        self.max_workers = max_workers
        self.proxy = proxy
        self.max_retries = max_retries
        self.session = requests.Session()

        # 状态记录
        self.failed_tasks: list[dict] = []
        self.missing_data: list[dict] = []
        self.success_count = 0

    def _execute_with_retry(self, tasks: Sequence[KlineTask], worker_func):
        """通用的多轮重试调度器。

        Args:
            tasks: 初始任务列表
            worker_func: 工作函数
        """
        current_tasks = list(tasks)
        for attempt in range(self.max_retries):
            logger.info(f"第 {attempt + 1} 轮执行，任务数: {len(current_tasks)}")

            # 清空当前失败列表，为本轮收集做准备
            self.failed_tasks.clear()

            self._process_tasks(current_tasks, worker_func)

            if not self.failed_tasks:
                logger.info(f"第 {attempt + 1} 轮执行完毕，无失败任务")
                break

            logger.warning(
                f"第 {attempt + 1} 轮执行完毕，失败任务数: {len(self.failed_tasks)}"
            )

            # 转换失败任务为下一轮的任务
            next_tasks = []
            for t_info in self.failed_tasks:
                task_obj = t_info.get("task_obj")
                if isinstance(task_obj, KlineTask):
                    retry_task = task_obj.create_retry_task(t_info)
                    if retry_task:
                        next_tasks.append(retry_task)

            current_tasks = next_tasks

        if self.failed_tasks:
            logger.error(f"达到最大重试次数，最终失败任务数: {len(self.failed_tasks)}")

    def _parse_params(
        self, symbols: str | None, start_date: str | None, end_date: str | None
    ) -> tuple[pd.DataFrame, dt.date | None, dt.date]:
        """统一解析 CLI 参数。

        Returns:
            (tickers_df, start_date, end_date)
        """
        symbol_list = symbols.split(",") if symbols else None
        start = (
            dt.datetime.strptime(start_date, "%Y%m%d").date() if start_date else None
        )
        end = (
            dt.datetime.strptime(end_date, "%Y%m%d").date()
            if end_date
            else dt.date.today()
        )

        tickers_df = get_active_tickers(symbol_list)
        return tickers_df, start, end

    def run_backfill(
        self,
        symbols: str | None,
        start_date: str | None,
        end_date: str | None,
    ):
        """历史数据回填。"""
        tickers_df, start_dt, end_dt = self._parse_params(symbols, start_date, end_date)
        if tickers_df.empty:
            logger.warning("未找到活跃交易对")
            return

        tasks = []
        for _, row in tickers_df.iterrows():
            symbol = row["symbol"]
            onboard_dt = pd.to_datetime(row["onboard_date"]).date()
            valid_onboard = max(onboard_dt, EARLIEST_REPO_DATE)

            effective_start = (
                max(start_dt, valid_onboard) if start_dt else valid_onboard
            )

            months = generate_months(effective_start, end_dt)
            for period in months:
                tasks.append(
                    BackfillTask(
                        symbol=symbol, onboard_date=valid_onboard, period=period
                    )
                )

        logger.info(f"开始回填，共{len(tasks)}个任务，使用{self.max_workers}线程")

        self._execute_with_retry(tasks, worker_func=self._download_month_task)
        self._generate_report("backfill")

    def run_update(
        self,
        symbols: str | None,
        start_date: str | None,
        end_date: str | None,
    ):
        """增量数据更新。"""
        tickers_df, start_dt, end_dt = self._parse_params(symbols, start_date, end_date)
        if tickers_df.empty:
            logger.warning("未找到活跃交易对")
            return

        logger.info(
            f"开始更新，共{len(tickers_df)}个交易对，使用{self.max_workers}线程"
        )

        tasks = []
        for _, row in tickers_df.iterrows():
            symbol = row["symbol"]
            onboard_dt = pd.to_datetime(row["onboard_date"]).date()

            # 确定该交易对的实际更新范围
            last_ts = get_last_timestamp(symbol)
            if last_ts:
                last_dt = last_ts.date()
            else:
                last_dt = max(onboard_dt, EARLIEST_REPO_DATE.date())

            effective_start = start_dt or last_dt
            effective_end = end_dt

            if effective_start > effective_end:
                logger.info(
                    f"{symbol} 无需更新 (范围: {effective_start} -> {effective_end})"
                )
                continue

            dates = generate_dates(effective_start, effective_end)
            tasks.append(
                UpdateTask(symbol=symbol, onboard_date=onboard_dt, dates=dates)
            )

        if not tasks:
            logger.info("没有需要执行的更新任务")
            return

        self._execute_with_retry(tasks, worker_func=self._update_symbol_task)
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

    def _download_month_task(self, task: BackfillTask):
        """下载单个交易对月度数据任务。"""
        symbol = task.symbol
        period = task.period

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
                    "task_obj": task,
                    "symbol": symbol,
                    "period": period,
                    "error": str(e),
                }
            )
            raise  # 重新抛出让调度器捕获

    def _update_symbol_task(self, task: UpdateTask):
        """更新单个交易对的最新数据任务。"""
        symbol = task.symbol
        dates_to_process = task.dates

        if not dates_to_process:
            logger.info(f"{symbol} 无需处理 (日期列表为空)")
            return

        failed_dates = []
        merged_data = {}  # period -> df

        try:
            for current in dates_to_process:
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
                except DataNotFoundError:
                    # 如果 ZIP 不存在，尝试 API
                    try:
                        start_ts = dt.datetime.combine(
                            current, dt.time.min, tzinfo=dt.timezone.utc
                        )
                        end_ts = dt.datetime.combine(
                            current, dt.time.max, tzinfo=dt.timezone.utc
                        )
                        df = fetch_api(symbol, start_ts, end_ts, proxy=self.proxy)
                    except Exception as e:
                        logger.error(f"下载{symbol} {current} API 出错: {e}")
                        failed_dates.append(current)
                except Exception as e:
                    logger.error(f"下载{symbol} {current} ZIP 出错: {e}")
                    failed_dates.append(current)

                if df is not None and not df.empty:
                    df = clean_kline_data(df)
                    data_to_merge = merged_data[period]
                    if data_to_merge.empty:
                        merged_data[period] = df
                    else:
                        merged_data[period] = pd.concat(
                            [data_to_merge, df]
                        ).drop_duplicates(subset="datetime", keep="last")

            # 保存每个月的合并数据
            for period, df in merged_data.items():
                if not df.empty:
                    df = df.sort_values("datetime")
                    save_monthly_data(df, symbol, period)
                    logger.info(f"{symbol} {period} 更新完成")

            if failed_dates:
                raise IncompleteUpdateError(
                    f"{symbol} 有 {len(failed_dates)} 天更新失败",
                    symbol=symbol,
                    failed_dates=failed_dates,
                )

        except IncompleteUpdateError as e:
            self.failed_tasks.append(
                {
                    "task_obj": task,
                    "symbol": symbol,
                    "failed_dates": e.failed_dates,
                    "error": str(e),
                }
            )
            raise
        except Exception as e:
            # 其他严重错误，记录整个任务失败
            self.failed_tasks.append(
                {
                    "task_obj": task,
                    "symbol": symbol,
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

        report_file = BASE_DATA_PATH / f"binance_klines_{operation}_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"报告已保存到 {report_file}")


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
