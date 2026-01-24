"""Binance k线任务数据模型。"""

from __future__ import annotations
import datetime as dt
from dataclasses import dataclass, field


@dataclass
class KlineTask:
    """K线下载任务基类。"""

    symbol: str
    onboard_date: dt.date

    def create_retry_task(self, error_info: dict) -> KlineTask | None:
        """根据错误信息创建重试任务。"""
        raise NotImplementedError


@dataclass
class BackfillTask(KlineTask):
    """历史回填任务（按月）。"""

    period: str  # YYYY-MM

    def create_retry_task(self, error_info: dict) -> BackfillTask | None:
        return BackfillTask(
            symbol=self.symbol, onboard_date=self.onboard_date, period=self.period
        )


@dataclass
class UpdateTask(KlineTask):
    """增量更新任务。"""

    dates: list[dt.date] = field(default_factory=list)

    def create_retry_task(self, error_info: dict) -> UpdateTask | None:
        failed_dates = error_info.get("failed_dates")
        if failed_dates:
            # 如果有具体的失败日期，仅重试这些日期
            return UpdateTask(
                symbol=self.symbol,
                onboard_date=self.onboard_date,
                dates=sorted(failed_dates),
            )
        # 如果是整体任务失败，重试原有的所有日期
        return UpdateTask(
            symbol=self.symbol, onboard_date=self.onboard_date, dates=self.dates
        )
