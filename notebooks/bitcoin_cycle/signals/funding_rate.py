from typing import List, Union

import pandas as pd
import plotly.graph_objects as go

from .base import Metric
from .utils import add_percentile_bands, calculate_percentile_bands


class FundingRate(Metric):
    """融资利率指标"""

    @property
    def name(self) -> str:
        return "Funding Rate"

    @property
    def description(self) -> str:
        pass

    def __init__(
        self,
        data: pd.DataFrame,
        price_col: str = "btcusd",
        metric_cols: Union[str, List[str]] = "funding_rate",
        cumulative_days: int = 30,
        rolling_period: int = 200,
        upper_band_percentile: float = 0.95,
        lower_band_percentile: float = 0.05,
    ) -> None:
        """
        初始化指标类

        Args:
            data: 包含 NRPL 数据的 DataFrame
        """
        self.cumulative_days = cumulative_days
        self.rolling_period = rolling_period
        self.upper_band_percentile = upper_band_percentile
        self.lower_band_percentile = lower_band_percentile
        super().__init__(data, price_col, metric_cols)

    def generate_signals(self) -> None:
        data = self.data.copy()
        data["cum_rate"] = (
            data["funding_rate"].rolling(pd.Timedelta(days=self.cumulative_days)).sum()
        )
        self.signals = calculate_percentile_bands(
            data=data,
            input_col="cum_rate",
            rolling_period=self.rolling_period,
            upper_band_percentile=self.upper_band_percentile,
            lower_band_percentile=self.lower_band_percentile,
        )

    def _add_indicator_traces(self, fig: go.Figure) -> None:
        add_percentile_bands(
            fig=fig,
            data=self.signals,
            metric_col="cum_rate",
            yaxis_title="Funding rate",
            metric_name=f"Cumulative funding rate({self.cumulative_days}-days)",
        )
