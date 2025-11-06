import datetime as dt
from pathlib import Path

from downloader.alternative import download_fear_greed_index
from downloader.blockchain import download_blockchain_metrics
from downloader.funding_rate import download_funding_rate
from downloader.ohlcv import download_ohlcv
from downloader.sentiment import download_lsr
from rich.console import Console

console = Console()


def main() -> None:
    # 参数
    data_directory = Path("/users/scofield/quant-research/notebooks/bitcoin_cycle/data")

    # 创建数据目录
    data_directory.mkdir(parents=True, exist_ok=True)

    # 最新日期
    end_date = dt.datetime.now(tz=dt.timezone.utc)

    # 下载数据
    download_ohlcv(
        filepath=data_directory / "btcusd.csv",
        symbol="BTC-USD",
        start_date=dt.datetime(2014, 1, 1, tzinfo=dt.timezone.utc),
        end_date=end_date,
    )

    download_blockchain_metrics(
        data_directory,
        metric_names=[
            "sth_realized_price",
            "sth_sopr",
            "sth_nupl",
            "sth_mvrv",
            "nrpl",
            "rhodl",
        ],
    )

    download_fear_greed_index(data_directory / "fear_greed_index.csv")

    download_lsr(data_directory, "BTCUSDT")

    download_funding_rate(
        filepath=data_directory / "funding_rate.csv",
        symbol="BTCUSDT",
        start_date=dt.datetime(2019, 1, 1, tzinfo=dt.timezone.utc),
        end_date=end_date,
    )


if __name__ == "__main__":
    main()
