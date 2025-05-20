import datetime as dt
from pathlib import Path

import pandas as pd
import yfinance as yf
from rich.console import Console

from source.alternative import get_fear_greed_index
from source.bgeometrics import BGClient
from source.binance import HistoricalFutureMetricsDownloader

console = Console()

# 若无法从雅虎财经下载数据，设置 vpn 作为网络代理
yf.set_config(proxy={"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"})


def download_ohlcv(data_directory: Path) -> None:
    """从雅虎财经下载历史价格"""
    console.print("[green bold]Downloading ohlcv...")

    tickers = ["BTC-USD", "ETH-USD"]
    for ticker in tickers:
        try:
            data = yf.download(
                tickers=ticker,
                ignore_tz=True,
                auto_adjust=True,
                progress=False,
                multi_level_index=False,
            )
            data = data[["Open", "High", "Low", "Close", "Volume"]]
            data.index.name = "datetime"
            data.columns = [x.lower() for x in data.columns]
            console.print(f"✅ Downloaded {ticker}, last:{data.index.max():%Y-%m-%d}")
        except Exception as e:
            console.print(f"❌ Failed to download {ticker}: {str(e)}")
        else:
            filepath = data_directory / f"{ticker.replace("-", "").lower()}.csv"
            data.to_csv(filepath, index=True)


def download_blockchain_metrics(data_directory: Path) -> None:
    """从 bgeometrics 下载区块链数据"""
    console.print("[green bold]Downloading block metrics...")

    client = BGClient()
    metrics = [
        "sth_realized_price",
        "sth_sopr",
        "sth_nupl",
        "sth_mvrv",
        "nrpl",
    ]

    for metric in metrics:
        try:
            df = client.get_metric(metric)
            console.print(f"✅ Downloaded {metric}, last:{df.index.max():%Y-%m-%d}")
        except Exception as e:
            console.print(f"❌ Failed to download {metric}: {str(e)}")
        else:
            filepath = data_directory / f"{metric}.csv"
            df.to_csv(filepath, index=True)


def download_fgi(data_directory: Path) -> None:
    """下载恐慌和贪婪指数"""
    console.print("[green bold]Downloading fear greed index...")

    try:
        df = get_fear_greed_index(limit=10 * 365)
        console.print(
            f"✅ Downloaded fear and greed index, last:{df.index.max():%Y-%m-%d}"
        )
    except Exception as e:
        console.print(f"❌ Failed to get fear greed index: {str(e)}")
    else:
        filepath = data_directory / "fear_greed_index.csv"
        df.to_csv(filepath, index=True)


def load_lsr(data_dir: Path) -> pd.DataFrame:
    """从本地加载原始的历史多空比例数据"""
    csv_files = sorted(data_dir.glob("*.csv"))

    if not csv_files:
        raise Exception(f"No data found in the directory: {str(data_dir)}")

    # 过滤2022年以前的文件，因为其不包含多空比例数据
    csv_files_2023_and_later = [file for file in csv_files if file.stem >= "20230101"]

    return pd.concat(pd.read_csv(f) for f in csv_files_2023_and_later)


def process_lsr(df: pd.DataFrame) -> pd.DataFrame:
    """处理多空比例，转化为日频数据"""
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


def download_lsr(data_directory: Path) -> None:
    """下载多空比例数据"""
    console.print("[green bold]Downloading long short ratio...")

    # 获取历史数据的最后一天
    raw_data_dir = data_directory / "metrics" / "BTCUSDT"
    csv_files = sorted(raw_data_dir.glob("*.csv"))
    last_date = dt.datetime.strptime(csv_files[-1].stem, "%Y%m%d")
    last_date = last_date.replace(tzinfo=dt.timezone.utc)

    # 更新数据的日期范围
    start_date = last_date + dt.timedelta(days=1)
    end_date = dt.datetime.now(tz=dt.timezone.utc)
    if start_date > end_date:
        console.print("Long short ratio is already up to date.")
        return
    console.print(f"Update range: {start_date:%Y%m%d} -> {end_date:%Y%m%d}")

    # 下载最新数据
    downloader = HistoricalFutureMetricsDownloader(data_directory / "metrics")
    downloader.download("BTCUSDT", start_date, end_date, max_workers=1)

    # 读取和处理数据
    lsr = load_lsr(raw_data_dir)
    daily_lsr = process_lsr(lsr)

    # 删除时间索引的时区信息，跟价格数据保持一致，时区默认为 UTC
    daily_lsr.index = daily_lsr.index.tz_convert(None)

    # 存储数据
    daily_lsr.to_csv(data_directory / "long_short_ratio.csv", index=True)

    console.print(
        f"✅ Downloaded long short ratio, last:{daily_lsr.index.max():%Y-%m-%d}"
    )


def main() -> None:
    # 参数
    data_directory = Path("data")

    # 创建数据目录
    data_directory.mkdir(parents=True, exist_ok=True)

    # 下载数据
    download_ohlcv(data_directory)
    download_blockchain_metrics(data_directory)
    download_fgi(data_directory)
    download_lsr(data_directory)


if __name__ == "__main__":
    main()
