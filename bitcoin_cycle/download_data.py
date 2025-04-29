import datetime as dt
from pathlib import Path

import yfinance as yf
from rich.console import Console

from bgeometrics import BGClient

console = Console()


def download_ohlcv(data_directory: Path) -> None:
    """从雅虎财经下载历史价格"""
    ticker = "BTC-USD"
    start_date = dt.datetime(2010, 1, 1)
    end_date = dt.datetime.today()

    console.print("\n====== Download ohlcv ======", style="bold blue")

    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        ignore_tz=True,
        progress=False,
        auto_adjust=True,
    )
    data.columns = data.columns.droplevel(1)  # 将多重索引转化为简单索引
    data = data[["Open", "High", "Low", "Close", "Volume"]]
    data.index.name = "datetime"
    data.columns = [x.lower() for x in data.columns]

    filepath = data_directory / f"{ticker.replace("-", "").lower()}.csv"
    data.to_csv(filepath, index=True)

    console.print(f"✅ Downloaded {ticker}", style="green bold")


def download_blockchain_metrics(data_directory: Path) -> None:
    """从 bgeometrics 下载区块链数据"""
    metrics = [
        "sth_realized_price",
        "sth_sopr",
        "sth_nupl",
        "sth_mvrv",
        "nrpl",
    ]

    client = BGClient()

    console.print("\n====== Download blockchain metrics ======", style="bold blue")

    for metric in metrics:
        try:
            df = client.get_metric(metric)
            console.print(f"✅ Downloaded {metric}", style="green bold")
        except Exception as e:
            console.print(f"❌ Failed to fet {metric}: {str(e)}", style="red bold")
        else:
            filepath = data_directory / f"{metric}.csv"
            df.to_csv(filepath, index=True)


def main() -> None:
    # 参数
    data_directory = Path("data")

    # 创建数据目录
    data_directory.mkdir(parents=True, exist_ok=True)

    # 下载数据
    download_ohlcv(data_directory)
    download_blockchain_metrics(data_directory)


if __name__ == "__main__":
    main()
