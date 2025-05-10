from pathlib import Path

import yfinance as yf
from rich.console import Console

from bgeometrics import BGClient
from alternative import get_fear_greed_index

console = Console()

# 若无法从雅虎财经下载数据，设置本地 vpn 作为网络代理
yf.set_config(proxy={"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"})


def download_ohlcv(data_directory: Path) -> None:
    """从雅虎财经下载历史价格"""
    console.print("\n====== Price info ======", style="bold blue")

    try:
        data = yf.download(
            tickers="BTC-USD",
            ignore_tz=True,
            auto_adjust=True,
            progress=False,
            multi_level_index=False,
        )
        data = data[["Open", "High", "Low", "Close", "Volume"]]
        data.index.name = "datetime"
        data.columns = [x.lower() for x in data.columns]
        console.print(f"✅ Downloaded BTCUSD, last:{data.index.max():%Y-%m-%d}")
    except Exception as e:
        console.print(f"❌ Failed to get btcusd ohlcv: {str(e)}")
    else:
        filepath = data_directory / "btcusd.csv"
        data.to_csv(filepath, index=True)


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

    console.print("\n====== Blockchain metrics ======", style="bold blue")

    for metric in metrics:
        try:
            df = client.get_metric(metric)
            console.print(f"✅ Downloaded {metric}, last:{df.index.max():%Y-%m-%d}")
        except Exception as e:
            console.print(f"❌ Failed to fet {metric}: {str(e)}")
        else:
            filepath = data_directory / f"{metric}.csv"
            df.to_csv(filepath, index=True)


def download_fgi(data_directory: Path) -> None:
    """下载恐慌和贪婪指数"""
    console.print("\n====== Sentiment metrics ======", style="bold blue")

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


def main() -> None:
    # 参数
    data_directory = Path("data")

    # 创建数据目录
    data_directory.mkdir(parents=True, exist_ok=True)

    # 下载数据
    download_ohlcv(data_directory)
    download_blockchain_metrics(data_directory)
    download_fgi(data_directory)


if __name__ == "__main__":
    main()
