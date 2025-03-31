from pathlib import Path

from rich import print

from bgeometrics import BGClient


def download_blockchain_metrics(data_directory: Path) -> None:
    """从 bgeometrics 下载区块链数据"""
    metrics = [
        "sth_realized_price",
        "sth_sopr",
        "sth_nupl",
        "sth_mvrv",
        "miner_sell_presure",
        "nrpl",
        "realized_profit_loss_ratio",
    ]

    client = BGClient()

    for metric in metrics:
        try:
            df = client.get_metric(metric)
            print(f"Download metric {metric}: done", style="green bold")
        except Exception as e:
            print(f"Failed to fet {metric}: {str(e)}", style="red bold")
        else:
            filepath = data_directory / f"{metric}.csv"
            df.to_csv(filepath, index=True)


def main() -> None:
    # 参数
    data_directory = Path("data")

    # 创建数据目录
    data_directory.mkdir(parents=True, exist_ok=True)

    # 下载数据
    download_blockchain_metrics(data_directory)


if __name__ == "__main__":
    main()
