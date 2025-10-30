from pathlib import Path

import pandas as pd
import requests
from rich.console import Console

console = Console()


def get_fear_greed_index(limit: int = 10, timeout: int = 10) -> pd.DataFrame:
    """从官网 api 获取恐慌和贪婪指数"""
    resp = requests.get(
        "https://api.alternative.me/fng/", params={"limit": limit}, timeout=timeout
    )
    resp.raise_for_status()
    data = resp.json()

    return (
        pd.DataFrame.from_records(data["data"], exclude=["time_until_update"])
        .astype({"value": "float64", "timestamp": "int"})
        .assign(datetime=lambda x: pd.to_datetime(x.timestamp, unit="s"))
        .sort_values("datetime", ascending=True)
        .set_index("datetime")
        .drop(columns=["timestamp"])
        .rename(columns={"value_classification": "sentiment", "value": "fgi"})
    )


def download_fear_greed_index(filepath: Path) -> None:
    """下载恐慌和贪婪指数的历史数据并保存到本地"""
    try:
        df = get_fear_greed_index(10 * 365)
        console.print(
            f"✅ Downloaded fear and greed index, last:{df.index.max():%Y-%m-%d}"
        )
    except Exception as e:
        console.print(f"[red]Failed to get fear greed index: {str(e)}")
    else:
        df.to_csv(filepath, index=True)


if __name__ == "__main__":
    download_fear_greed_index(
        Path("/users/scofield/quant-research/bitcoin_cycle/data/fear_greed_index.csv")
    )
