import pandas as pd
import requests


def get_fear_greed_index(limit: int = 10, timeout: int = 5) -> pd.DataFrame:
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


if __name__ == "__main__":
    df = get_fear_greed_index(10000)
    print(df.tail())
    print(df.info())
