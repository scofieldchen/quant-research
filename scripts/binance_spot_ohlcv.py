import requests
import pandas as pd


tickers = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "SOLUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "TRXUSDT",
    "LINKUSDT",
    "AVAXUSDT",
    "MATICUSDT",
    "DOTUSDT",
    "LTCUSDT",
    "SHIBUSDT",
    "BCHUSDT",
    "UNIUSDT",
    "ATOMUSDT",
    "XLMUSDT",
    "XMRUSDT",
    "ETCUSDT",
    "FILUSDT",
    "LDOUSDT",
    "HBARUSDT",
    "ICPUSDT",
    "APTUSDT",
    "RUNEUSDT",
    "NEARUSDT",
    "IMXUSDT",
    "VETUSDT",
    "OPUSDT",
    "MKRUSDT",
    "INJUSDT",
    "GRTUSDT",
    "AAVEUSDT",
    "ARBUSDT",
    "QNTUSDT",
    "RNDRUSDT",
    "EGLDUSDT",
    "ALGOUSDT",
]


def get_daily_ohlcv(ticker: str) -> pd.DataFrame:
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": ticker,
        "interval": "1d",
        "limit": 1000,
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    return process_ohlcv(data)


def process_ohlcv(data: list) -> pd.DataFrame:
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "num_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]
    df = pd.DataFrame.from_records(data, columns=columns)
    return (
        df.assign(open_time=pd.to_datetime(df.open_time, unit="ms"))
        .drop(columns=["close_time", "ignore"])
        .drop_duplicates(subset="open_time", keep="first")
        .set_index("open_time")
        .astype(float)
    )


df = get_daily_ohlcv("BTCUSDT")
print(df.head())
print(df.tail())
print(df.info())
