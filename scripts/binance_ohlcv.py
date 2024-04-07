import ccxt
import pandas as pd


# 市值排名前30的货币对，binance
tickers = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "DOGE/USDT",
    "ADA/USDT",
    "AVAX/USDT",
    "SHIB/USDT",
    "DOT/USDT",
    "LINK/USDT",
    "TRX/USDT",
    "MATIC/USDT",
    "BCH/USDT",
    "ICP/USDT",
    "NEAR/USDT",
    "UNI/USDT",
    "APT/USDT",
    "LTC/USDT",
    "STX/USDT",
    "FIL/USDT",
    "ATOM/USDT",
    "ETC/USDT",
    "ARB/USDT",
    "RONIN/USDT",
    "RNDR/USDT",
    "IMX/USDT",
    "HBAR/USDT",
    "OP/USDT",
    "XLM/USDT",
    "GRT/USDT",
    "INJ/USDT",
    "PEPE/USDT",
    "RUNE/USDT",
    "VET/USDT",
    "WIF/USDT",
    "THETA/USDT",
    "FTM/USDT",
    "MKR/USDT",
    "LDO/USDT",
    "AR/USDT",
    "SUI/USDT",
    "FET/USDT",
    "TIA/USDT",
    "SEI/USDT",
    "FLOKI/USDT",
    "ALGO/USDT",
    "FLOW/USDT",
    "GALA/USDT",
    "AAVE/USDT",
    "CFX/USDT",
    "JUP/USDT",
    "STRK/USDT",
    "BONK/USDT",
    "EGLD/USDT",
    "QNT/USDT",
    "DYDX/USDT",
    "AXS/USDT",
    "SNX/USDT",
    "SAND/USDT",
    "AGIX/USDT",
    "PYTH/USDT",
    "MINA/USDT",
    "ORDI/USDT",
    "XTZ/USDT",
    "WLD/USDT",
    "MANA/USDT",
    "CHZ/USDT",
    "AXL/USDT",
    "XEC/USDT",
    "APE/USDT",
    "EOS/USDT",
    "IOTA/USDT",
    "NEO/USDT",
    "KAVA/USDT",
    "JASMY/USDT",
    "CAKE/USDT",
    "PENDLE/USDT",
    "ROSE/USDT",
    "KLAY/USDT",
    "ZRX/USDT",
    "GNO/USDT",
    "LUNC/USDT",
    "BLUR/USDT",
    "CKB/USDT",
    "WOO/USDT",
    "OSMO/USDT",
    "DYM/USDT",
    "CRV/USDT",
]


# tickers = [
#     "BTCUSDT",
#     "ETHUSDT",
#     "BNBUSDT",
#     "XRPUSDT",
#     "SOLUSDT",
#     "ADAUSDT",
#     "DOGEUSDT",
#     "TRXUSDT",
#     "LINKUSDT",
#     "AVAXUSDT",
#     "MATICUSDT",
#     "DOTUSDT",
#     "LTCUSDT",
#     "SHIBUSDT",
#     "BCHUSDT",
#     "UNIUSDT",
#     "ATOMUSDT",
#     "XLMUSDT",
#     "XMRUSDT",
#     "ETCUSDT",
#     "FILUSDT",
#     "LDOUSDT",
#     "HBARUSDT",
#     "ICPUSDT",
#     "APTUSDT",
#     "RUNEUSDT",
#     "NEARUSDT",
#     "IMXUSDT",
#     "VETUSDT",
#     "OPUSDT",
#     "MKRUSDT",
#     "INJUSDT",
#     "GRTUSDT",
#     "AAVEUSDT",
#     "ARBUSDT",
#     "QNTUSDT",
#     "RNDRUSDT",
#     "EGLDUSDT",
#     "ALGOUSDT",
# ]


# def get_daily_ohlcv(ticker: str) -> pd.DataFrame:
#     url = "https://api.binance.com/api/v3/klines"
#     params = {
#         "symbol": ticker,
#         "interval": "1d",
#         "limit": 1000,
#     }
#     response = requests.get(url, params=params)
#     response.raise_for_status()
#     data = response.json()
#     return process_ohlcv(data)


# def process_ohlcv(data: list) -> pd.DataFrame:
#     columns = [
#         "open_time",
#         "open",
#         "high",
#         "low",
#         "close",
#         "volume",
#         "close_time",
#         "quote_volume",
#         "num_trades",
#         "taker_buy_base_volume",
#         "taker_buy_quote_volume",
#         "ignore",
#     ]
#     df = pd.DataFrame.from_records(data, columns=columns)
#     return (
#         df.assign(open_time=pd.to_datetime(df.open_time, unit="ms"))
#         .drop(columns=["close_time", "ignore"])
#         .drop_duplicates(subset="open_time", keep="first")
#         .set_index("open_time")
#         .astype(float)
#     )


# df = get_daily_ohlcv("BTCUSDT")
# print(df.head())
# print(df.tail())
# print(df.info())
