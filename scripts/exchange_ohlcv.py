"""
Download ohlcv data from exchanges
"""

import ccxt
import pandas as pd

# 初始化Bitstamp交易所
exchange = ccxt.bitstamp()

# 获取BTC/USD的日线历史数据
data = exchange.fetch_ohlcv("BTC/USD", "1d")

# 转换为pandas DataFrame
df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])

# 将时间戳转换为日期
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

print(df)
