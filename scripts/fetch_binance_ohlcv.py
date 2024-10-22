import os
import json
import datetime as dt

import ccxt
import pandas as pd

from fetch_exchange_ohlcv import get_ohlcv


def main():
    # 参数
    exchange_id = "binance"  # ccxt exchange id
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "FTM/USDT"]
    timeframe = ["4h", "1d"]
    start_date = dt.datetime(2017, 1, 1)
    end_date = dt.datetime.today()
    output_dir = "../data/binance"

    # 创建存储目录
    os.makedirs(output_dir, exist_ok=True)

    # 下载数据
    exchange = getattr(ccxt, exchange_id)()


if __name__ == "__main__":
    main()
