import datetime as dt
import os

import pandas as pd
from binance.client import Client
from dotenv import load_dotenv

load_dotenv()

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

start_time = dt.datetime(2024, 1, 1)
end_time = dt.datetime(2024, 12, 1)


# history = client.get_margin_interest_history(
#     asset="BTC",  # 借款资产
#     isolatedSymbol=None,  # 逐仓账户货币对，全仓账户不需要
#     startTime=int(start_time.timestamp() * 1000),  # 开始时间，时间戳，毫秒
#     endTime=int(end_time.timestamp() * 1000),  # 结束时间，时间戳，毫秒
#     current=1,  # 当前页数，默认1
#     size=100,  # 每页数量，默认10，最大100
#     archived=True,  # 是否查询已归档数据，默认false，若想查询6个月以前的数据则设置为true
#     recvWindows=5000,  # 交易窗口，毫秒
# )
# print(history)

response = client._request_margin_api(
    "post",
    "/sapi/v1/margin/interestRateHistory",
    True,
    data={
        "asset": "BTC",
        # "timestamp": int(dt.datetime.now().timestamp() * 1000),
        # "startTime": int(start_time.timestamp() * 1000),
        # "endTime": int(end_time.timestamp() * 1000),
    },
)
print(response)
