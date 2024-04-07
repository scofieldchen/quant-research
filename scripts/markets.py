import math
import time

import requests


COINGECKO_API_URL = "https://api.coingecko.com/api/v3"


def get_coins(page: int = 1) -> list[dict]:
    url = f"{COINGECKO_API_URL}/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 250,
        "page": page,
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    return [
        {
            "coingecko_id": coin["id"],
            "symbol": coin["symbol"],
            "name": coin["name"],
            "current_price": coin["current_price"],
            "market_cap": coin["market_cap"],
            "market_cap_rank": coin["market_cap_rank"],
        }
        for coin in data
    ]


def get_top_coins(num: int = 100) -> list[dict]:
    pages = math.ceil(num / 250)
    coins = []
    for page in range(1, pages + 1):
        coins.extend(get_coins(page))
        time.sleep(1)

    return coins[:num]


def get_coin_category(coingecko_id: str) -> list[str]:
    url = f"{COINGECKO_API_URL}/coins/{coingecko_id}"
    params = {
        "localization": False,
        "tickers": False,
        "market_data": False,
        "community_data": False,
        "developer_data": False,
        "sparkline": False,
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    return data["categories"]


# coins = get_top_coins(500)
# pprint(coins)
# print(len(coins))
category = get_coin_category("ethereum")
print(category)
