"""
Download cryptocurrency names and sector information from CoinGecko API.
"""

import math
import time
import json
import os

from dotenv import load_dotenv
import requests


COINGECKO_API_URL = "https://api.coingecko.com/api/v3"


def get_coins(
    api_key: str,
    vs_currency: str = "usd",
    order: str = "market_cap_desc",
    per_page: int = 250,
    page: int = 1,
) -> list[dict]:
    """Get a list of coins from the CoinGecko API.

    Args:
        api_key (str): The CoinGecko API key.
        vs_currency (str): The currency to display the data in.
        order (str): The order to sort the data.
        per_page (int): The number of coins to display per page.
        page (int): The page number to retrieve.

    Returns:
        list[dict]: A list of coin data.
    """
    resp = requests.get(
        f"{COINGECKO_API_URL}/coins/markets",
        params={
            "vs_currency": vs_currency,
            "order": order,
            "per_page": per_page,
            "page": page,
            "x_cg_demo_api_key": api_key,
        },
    )
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


def get_top_coins(num: int, api_key: str) -> list[dict]:
    """Get the top N coins from the CoinGecko API."""
    pages = math.ceil(num / 250)
    coins = []
    for page in range(1, pages + 1):
        coins.extend(get_coins(api_key, page=page))

    return coins[:num]


def get_coin_category(coingecko_id: str, api_key: str) -> list[str]:
    """Get the category of a coin from the CoinGecko API."""
    resp = requests.get(
        f"{COINGECKO_API_URL}/coins/{coingecko_id}",
        params={
            "localization": False,
            "tickers": False,
            "market_data": False,
            "community_data": False,
            "developer_data": False,
            "sparkline": False,
            "x_cg_demo_api_key": api_key,
        },
    )
    resp.raise_for_status()
    data = resp.json()
    return data["categories"]


def main():
    # Script inputs
    num_coins = 200  # Number of top coins to retrieve
    json_file = "../data/coingecko_coins.json"  # File to save the data

    # Load coingecko api key
    load_dotenv()
    api_key = os.getenv("COINGECKO_API_KEY")

    t0 = time.time()

    # Get id,name,marketcap of top coins
    coins = get_top_coins(num_coins, api_key)
    print(f"Get top {len(coins)} coins")

    # Get sector information for each coin
    for coin in coins:
        try:
            category = get_coin_category(coin["coingecko_id"], api_key)
        except Exception as e:
            print(f"{coin['name']}: {e}")
        else:
            coin.update({"category": category})
            print(f"{coin['name']}: update sector information")
        time.sleep(2.5)  # Avoid rate limiting, 30 calls per minute

    # Save the data to local json file
    with open(json_file, "w") as f:
        json.dump(coins, f, indent=4)
        print("Save the data to local json file")

    print(f"Time taken: {time.time() - t0:.2f} seconds")


if __name__ == "__main__":
    main()
