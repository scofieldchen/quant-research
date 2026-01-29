## 任务

优化数据管道`src/pipelines/binance_tickers/task.py`。增加字段：`coingecko_market_cap`。

## 背景和目标

该数据管道从binance api获取永续合约交易对的信息，但是交易对过多，不利于做横截面分析。

我们需要补充一些数据，例如总市值，这样就可以对交易对进行筛选，横截面分析模型可以选择市值排名前30或者前100的主流交易对。

## 核心逻辑

1. 从binance api获取永续合约的交易对信息。
2. 从coingecko获取币种的市值信息。
3. 合并数据，添加新字段。
4. 保存数据到`data/cleaned/binance_tickers_perp.parquet`。

原来的数据获取和清洗逻辑保持不变，关键是添加新字段，获取市值信息。

## 参考代码

API信息存储在项目根目录的`.env`文件，`COINGECKO_API_KEY`.

```python
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
```
