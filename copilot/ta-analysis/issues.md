## todo

将交易对数量增加到30-50个。选择市场最大的主流交易对。

创建`marimo notebook`，自动筛选主流交易对。

1. 从`coingecko`获取市场排名最高的k个货币对。
2. 从`/data/cleaned/binance_tickers_perp.parquet`加载数据。
3. 筛选能够在币安进行永续合约交易的市场最高的交易对。
  - 能够交易：status = 'TRADING'
  - 上市日期：早于2024年
  - 计价货币：USDT
  - 排除那些稳定币如`USDC`, `FUSD`等
    
4. 使用数据框展示结果。

辅助代码：

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

更新数据，创作内容，发布内容。

该指标在识别中期顶部和底部方面似乎有价值，值得进一步研究，引入统计模型。

问题：当指标超过阈值，未来7天涨跌概率？参考收益率曲线研究。
