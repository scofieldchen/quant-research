import os
import math
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

from src.core.logger import get_logger

# 加载环境变量
load_dotenv()

# 数据目录
RAW_DIR = Path("data/raw/binance_tickers")
CLEANED_DIR = Path("data/cleaned")
COINGECKO_API_URL = "https://api.coingecko.com/api/v3"

logger = get_logger("binance_tickers")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def fetch_exchange_info() -> dict:
    """
    从 Binance Futures API 获取交易所信息，包括所有交易对。

    Returns:
        Dict: 包含交易所信息的字典。

    Raises:
        requests.RequestException: 如果网络请求失败。
    """
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    logger.info("正在从 Binance API 获取交易所信息...")
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    # 保存原始数据
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    raw_file = RAW_DIR / f"exchange_info_{date_str}.json"
    with open(raw_file, "w", encoding="utf-8") as f:
        import json

        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"原始数据已保存到 {raw_file}")

    return data


def clean_perp_tickers(data: dict) -> pd.DataFrame:
    """
    清洗数据，过滤出永续合约交易对，并提取并转换字段。

    Args:
        data (dict): 原始 API 响应数据。

    Returns:
        pd.DataFrame: 清洗后的 DataFrame，包含转换后的字段。
    """
    logger.info("正在过滤永续合约...")
    symbols = data.get("symbols", [])
    perp_tickers = [
        {
            "symbol": s.get("symbol"),
            "pair": s.get("pair"),
            "contract_type": s.get("contractType"),
            "onboard_date": datetime.fromtimestamp(s.get("onboardDate") / 1000)
            if s.get("onboardDate")
            else None,
            "status": s.get("status"),
            "maint_margin_percent": s.get("maintMarginPercent"),
            "required_margin_percent": s.get("requiredMarginPercent"),
            "base_asset": s.get("baseAsset"),
            "quote_asset": s.get("quoteAsset"),
            "margin_asset": s.get("marginAsset"),
            "price_precision": s.get("pricePrecision"),
            "quantity_precision": s.get("quantityPrecision"),
        }
        for s in symbols
        if s.get("contractType") == "PERPETUAL"
    ]
    df = pd.DataFrame(perp_tickers)
    logger.info(f"已清洗 {len(df)} 个永续合约交易对")
    return df


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_coingecko_markets(api_key: str, page: int = 1) -> list[dict]:
    """
    从 CoinGecko 获取币种市场信息（包含市值）。

    Args:
        api_key: CoinGecko API Key.
        page: 分页页码.

    Returns:
        List[Dict]: 币种信息列表.
    """
    logger.info(f"正在从 CoinGecko 获取第 {page} 页市场信息...")
    resp = requests.get(
        f"{COINGECKO_API_URL}/coins/markets",
        params={
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 250,
            "page": page,
            "x_cg_demo_api_key": api_key,
        },
    )
    resp.raise_for_status()
    data = resp.json()
    return [
        {
            "coingecko_id": coin["id"],
            "symbol": coin["symbol"].upper(),
            "market_cap": coin["market_cap"],
            "market_cap_rank": coin["market_cap_rank"],
        }
        for coin in data
    ]


def get_top_coingecko_coins(api_key: str, num: int = 1000) -> pd.DataFrame:
    """
    获取市值前 N 的 CoinGecko 币种信息并转换为 DataFrame。
    """
    pages = math.ceil(num / 250)
    all_coins = []
    for page in range(1, pages + 1):
        all_coins.extend(fetch_coingecko_markets(api_key, page))

    df_cg = pd.DataFrame(all_coins)
    # 处理 symbol 重复：保留市值排名靠前的（rank较小的）
    df_cg = df_cg.sort_values("market_cap_rank").drop_duplicates("symbol", keep="first")
    return df_cg


def task():
    """
    主任务函数：获取并清洗 Binance 永续合约交易对数据，并补充市值信息。
    """
    logger.info("开始运行 binance_tickers 数据管道")

    # 1. 获取币安数据
    data = fetch_exchange_info()
    df_binance = clean_perp_tickers(data)

    # 2. 获取 CoinGecko 市值数据
    api_key = os.getenv("COINGECKO_API_KEY")
    if not api_key:
        logger.warning("未找到 COINGECKO_API_KEY，将跳过市值补充")
        df_final = df_binance
        df_final["coingecko_market_cap"] = 0
    else:
        try:
            df_cg = get_top_coingecko_coins(api_key, num=2000)
            # 合并数据：根据 base_asset 匹配 CoinGecko 的 symbol
            df_final = (
                df_binance.merge(
                    df_cg[["symbol", "market_cap"]],
                    left_on="base_asset",
                    right_on="symbol",
                    how="left",
                    suffixes=("", "_cg"))
                .rename(columns={"market_cap": "coingecko_market_cap"})
                .fillna({"coingecko_market_cap": 0})
                .drop(columns=["symbol_cg"])
            )
            logger.info("成功补充 CoinGecko 市值信息（覆盖前 2000 名）")
        except Exception as e:
            logger.error(f"获取 CoinGecko 数据失败: {e}")
            df_final = df_binance
            df_final["coingecko_market_cap"] = 0

    # 3. 保存清洗数据
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    output_file = CLEANED_DIR / "binance_tickers_perp.parquet"
    df_final.to_parquet(output_file, index=False)
    logger.info(f"清洗数据已保存到 {output_file}")

    logger.info("数据管道运行完成")


if __name__ == "__main__":
    task()
