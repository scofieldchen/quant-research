from pathlib import Path
from datetime import datetime

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.logger import get_logger


# 数据目录
RAW_DIR = Path("data/raw/binance_tickers")
CLEANED_DIR = Path("data/cleaned")

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


def task():
    """
    主任务函数：获取并清洗 Binance 永续合约交易对数据。
    """
    logger.info("开始运行 binance_tickers 数据管道")

    # 获取数据
    data = fetch_exchange_info()

    # 清洗数据
    df = clean_perp_tickers(data)

    # 保存清洗数据
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    output_file = CLEANED_DIR / "binance_tickers_perp.parquet"
    df.to_parquet(output_file, index=False)
    logger.info(f"清洗数据已保存到 {output_file}")

    logger.info("数据管道运行完成")


if __name__ == "__main__":
    task()
