from datetime import datetime
from pathlib import Path

import pandas as pd

from src.core.bgeometrics import BGClient
from src.core.logger import get_logger

RAW_DATA_DIR = Path("data/raw/sth_mvrv")

logger = get_logger()


def download_sth_mvrv_data():
    """
    下载 STH-MVRV 指标的历史数据，并保存为带时间戳的 CSV 文件到 RAW_DATA_DIR 目录。
    """
    # 初始化客户端
    client = BGClient()

    # 获取 STH-MVRV 数据
    df = client.get_sth_mvrv()

    # 确保目录存在
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d")
    file_path = RAW_DATA_DIR / f"sth_mvrv_{timestamp}.csv"

    # 保存为 CSV（包含 datetime 索引）
    df.to_csv(file_path, index=True)
    logger.info(f"STH-MVRV 数据已下载并保存到 {file_path}")


if __name__ == "__main__":
    download_sth_mvrv_data()
