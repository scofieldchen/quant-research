"""测试 BinanceKlinePipeline 的模拟下载。"""

import datetime as dt
import io
import zipfile
from unittest.mock import patch, MagicMock

import pandas as pd

from src.pipelines.binance_klines.task import BinanceKlinePipeline


def create_mock_zip_content():
    """创建模拟的 ZIP 文件内容。"""
    # 模拟 CSV 数据
    csv_data = """open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore
1577836800000,7169.76,7230.00,7150.00,7193.85,2892.123456,1577923199999,20826201.234567,12345,1456.789012,10413100.567890,0
1577923200000,7193.85,7250.00,7175.00,7200.00,2456.987654,1578009599999,17712345.678901,9876,1234.567890,8901234.567890,0
"""

    # 创建 ZIP 文件的字节内容
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        zip_file.writestr("BTCUSDT-1m-2020-01-01.zip", csv_data)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


@patch("src.pipelines.binance_klines.downloader.requests.get")
def test_pipeline_backfill_mock(mock_get):
    """测试 Pipeline 回填功能，使用 mock 模拟下载。"""
    # 设置 mock 返回
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = create_mock_zip_content()
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    # 创建 Pipeline 实例
    pipeline = BinanceKlinePipeline(max_workers=1)

    # 模拟回填（小样本：BTCUSDT 2020-01）
    try:
        pipeline.run_backfill(
            symbols="BTCUSDT", start_date="20200101", end_date="20200102"
        )
        print("✅ 模拟回填测试成功")
        print(f"成功计数: {pipeline.success_count}")
        print(f"失败任务: {len(pipeline.failed_tasks)}")
        print(f"缺失数据: {len(pipeline.missing_data)}")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        raise


if __name__ == "__main__":
    test_pipeline_backfill_mock()
