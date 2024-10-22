import concurrent.futures
import datetime as dt
import os

import ccxt
from fetch_exchange_ohlcv import get_ohlcv
from rich import print


def download_and_save_ohlcv(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    start_date: dt.datetime,
    end_date: dt.datetime,
    output_dir: str,
):
    try:
        df = get_ohlcv(exchange, symbol, timeframe, start_date, end_date)
        output_file = os.path.join(
            output_dir,
            f"binance_{symbol.replace('/', '').lower()}_{timeframe}_ohlcv.csv",
        )
        df.to_csv(output_file)
        print(
            f"[bold green]Successfully downloaded OHLCV data for {symbol} on {timeframe} to {output_file}[/bold green]"
        )
    except Exception as e:
        print(
            f"[bold red]Error downloading OHLCV data for {symbol} on {timeframe}:[/bold red] {e}"
        )


def main():
    # 参数
    exchange_id = "binance"  # ccxt exchange id
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "FTM/USDT"]
    timeframes = ["4h", "1d"]
    start_date = dt.datetime(2017, 1, 1)
    end_date = dt.datetime.today()
    output_dir = "../data/binance"

    # 创建存储目录
    os.makedirs(output_dir, exist_ok=True)

    # 下载数据
    exchange = getattr(ccxt, exchange_id)()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for symbol in symbols:
            for tf in timeframes:
                futures.append(
                    executor.submit(
                        download_and_save_ohlcv,
                        exchange,
                        symbol,
                        tf,
                        start_date,
                        end_date,
                        output_dir,
                    )
                )
        for future in concurrent.futures.as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
