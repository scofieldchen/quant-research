import json
import datetime as dt

import ccxt
import pandas as pd

import utils


def main():
    # Script inputs
    input_json_file = (
        "../data/coingecko_coins.json"  # json file with token informations
    )
    year = 2024  # year to download ohlcv data
    timeframe = "1d"  # timeframe for ohlcv data
    output_csv_file = "../data/binance_daily_ohlcv_2024.csv"  # output csv file

    # Load tokens data from json file
    with open(input_json_file, "r") as f:
        coins = json.load(f)
    print(f"Number of tokens: {len(coins)}")

    # Generate USDT pairs for tokens, exclude stablecoins
    tickers = [
        f"{coin['symbol'].upper()}/USDT"
        for coin in coins
        if "Stablecoins" not in coin["category"]
        or "USD Stablecoin" not in coin["category"]
    ]
    print(f"Number of Non-stable tokens: {len(tickers)}")

    # Check if the ticker are available on Binance
    exchange = ccxt.binance()
    markets = exchange.load_markets()
    tickers = [ticker for ticker in tickers if ticker in list(markets.keys())]
    print(f"Number of tokens available on Binance: {len(tickers)}")

    # Download spot ohlcv data from Binance
    start_date = dt.datetime(year, 1, 1)
    end_date = dt.datetime(year + 1, 1, 1) - dt.timedelta(days=1)

    df_list = []
    for ticker in tickers:
        try:
            df = utils.get_ohlcv(exchange, ticker, timeframe, start_date, end_date)
            if not df.empty:
                df["symbol"] = ticker
                df_list.append(df)
                print(f"Downloaded {ticker} data")
            else:
                print(f"No data for {ticker}")
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")

    df_joined = pd.concat(df_list)

    # Save ohlcv data to csv file
    df_joined.to_csv(output_csv_file, index=True)
    print(f"Saved data to {output_csv_file}")


if __name__ == "__main__":
    main()
