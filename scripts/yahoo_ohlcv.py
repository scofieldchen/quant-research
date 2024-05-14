import os
import datetime as dt

import yfinance as yf


ASSETS = [
    {"category": "Stock index", "asset": "SP500", "yahoo_id": "^GSPC"},
    {"category": "Stock index", "asset": "NASDAQ", "yahoo_id": "^IXIC"},
    {"category": "Stock index", "asset": "FTSE100", "yahoo_id": "^FTSE"},
    {"category": "Stock index", "asset": "DAX", "yahoo_id": "^GDAXI"},
    {"category": "Stock index", "asset": "CAC", "yahoo_id": "^FCHI"},
    {"category": "Stock index", "asset": "ESTX50", "yahoo_id": "^STOXX50E"},
    {"category": "Stock index", "asset": "Nikkei225", "yahoo_id": "^N225"},
    {
        "category": "Stock index",
        "asset": "SSE Composite",
        "yahoo_id": "000001.SS",
    },
    {"category": "Future", "asset": "Gold", "yahoo_id": "GC=F"},
    {"category": "Future", "asset": "Silver", "yahoo_id": "SI=F"},
    {"category": "Future", "asset": "Platinum", "yahoo_id": "PL=F"},
    {"category": "Future", "asset": "Copper", "yahoo_id": "HG=F"},
    {"category": "Future", "asset": "Crude oil", "yahoo_id": "CL=F"},
    {"category": "Future", "asset": "Natural gas", "yahoo_id": "NG=F"},
    {"category": "Future", "asset": "RBOB gasoline", "yahoo_id": "RB=F"},
    {"category": "Future", "asset": "US 10-Year Bond", "yahoo_id": "ZF=F"},
    {"category": "Future", "asset": "US 2-Year Bond", "yahoo_id": "ZT=F"},
    {"category": "Currency", "asset": "EURUSD", "yahoo_id": "EURUSD=X"},
    {"category": "Currency", "asset": "USDJPY", "yahoo_id": "JPY=X"},
    {"category": "Currency", "asset": "GBPUSD", "yahoo_id": "GBPUSD=X"},
    {"category": "Currency", "asset": "ICE US Dollar Index", "yahoo_id": "^NYICDX"},
    {"category": "Crypto", "asset": "Bitcoin", "yahoo_id": "BTC-USD"},
    {"category": "Crypto", "asset": "Ethereum", "yahoo_id": "ETH-USD"},
]
START_DATE = dt.datetime(2000, 1, 1)
END_DATE = dt.datetime.today()
OUTPUT_DIR = "../data/yahoo"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for asset in ASSETS:
        asset_name = asset["asset"]
        asset_id = asset["yahoo_id"]
        output_file = os.path.join(OUTPUT_DIR, f"{asset_name}.csv")
        try:
            data = yf.download(asset_id, start=START_DATE, end=END_DATE)
            data.to_csv(output_file)
            print(
                f"{asset_name}: download data from {data.index[0]:%Y-%m-%d} to {data.index[-1]:%Y-%m-%d}"
            )
        except Exception as e:
            print(f"Error downloading {asset_name}: {e}")


if __name__ == "__main__":
    main()
