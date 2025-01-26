import os
import datetime as dt

import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from fredapi import Fred

load_dotenv()


def fetch_operating_cash_balance(
    start_date: dt.datetime = None, end_date: dt.datetime = None
) -> pd.DataFrame:
    """Fetch operating cash balance table of TGA from the Treasury API.

    Args:
        start_date (datetime): start date of the data to fetch
        end_date (datetime): end date of the data to fetch

    Returns:
        DataFrame: operating cash balance table
    """
    if end_date is None:
        end_date = dt.datetime.today()
    if start_date is None:
        start_date = end_date - dt.timedelta(days=365)

    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    page_number = 1
    page_size = 1000  # max size is 10000

    url = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/dts/operating_cash_balance"
    params = {
        "fields": "record_date,account_type,open_today_bal,close_today_bal",
        "filter": f"record_date:gte:{start_date_str},record_date:lte:{end_date_str}",
        "sort": "record_date",
        "format": "json",
        "page[number]": page_number,
        "page[size]": page_size,
    }

    all_data = []

    while True:
        try:
            params.update({"page[number]": page_number})
            resp = requests.get(url, params=params)
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(e)
            break
        else:
            resp_json = resp.json()
            data = resp_json["data"]  # list of rows
            count = resp_json["meta"]["count"]  # number of rows returned
            if not data:  # no rows returned
                break
            all_data.extend(data)
            if count < page_size:  # last page
                break
            page_number += 1

    return pd.DataFrame.from_records(all_data)


def extract_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Extract balance data from the operating cash balance table.

    Args:
        df (DataFrame): operating cash balance table

    Returns:
        DataFrame: balance data
    """
    # account_type == 'Federal Reserve Account'
    balance_1 = (
        df.query("account_type == 'Federal Reserve Account'")
        .loc[:, ["record_date", "close_today_bal"]]
        .rename(columns={"record_date": "date", "close_today_bal": "balance"})
    )

    # account_type == 'Treasury General Account'
    balance_2 = (
        df.query("account_type == 'Treasury General Account (TGA)'")
        .loc[:, ["record_date", "close_today_bal"]]
        .rename(columns={"record_date": "date", "close_today_bal": "balance"})
    )

    # account_type == 'Treasury General Account (TGA) Closing Balance'
    balance_3 = (
        df.query("account_type == 'Treasury General Account (TGA) Closing Balance'")
        .loc[:, ["record_date", "open_today_bal"]]
        .rename(columns={"record_date": "date", "open_today_bal": "balance"})
    )

    joined_balance = pd.concat(
        [balance_1, balance_2, balance_3], axis=0, ignore_index=True
    )
    joined_balance = joined_balance.astype({"balance": np.float64})
    joined_balance.set_index("date", inplace=True)

    return joined_balance


def main():
    # Set date range
    start = dt.datetime(2005, 1, 1)
    end = dt.datetime.today()

    # Fetch Treasury General Account balance
    balance = fetch_operating_cash_balance(start, end)
    balance = extract_balance(balance)
    balance.to_csv("../data/tga_balance.csv", index=True)
    print("Fetched TGA balance from Treasury API")

    # Fetch liquidity datasets from FRED
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))
    series_ids = [
        ("WALCL", "fed_balance_sheet"),
        ("RRPONTSYD", "reverse_repo"),
        ("H41RESPPALDKNWW", "bank_term_funding_program"),
    ]

    for series_id, name in series_ids:
        data = fred.get_series(series_id, start, end)
        data.name = name
        data.index.name = "date"
        filepath = f"../data/{name}.csv"
        data.to_csv(filepath)
        print(f"Fetched {name} from FRED")


if __name__ == "__main__":
    main()
