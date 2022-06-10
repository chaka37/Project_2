import pandas as pd
import numpy as np
from pathlib import Path
import hvplot.pandas
import matplotlib.pyplot as plt
import os
import requests
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import REST, TimeFrame

def trade_api():
    load_dotenv()
    ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
    ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
    print(f"Alpaca Key type: {type(ALPACA_API_KEY)}")
    print(f"Alpaca Secret Key type: {type(ALPACA_SECRET_KEY)}")
    alpaca = tradeapi.REST(
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    api_version="v2")
    return alpaca

alpaca = trade_api()

start = input(print("Please Enter a Date, format ex: 2017-13-01"))
end = input(print("Please Enter a Date, format ex: 2017-13-01"))


def get_company(ticker, start, end, alpaca):
    ticker = alpaca.get_bars(
        ticker,
        TimeFrame.Day,
        start,
        end
        ).df.drop(["trade_count", "vwap"], axis=1)
    return ticker
ticker_df = get_company()

def save_csv():
    ticker_df.to_csv('../Project_2/stock_data.csv', index=True)
    return ticker_df
