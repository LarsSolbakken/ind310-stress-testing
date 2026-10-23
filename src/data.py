"""Download prices, convert to daily log returns, and split into train/test.

Assets (local currency, no FX conversion):
  - OSEBX  : Oslo Bors Benchmark Index, read from data/osebx.csv (not on Yahoo before 2013)
  - SP500  : ^GSPC via yfinance
  - BONDS  : AGG (iShares US Aggregate Bond ETF) via yfinance
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OSEBX_CSV = DATA_DIR / "osebx.csv"
YF_CACHE = DATA_DIR / "yf_prices.csv"

YF_TICKERS = {"SP500": "^GSPC", "BONDS": "AGG"}
START, END = "2004-01-01", "2025-01-01"  # END is exclusive in yfinance
SPLIT_DATE = "2018-12-31"  # train: up to and including this date, test: after

PRICE_COLUMNS = ["Close", "Adj Close", "Price", "Last", "Lukk"]


def load_osebx(path=OSEBX_CSV):
    """Read OSEBX closing prices from a CSV with a date column and a close/price column."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the OSEBX history (2004-2024) as CSV from "
            "Euronext Oslo Bors or Investing.com and save it there."
        )
    df = pd.read_csv(path, sep=None, engine="python")  # sep=None sniffs ; or ,
    date_col = next((c for c in df.columns if c.strip().lower() in ("date", "dato")), df.columns[0])
    price_col = next((c for c in df.columns if c.strip() in PRICE_COLUMNS), None)
    if price_col is None:
        raise ValueError(f"No price column found in {path}; columns are {list(df.columns)}")

    prices = df[price_col]
    if prices.dtype == object:  # e.g. "1,234.56" or "1 234,56"
        prices = prices.str.replace(" ", "").str.replace(",", "", regex=False)
    prices = pd.to_numeric(prices)
    prices.index = pd.to_datetime(df[date_col])
    return prices.sort_index().rename("OSEBX")[START:END]


def download_prices(refresh=False):
    """Adjusted close prices for the yfinance assets, cached in data/yf_prices.csv."""
    if YF_CACHE.exists() and not refresh:
        return pd.read_csv(YF_CACHE, index_col=0, parse_dates=True)
    raw = yf.download(list(YF_TICKERS.values()), start=START, end=END, auto_adjust=True, progress=False)
    prices = raw["Close"].rename(columns={v: k for k, v in YF_TICKERS.items()})
    DATA_DIR.mkdir(exist_ok=True)
    prices.to_csv(YF_CACHE)
    return prices


def compute_returns(prices):
    """Daily log returns on days where all assets traded (different holidays are dropped)."""
    return np.log(prices).diff().dropna()


def split_train_test(returns, split_date=SPLIT_DATE):
    """Train = up to split_date, test = after. Fit models on train only."""
    return returns.loc[:split_date], returns.loc[split_date:].iloc[1:]


def load_data(osebx_path=OSEBX_CSV, refresh=False):
    """Download/read everything and return (train, test) DataFrames of log returns."""
    prices = download_prices(refresh).join(load_osebx(osebx_path), how="inner")
    prices = prices[["OSEBX", "SP500", "BONDS"]]
    return split_train_test(compute_returns(prices))
