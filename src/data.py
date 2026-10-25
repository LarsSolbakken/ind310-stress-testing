"""Download prices, convert to daily simple returns, and split into train/test.

Assets (all downloaded with yfinance, USD):
  - SP500 : ^GSPC
  - BONDS : AGG (iShares US Aggregate Bond ETF)

To add an asset, add one line to YF_TICKERS, e.g. "GOLD": "GLD".
OSEBX (Norway) is not on Yahoo before 2013; the CSV loader is kept below, commented out.
"""
from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
YF_CACHE = DATA_DIR / "yf_prices.csv"

YF_TICKERS = {"SP500": "^GSPC", "BONDS": "AGG"}  # name -> Yahoo ticker
HOLDOUT_START = "2025-01-01"  # 2025 onwards is a final holdout: never downloaded or used
START, END = "2004-01-01", HOLDOUT_START  # END is exclusive in yfinance
SPLIT_DATE = "2018-12-31"  # train: up to and including this date, test: after

# --- OSEBX from CSV (disabled for now) ---------------------------------------------
# To re-enable: uncomment, add "OSEBX" to the column list in load_data(), and save the
# index history as data/osebx.csv (date + close columns).
#
# OSEBX_CSV = DATA_DIR / "osebx.csv"
# PRICE_COLUMNS = ["Close", "Adj Close", "Price", "Last", "Lukk"]
#
# def load_osebx(path=OSEBX_CSV):
#     """Read OSEBX closing prices from a CSV with a date column and a close/price column."""
#     path = Path(path)
#     if not path.exists():
#         raise FileNotFoundError(
#             f"{path} not found. Download the OSEBX history (2004-2024) as CSV from "
#             "Euronext Oslo Bors or Investing.com and save it there."
#         )
#     df = pd.read_csv(path, sep=None, engine="python")  # sep=None sniffs ; or ,
#     date_col = next((c for c in df.columns if c.strip().lower() in ("date", "dato")), df.columns[0])
#     price_col = next((c for c in df.columns if c.strip() in PRICE_COLUMNS), None)
#     if price_col is None:
#         raise ValueError(f"No price column found in {path}; columns are {list(df.columns)}")
#     prices = df[price_col]
#     if prices.dtype == object:  # e.g. "1,234.56" or "1 234,56"
#         prices = prices.str.replace(" ", "").str.replace(",", "", regex=False)
#     prices = pd.to_numeric(prices)
#     prices.index = pd.to_datetime(df[date_col])
#     return prices.sort_index().rename("OSEBX")[START:END]
# ------------------------------------------------------------------------------------


def download_prices(refresh=False):
    """Adjusted close prices for the yfinance assets, cached in data/yf_prices.csv."""
    if YF_CACHE.exists() and not refresh:
        cached = pd.read_csv(YF_CACHE, index_col=0, parse_dates=True)
        if set(YF_TICKERS) <= set(cached.columns):  # cache covers all current assets
            return check_no_holdout(cached[list(YF_TICKERS)])
    raw = yf.download(list(YF_TICKERS.values()), start=START, end=END, auto_adjust=True, progress=False)
    prices = check_no_holdout(raw["Close"].rename(columns={v: k for k, v in YF_TICKERS.items()})[list(YF_TICKERS)])
    DATA_DIR.mkdir(exist_ok=True)
    prices.to_csv(YF_CACHE)
    return prices


def check_no_holdout(prices):
    """Stop if any data from the final holdout period (HOLDOUT_START onwards) is present."""
    if prices.index.max() >= pd.Timestamp(HOLDOUT_START):
        raise ValueError(f"Data from {HOLDOUT_START} or later found. That period is the final holdout "
                         "and must not be downloaded or used.")
    return prices


def compute_returns(prices):
    """Daily simple returns (P_t / P_{t-1} - 1) on days where all assets traded (different holidays are dropped)."""
    return prices.dropna().pct_change().dropna()


def split_train_test(returns, split_date=SPLIT_DATE):
    """Train = up to split_date, test = after. Fit models on train only."""
    return returns.loc[:split_date], returns.loc[split_date:].iloc[1:]


def load_data(refresh=False):
    """Download everything and return (train, test) DataFrames of simple returns."""
    return split_train_test(compute_returns(download_prices(refresh)))
