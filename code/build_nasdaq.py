"""Download NASDAQ-100 OHLCV via yfinance and convert to StockFormer CSV format.

Schema matches data/CSI/<ticker>.csv :
  ,date,open,close,high,low,volume,dopen,dclose,dhigh,dlow,dvolume,price

- open/close/high/low/volume are per-stock per-column min-max normalized
  via division by the column max over the full date range.
- dXX columns are first-order differences of the normalized XX columns.
- price column is the raw (unnormalized) close.

Filter: keep only stocks with >= 98% trading days in the train period
(2011-01-17 to 2018-12-28), matching the StockFormer protocol.

Usage:
    python build_nasdaq.py --out_dir /path/to/data/NASDAQ
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import yfinance as yf


# Current NASDAQ-100 constituents (as of late 2024 / early 2025).
# StockFormer paper says they used ~86 stocks after the 98% trading-days filter
# during 2011-01-17 to 2021-12-30. Using the current 100 list and filtering by
# data availability lets us recover a comparable pool.
NASDAQ100_TICKERS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "META", "GOOGL", "GOOG", "TSLA", "AVGO", "COST",
    "PEP", "ADBE", "CSCO", "NFLX", "AMD", "TMUS", "CMCSA", "INTC", "TXN", "QCOM",
    "AMGN", "HON", "INTU", "AMAT", "ISRG", "BKNG", "SBUX", "MDLZ", "ADP", "VRTX",
    "GILD", "LRCX", "ADI", "REGN", "PYPL", "MU", "PANW", "MELI", "ASML", "KLAC",
    "SNPS", "CDNS", "CHTR", "MAR", "MRVL", "ABNB", "ORLY", "MNST", "FTNT", "WDAY",
    "ADSK", "CSX", "PCAR", "KDP", "ROP", "NXPI", "DXCM", "AEP", "PAYX", "FAST",
    "LULU", "CTAS", "AZN", "MCHP", "ODFL", "EXC", "BIIB", "IDXX", "VRSK", "GEHC",
    "KHC", "CEG", "DLTR", "XEL", "CTSH", "ANSS", "WBD", "BKR", "GFS", "ON",
    "CRWD", "FANG", "EA", "TEAM", "TTD", "DDOG", "SIRI", "ZS", "VRSN", "ALGN",
    "WBA", "LCID", "MRNA", "ENPH", "ILMN", "ATVI", "JD", "SGEN", "PDD", "DOCU",
]


def fetch_one(ticker: str, start: str, end: str) -> pd.DataFrame:
    try:
        df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True, interval="1d")
    except Exception as e:
        print(f"  ! {ticker}: fetch error {e}")
        return None
    if df is None or len(df) == 0:
        return None
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df[["date", "Open", "Close", "High", "Low", "Volume"]].rename(
        columns={"Open": "open", "Close": "close", "High": "high", "Low": "low", "Volume": "volume"}
    )
    return df


def normalize_and_diff(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["price"] = out["close"].astype(float)
    for col in ("open", "close", "high", "low", "volume"):
        m = float(out[col].max())
        if m > 0:
            out[col] = out[col] / m
        else:
            out[col] = 0.0
    for col in ("open", "close", "high", "low", "volume"):
        d = out[col].diff()
        d.iloc[0] = out[col].iloc[0]  # match the StockFormer-CSI convention: day-0 dXX = base value
        out[f"d{col}"] = d
    out = out[
        ["date", "open", "close", "high", "low", "volume",
         "dopen", "dclose", "dhigh", "dlow", "dvolume", "price"]
    ]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--start", default="2010-01-04")
    p.add_argument("--end", default="2022-05-07")
    p.add_argument("--filter_period", default="2011-01-17:2018-12-28",
                   help="train period for 98% trading-day filter (start:end)")
    p.add_argument("--min_coverage", type=float, default=0.98)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    fs, fe = args.filter_period.split(":")
    fs = pd.to_datetime(fs); fe = pd.to_datetime(fe)

    # use AAPL to compute the trading-day calendar reference
    ref = fetch_one("AAPL", args.start, args.end)
    ref_days = len(ref[(ref["date"] >= fs) & (ref["date"] <= fe)])
    print(f"reference trading days in filter period: {ref_days}")

    kept = []
    for i, t in enumerate(NASDAQ100_TICKERS):
        print(f"[{i+1}/{len(NASDAQ100_TICKERS)}] {t} ...", end=" ")
        df = fetch_one(t, args.start, args.end)
        if df is None or len(df) == 0:
            print("no data")
            continue
        n_in = len(df[(df["date"] >= fs) & (df["date"] <= fe)])
        cov = n_in / ref_days
        if cov < args.min_coverage:
            print(f"coverage {cov:.2%} < {args.min_coverage:.0%}, drop")
            continue
        out = normalize_and_diff(df)
        out.to_csv(os.path.join(args.out_dir, f"{t}.csv"))
        kept.append(t)
        print(f"kept ({len(out)} rows, coverage {cov:.2%})")

    print(f"\nKept {len(kept)}/{len(NASDAQ100_TICKERS)} tickers")
    with open(os.path.join(args.out_dir, "_tickers.txt"), "w") as fh:
        fh.write("\n".join(kept))


if __name__ == "__main__":
    main()
