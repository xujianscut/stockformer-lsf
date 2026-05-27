"""Download Nikkei 225 OHLCV via yfinance and convert to StockFormer CSV format."""
import argparse
import os

import numpy as np
import pandas as pd
import yfinance as yf


# Nikkei 225 constituents (.T suffix for Tokyo Stock Exchange).
# This is a snapshot of well-known Nikkei 225 members; we filter by trading-day
# coverage to drop late-listed names.
NIKKEI225_TICKERS = [
    "7203.T", "6758.T", "8306.T", "8035.T", "9984.T", "6098.T", "6861.T",
    "8316.T", "9433.T", "9432.T", "6501.T", "4063.T", "6594.T", "8001.T",
    "8058.T", "7741.T", "4661.T", "6981.T", "4519.T", "9434.T", "8031.T",
    "6273.T", "4503.T", "8411.T", "7267.T", "7011.T", "6367.T", "7974.T",
    "9983.T", "4502.T", "4543.T", "9020.T", "8053.T", "8002.T", "4901.T",
    "6326.T", "3382.T", "4523.T", "4452.T", "4568.T", "4307.T", "6201.T",
    "6503.T", "7751.T", "6857.T", "7733.T", "6920.T", "4578.T", "6902.T",
    "6502.T", "4684.T", "6504.T", "6724.T", "6645.T", "6952.T", "6471.T",
    "6701.T", "8801.T", "8802.T", "8830.T", "9022.T", "9101.T", "9202.T",
    "9301.T", "9501.T", "9502.T", "9503.T", "9531.T", "9532.T", "5108.T",
    "5201.T", "5202.T", "5301.T", "5333.T", "5401.T", "5406.T", "5411.T",
    "5713.T", "5714.T", "5802.T", "5803.T", "5901.T", "1605.T", "1721.T",
    "1801.T", "1802.T", "1803.T", "1808.T", "1812.T", "1925.T", "1928.T",
    "1963.T", "2002.T", "2269.T", "2282.T", "2413.T", "2432.T", "2501.T",
    "2502.T", "2531.T", "2768.T", "2801.T", "2802.T", "2871.T", "2914.T",
    "3086.T", "3099.T", "3101.T", "3105.T", "3110.T", "3401.T", "3402.T",
    "3405.T", "3407.T", "3436.T", "3861.T", "3863.T", "4004.T", "4005.T",
    "4021.T", "4042.T", "4043.T", "4151.T", "4183.T", "4188.T", "4208.T",
    "4324.T", "4506.T", "4507.T", "4536.T", "4151.T", "4631.T", "4751.T",
    "4755.T", "4911.T", "5019.T", "5020.T", "5021.T", "5101.T",
]


def fetch_one(ticker, start, end):
    try:
        df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True, interval="1d")
    except Exception as e:
        return None
    if df is None or len(df) == 0:
        return None
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df[["date", "Open", "Close", "High", "Low", "Volume"]].rename(
        columns={"Open": "open", "Close": "close", "High": "high", "Low": "low", "Volume": "volume"}
    )


def normalize_and_diff(df):
    out = df.copy()
    out["price"] = out["close"].astype(float)
    for col in ("open", "close", "high", "low", "volume"):
        m = float(out[col].max())
        out[col] = out[col] / m if m > 0 else 0.0
    for col in ("open", "close", "high", "low", "volume"):
        d = out[col].diff()
        d.iloc[0] = out[col].iloc[0]
        out[f"d{col}"] = d
    return out[
        ["date", "open", "close", "high", "low", "volume",
         "dopen", "dclose", "dhigh", "dlow", "dvolume", "price"]
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--start", default="2010-01-04")
    p.add_argument("--end", default="2022-05-07")
    p.add_argument("--filter_period", default="2011-01-17:2018-12-28")
    p.add_argument("--min_coverage", type=float, default=0.98)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    fs, fe = args.filter_period.split(":")
    fs = pd.to_datetime(fs); fe = pd.to_datetime(fe)

    ref = fetch_one("7203.T", args.start, args.end)  # Toyota as reference
    ref_days = len(ref[(ref["date"] >= fs) & (ref["date"] <= fe)])
    print(f"reference trading days in filter period: {ref_days}")

    kept = []
    seen = set()
    for i, t in enumerate(NIKKEI225_TICKERS):
        if t in seen:
            continue
        seen.add(t)
        print(f"[{i+1}/{len(NIKKEI225_TICKERS)}] {t} ...", end=" ")
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
        print(f"kept ({len(out)} rows)")

    print(f"\nKept {len(kept)} tickers")
    with open(os.path.join(args.out_dir, "_tickers.txt"), "w") as fh:
        fh.write("\n".join(kept))


if __name__ == "__main__":
    main()
