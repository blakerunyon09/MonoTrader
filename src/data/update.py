import os
import sys
from datetime import date, timedelta
import pandas as pd
import yfinance as yf

def load_existing(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path, index_col=0)

    if isinstance(df.index[0], str) and df.index[0].lower() in {"price", "date"}:
        df = pd.read_csv(path, skiprows=3, header=None,
                         names=["Date", "Close", "High", "Low", "Open", "Volume"])
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    else:
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()]

    df = df.sort_index()

    if "Close" not in df.columns:
        if "Price" in df.columns:
            df["Close"] = df["Price"]
        else:
            raise ValueError("No 'Close' or 'Price' column found.")
    return df

def latest_trading_day(today: date) -> date:
    wd = today.weekday()
    if wd == 5:  # Saturday
        return today - timedelta(days=1)
    if wd == 6:  # Sunday
        return today - timedelta(days=2)
    return today

def main(symbol: str):
    file_path = f"data/{symbol.upper()}.csv"
    df = load_existing(file_path)

    if df.empty:
        last_date = date(2020, 1, 1) - timedelta(days=1)
    else:
        last_date = df.index.max().date()

    mkt_last = latest_trading_day(date.today())

    if last_date >= mkt_last:
        print(f"{symbol.upper()} data is already up to date; skipping download.")
        return

    start = last_date + timedelta(days=1) if not df.empty else date(2020, 1, 1)
    new_data = yf.download(symbol.upper(), start=start, interval="1d", progress=False, auto_adjust=True)

    if new_data.empty:
        print(f"No new data for {symbol.upper()} (market closed or no updates).")
        return

    keep_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in new_data.columns]
    new_data = new_data[keep_cols]

    if df.empty:
        combined = new_data
    else:
        combined = pd.concat([df, new_data])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    combined.to_csv(file_path, date_format="%Y-%m-%d")
    print(f"Added {len(new_data)} new rows for {symbol.upper()}. Now {len(combined)} total.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python update.py SYMBOL")
        sys.exit(1)
    main(sys.argv[1])
