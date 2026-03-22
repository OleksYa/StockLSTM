"""
Fetches historical OHLCV data for a list of stocks using yfinance
and stores it in a local database.
"""

import sqlite3
import yfinance as yf
import pandas as pd

import os
DB_PATH = os.path.join(os.path.dirname(__file__), "stock_data.db")
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
START_DATE = "2018-01-01"
END_DATE = "2024-01-01"


def create_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT    NOT NULL,
            date        TEXT    NOT NULL,
            open        REAL    NOT NULL,
            high        REAL    NOT NULL,
            low         REAL    NOT NULL,
            close       REAL    NOT NULL,
            volume      REAL    NOT NULL,
            UNIQUE(ticker, date)
        )
    """)
    conn.commit()


def fetch_and_store(ticker: str, conn: sqlite3.Connection,
                    start_date: str = START_DATE, end_date: str = END_DATE) -> int:
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)

    if df.empty:
        print(f"  No data returned for {ticker}")
        return 0

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    df["ticker"] = ticker
    df["date"] = df["date"].astype(str)

    rows = df[["ticker", "date", "open", "high", "low", "close", "volume"]].values.tolist()

    conn.executemany("""
        INSERT OR IGNORE INTO ohlcv (ticker, date, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, rows)
    conn.commit()
    return len(rows)


def main() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        create_table(conn)
        for ticker in TICKERS:
            n = fetch_and_store(ticker, conn)
            print(f"  {ticker}: {n} rows stored")
    print("Done.")


if __name__ == "__main__":
    main()