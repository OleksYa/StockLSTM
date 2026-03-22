## What this does

Fetches historical OHLCV data (open, high, low, close, volume) for a stock
and trains a model to predict how much that stock will move the next day — defined as `(high - low) / close`.
You can tweak the ticker, date range, and training parameters directly from the interface.

## Stack

- **yfinance** — pulls historical stock data
- **SQLite** — stores it locally
- **pandas / NumPy / scikit-learn** — preprocessing pipeline
- **PyTorch** — LSTM model and training loop
- **tkinter + matplotlib** — desktop GUI with embedded plot
