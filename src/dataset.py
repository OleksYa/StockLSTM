"""
Preprocessing pipeline and PyTorch Dataset for stock volatility prediction.

Target: next-day volatility = (high - low) / close  (normalised range).
"""

import sqlite3
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

import os
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "stock_data.db")
WINDOW_SIZE = 10  # number of past days used to predict next day


def load_from_db(db_path: str = DB_PATH, ticker: str = "AAPL") -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            "SELECT date, open, high, low, close, volume FROM ohlcv WHERE ticker = ? ORDER BY date ASC",
            conn,
            params=(ticker,)
        )
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["volatility"] = (df["high"] - df["low"]) / df["close"]

    df["daily_return"] = df["close"].pct_change()

    df["rolling_vol_5"] = df["volatility"].rolling(5).mean()

    df["volume_change"] = df["volume"].pct_change()

    # Drop rows with NaNs introduced by rolling or pct_change
    df = df.dropna().reset_index(drop=True)

    return df


def preprocess(
    df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    train_ratio: float = 0.8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:

    df = engineer_features(df)

    feature_cols = ["open", "high", "low", "close", "volume",
                    "daily_return", "rolling_vol_5", "volume_change"]
    target_col = "volatility"

    features = df[feature_cols].values
    targets = df[target_col].values

    # Train and val split befor scaling to prevent data leakage
    split_idx = int(len(df) * train_ratio)

    train_features = features[:split_idx]
    val_features   = features[split_idx:]
    train_targets  = targets[:split_idx]
    val_targets    = targets[split_idx:]

    # Fit scaler on training data only
    scaler = MinMaxScaler()
    train_features = scaler.fit_transform(train_features)
    val_features   = scaler.transform(val_features)

    def build_windows(feat: np.ndarray, tgt: np.ndarray, w: int):
        X, y = [], []
        for i in range(len(feat) - w):
            X.append(feat[i: i + w])
            y.append(tgt[i + w])  # next-day target
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    X_train, y_train = build_windows(train_features, train_targets, window_size)
    X_val,   y_val   = build_windows(val_features,   val_targets,   window_size)

    return X_train, y_train, X_val, y_val, scaler


class VolatilityDataset(Dataset):

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
