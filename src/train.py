"""
Training loop for the VolatilityLSTM model.
Trains on AAPL data by default, evaluates on a held out validation set,
saves the best model checkpoint, and plots predicted vs actual volatility.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.dataset import load_from_db, preprocess, VolatilityDataset
from src.model import VolatilityLSTM

TICKER      = "AAPL"
WINDOW_SIZE = 10
BATCH_SIZE  = 32
HIDDEN_SIZE = 64
DROPOUT     = 0.2
LR          = 0.001
NUM_EPOCHS  = 50
PATIENCE    = 8
CHECKPOINT  = "best_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss  = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            preds = model(X_batch)
            total_loss += criterion(preds, y_batch).item() * X_batch.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, np.array(all_preds), np.array(all_targets)


def plot_predictions(preds: np.ndarray, targets: np.ndarray) -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(targets, label="Actual volatility",    linewidth=1.2)
    plt.plot(preds,   label="Predicted volatility", linewidth=1.2, alpha=0.8)
    plt.title(f"{TICKER} — Next-Day Volatility: Predicted vs Actual")
    plt.xlabel("Trading day (validation set)")
    plt.ylabel("Volatility (high-low / close)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("predictions.png", dpi=150)
    print("Plot saved to predictions.png")


def main():
    print(f"Device: {DEVICE}")

    # Data
    df = load_from_db(ticker=TICKER)
    X_train, y_train, X_val, y_val, scaler = preprocess(df, window_size=WINDOW_SIZE)

    n_features = X_train.shape[2]
    print(f"Train samples: {len(X_train)}  |  Val samples: {len(X_val)}  |  Features: {n_features}")

    train_loader = DataLoader(VolatilityDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(VolatilityDataset(X_val,   y_val),   batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model     = VolatilityLSTM(input_size=n_features, hidden_size=HIDDEN_SIZE, dropout=DROPOUT).to(DEVICE)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Training loop
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, preds, targets = eval_epoch(model, val_loader, criterion)

        print(f"Epoch [{epoch:02d}/{NUM_EPOCHS}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), CHECKPOINT)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    # Evaluation
    print(f"\nLoading best model from {CHECKPOINT}")
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    _, preds, targets = eval_epoch(model, val_loader, criterion)

    rmse = np.sqrt(mean_squared_error(targets, preds))
    mae  = mean_absolute_error(targets, preds)
    print(f"\nFinal Evaluation on Validation Set:")
    print(f"  RMSE : {rmse:.6f}")
    print(f"  MAE  : {mae:.6f}")

    plot_predictions(preds, targets)


if __name__ == "__main__":
    main()