"""
LSTM-based model for next-day stock volatility prediction.
"""

import torch
import torch.nn as nn


class VolatilityLSTM(nn.Module):
    """
    Single-layer LSTM followed by a fully connected prediction head.
    Input:  (batch, window_size, n_features)
    Output: (batch,)  — scalar volatility prediction per sample
    """

    def __init__(self, input_size: int, hidden_size: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]        # take output at final time step
        last_step = self.dropout(last_step)
        return self.fc(last_step).squeeze(-1)  # shape: (batch,)