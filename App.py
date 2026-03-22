"""
GUI for the Stock Volatility Predictor.
"""

import threading
import tkinter as tk
from tkinter import ttk
import queue
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from data.fetch_and_store import fetch_and_store, create_table
from src.dataset import load_from_db, preprocess, VolatilityDataset
from src.model import VolatilityLSTM

import sqlite3, os, torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Paths
DB_PATH = os.path.join(os.path.dirname(__file__), "data", "stock_data.db")

# Data options
TICKERS             = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
DEFAULT_TICKER      = "AAPL"
DEFAULT_START_DATE  = "2018-01-01"
DEFAULT_END_DATE    = "2024-01-01"

# Model defaults
DEFAULT_WINDOW_SIZE = 10
DEFAULT_EPOCHS      = 50
DEFAULT_BATCH_SIZE  = 32
DEFAULT_HIDDEN_SIZE = 64

# Spinbox ranges
WINDOW_MIN  = 5
WINDOW_MAX  = 30
EPOCHS_MIN  = 5
EPOCHS_MAX  = 200
BATCH_MIN   = 8
BATCH_MAX   = 128
BATCH_STEP  = 8
HIDDEN_MIN  = 16
HIDDEN_MAX  = 256
HIDDEN_STEP = 16

# Training
EARLY_STOP_PATIENCE = 8
LEARNING_RATE       = 0.001

# Layout
PAD_X            = 10
PAD_Y            = 6
LEFT_PADDING     = 10
RIGHT_PADDING    = 10
SEPARATOR_PAD_Y  = 10
TITLE_PAD_BOTTOM = 12
ENTRY_WIDTH      = 12
COMBO_WIDTH      = 10
SPIN_WIDTH       = 8
LOG_WIDTH        = 60
LOG_HEIGHT       = 12
PLOT_WIDTH       = 7
PLOT_HEIGHT      = 3
PLOT_DPI         = 90

# Fonts
FONT_FAMILY       = "Arial"
FONT_FAMILY_MONO  = "Courier"
FONT_SIZE_TITLE   = 13
FONT_SIZE_SECTION = 10
FONT_SIZE_LOG     = 9

# Colours
COLOR_METRICS         = "#1a5276"
COLOR_LOG_BG          = "#1e1e1e"
COLOR_LOG_FG          = "#d4d4d4"
COLOR_PLOT_BG         = "#f9f9f9"
COLOR_PLOT_EMPTY      = "#aaaaaa"
COLOR_ACTUAL          = "#2980b9"
COLOR_PREDICTED       = "#e74c3c"
COLOR_PREDICTED_ALPHA = 0.85

# Strings
APP_TITLE            = "Stock Volatility Predictor"
LABEL_TICKER         = "Ticker"
LABEL_START          = "Start date"
LABEL_END            = "End date"
LABEL_WINDOW         = "Window size"
LABEL_EPOCHS         = "Epochs"
LABEL_BATCH          = "Batch size"
LABEL_HIDDEN         = "Hidden size"
LABEL_FETCH          = "Fetch Data"
LABEL_TRAIN          = "Train Model"
LABEL_EXIT           = "Exit"
LABEL_METRICS        = "Metrics"
LABEL_LOG            = "Training Log"
LABEL_PLOT           = "Predictions vs Actual"
LABEL_RMSE_EMPTY     = "RMSE: \u2014"
LABEL_MAE_EMPTY      = "MAE:  \u2014"
LABEL_PLOT_EMPTY     = "Train a model to see predictions"
PLOT_TITLE_TEMPLATE  = "{ticker} \u2014 Next-Day Volatility"
PLOT_X_LABEL         = "Validation day"
PLOT_Y_LABEL         = "Volatility"
LEGEND_ACTUAL        = "Actual"
LEGEND_PREDICTED     = "Predicted"
LOG_FETCHING         = "Fetching {ticker}  {start} -> {end}..."
LOG_STORED           = "Stored {n} rows for {ticker}."
LOG_NO_DATA          = "No data for {ticker}. Run Fetch Data first."
LOG_DEVICE           = "Device: {device}"
LOG_SHAPES           = "Train: {train}  Val: {val}  Features: {features}"
LOG_EPOCH            = "Epoch [{epoch:02d}/{total}] Val Loss: {loss:.6f}"
LOG_EARLY_STOP       = "Early stopping at epoch {epoch}."
LOG_METRICS          = "\nRMSE: {rmse:.6f}  |  MAE: {mae:.6f}"
LOG_ERROR            = "Error: {error}"
RMSE_TEMPLATE        = "RMSE: {value:.6f}"
MAE_TEMPLATE         = "MAE:  {value:.6f}"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.resizable(False, False)
        self.log_queue = queue.Queue()
        self._build_ui()
        self._poll_log()

    # UI construction

    def _build_ui(self):
        PAD = dict(padx=PAD_X, pady=PAD_Y)

        left = ttk.Frame(self, padding=LEFT_PADDING)
        left.grid(row=0, column=0, sticky="ns")

        ttk.Label(left, text=APP_TITLE,
                  font=(FONT_FAMILY, FONT_SIZE_TITLE, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(0, TITLE_PAD_BOTTOM))

        ttk.Label(left, text=LABEL_TICKER).grid(row=1, column=0, sticky="w", **PAD)
        self.ticker_var = tk.StringVar(value=DEFAULT_TICKER)
        ttk.Combobox(left, textvariable=self.ticker_var, values=TICKERS,
                     state="readonly", width=COMBO_WIDTH).grid(row=1, column=1, sticky="w", **PAD)

        ttk.Label(left, text=LABEL_START).grid(row=2, column=0, sticky="w", **PAD)
        self.start_var = tk.StringVar(value=DEFAULT_START_DATE)
        ttk.Entry(left, textvariable=self.start_var, width=ENTRY_WIDTH).grid(
            row=2, column=1, sticky="w", **PAD)

        ttk.Label(left, text=LABEL_END).grid(row=3, column=0, sticky="w", **PAD)
        self.end_var = tk.StringVar(value=DEFAULT_END_DATE)
        ttk.Entry(left, textvariable=self.end_var, width=ENTRY_WIDTH).grid(
            row=3, column=1, sticky="w", **PAD)

        ttk.Label(left, text=LABEL_WINDOW).grid(row=4, column=0, sticky="w", **PAD)
        self.window_var = tk.IntVar(value=DEFAULT_WINDOW_SIZE)
        ttk.Spinbox(left, from_=WINDOW_MIN, to=WINDOW_MAX,
                    textvariable=self.window_var, width=SPIN_WIDTH).grid(
            row=4, column=1, sticky="w", **PAD)

        ttk.Label(left, text=LABEL_EPOCHS).grid(row=5, column=0, sticky="w", **PAD)
        self.epochs_var = tk.IntVar(value=DEFAULT_EPOCHS)
        ttk.Spinbox(left, from_=EPOCHS_MIN, to=EPOCHS_MAX,
                    textvariable=self.epochs_var, width=SPIN_WIDTH).grid(
            row=5, column=1, sticky="w", **PAD)

        ttk.Label(left, text=LABEL_BATCH).grid(row=6, column=0, sticky="w", **PAD)
        self.batch_var = tk.IntVar(value=DEFAULT_BATCH_SIZE)
        ttk.Spinbox(left, from_=BATCH_MIN, to=BATCH_MAX, increment=BATCH_STEP,
                    textvariable=self.batch_var, width=SPIN_WIDTH).grid(
            row=6, column=1, sticky="w", **PAD)

        ttk.Label(left, text=LABEL_HIDDEN).grid(row=7, column=0, sticky="w", **PAD)
        self.hidden_var = tk.IntVar(value=DEFAULT_HIDDEN_SIZE)
        ttk.Spinbox(left, from_=HIDDEN_MIN, to=HIDDEN_MAX, increment=HIDDEN_STEP,
                    textvariable=self.hidden_var, width=SPIN_WIDTH).grid(
            row=7, column=1, sticky="w", **PAD)

        ttk.Separator(left, orient="horizontal").grid(
            row=8, column=0, columnspan=2, sticky="ew", pady=SEPARATOR_PAD_Y)

        self.fetch_btn = ttk.Button(left, text=LABEL_FETCH, command=self._on_fetch)
        self.fetch_btn.grid(row=9, column=0, columnspan=2, sticky="ew", **PAD)

        self.train_btn = ttk.Button(left, text=LABEL_TRAIN, command=self._on_train)
        self.train_btn.grid(row=10, column=0, columnspan=2, sticky="ew", **PAD)

        self.exit_btn = ttk.Button(left, text=LABEL_EXIT, command=self.quit)
        self.exit_btn.grid(row=11, column=0, columnspan=2, sticky="ew", **PAD)

        ttk.Separator(left, orient="horizontal").grid(
            row=12, column=0, columnspan=2, sticky="ew", pady=SEPARATOR_PAD_Y)

        ttk.Label(left, text=LABEL_METRICS,
                  font=(FONT_FAMILY, FONT_SIZE_SECTION, "bold")).grid(
            row=13, column=0, columnspan=2, sticky="w")

        self.rmse_var = tk.StringVar(value=LABEL_RMSE_EMPTY)
        self.mae_var  = tk.StringVar(value=LABEL_MAE_EMPTY)
        ttk.Label(left, textvariable=self.rmse_var, foreground=COLOR_METRICS).grid(
            row=14, column=0, columnspan=2, sticky="w", padx=PAD_X)
        ttk.Label(left, textvariable=self.mae_var, foreground=COLOR_METRICS).grid(
            row=15, column=0, columnspan=2, sticky="w", padx=PAD_X)

        right = ttk.Frame(self, padding=RIGHT_PADDING)
        right.grid(row=0, column=1, sticky="nsew")

        ttk.Label(right, text=LABEL_LOG,
                  font=(FONT_FAMILY, FONT_SIZE_SECTION, "bold")).pack(anchor="w")

        self.log_box = tk.Text(right, width=LOG_WIDTH, height=LOG_HEIGHT,
                               font=(FONT_FAMILY_MONO, FONT_SIZE_LOG),
                               state="disabled", bg=COLOR_LOG_BG, fg=COLOR_LOG_FG,
                               relief="flat", wrap="none")
        self.log_box.pack(fill="x", pady=(PAD_Y, PAD_Y + 4))

        ttk.Label(right, text=LABEL_PLOT,
                  font=(FONT_FAMILY, FONT_SIZE_SECTION, "bold")).pack(anchor="w")

        self.fig, self.ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT), dpi=PLOT_DPI)
        self.fig.patch.set_facecolor(COLOR_PLOT_BG)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._draw_empty_plot()

    # Logging

    def _log(self, msg: str):
        self.log_queue.put(msg)

    def _poll_log(self):
        while not self.log_queue.empty():
            msg = self.log_queue.get_nowait()
            self.log_box.configure(state="normal")
            self.log_box.insert("end", msg + "\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")
        self.after(100, self._poll_log)

    # Plot

    def _draw_empty_plot(self):
        self.ax.clear()
        self.ax.set_facecolor(COLOR_PLOT_BG)
        self.ax.text(0.5, 0.5, LABEL_PLOT_EMPTY,
                     ha="center", va="center", transform=self.ax.transAxes,
                     color=COLOR_PLOT_EMPTY, fontsize=FONT_SIZE_SECTION)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.draw()

    def _draw_plot(self, preds: np.ndarray, targets: np.ndarray, ticker: str):
        self.ax.clear()
        self.ax.set_facecolor(COLOR_PLOT_BG)
        self.ax.plot(targets, label=LEGEND_ACTUAL,    linewidth=1.3, color=COLOR_ACTUAL)
        self.ax.plot(preds,   label=LEGEND_PREDICTED, linewidth=1.3,
                     color=COLOR_PREDICTED, alpha=COLOR_PREDICTED_ALPHA)
        self.ax.set_title(PLOT_TITLE_TEMPLATE.format(ticker=ticker), fontsize=FONT_SIZE_SECTION)
        self.ax.set_xlabel(PLOT_X_LABEL, fontsize=FONT_SIZE_LOG)
        self.ax.set_ylabel(PLOT_Y_LABEL, fontsize=FONT_SIZE_LOG)
        self.ax.legend(fontsize=FONT_SIZE_LOG)
        self.ax.tick_params(labelsize=FONT_SIZE_LOG)
        self.fig.tight_layout()
        self.canvas.draw()

    # Button handlers

    def _set_buttons(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        self.fetch_btn.config(state=state)
        self.train_btn.config(state=state)

    def _on_fetch(self):
        ticker = self.ticker_var.get()
        start  = self.start_var.get().strip()
        end    = self.end_var.get().strip()
        self._set_buttons(False)
        threading.Thread(target=self._fetch_worker, args=(ticker, start, end), daemon=True).start()

    def _fetch_worker(self, ticker: str, start: str, end: str):
        self._log(LOG_FETCHING.format(ticker=ticker, start=start, end=end))
        try:
            with sqlite3.connect(DB_PATH) as conn:
                create_table(conn)
                n = fetch_and_store(ticker, conn, start_date=start, end_date=end)
            self._log(LOG_STORED.format(n=n, ticker=ticker))
        except Exception as e:
            self._log(LOG_ERROR.format(error=e))
        finally:
            self.after(0, lambda: self._set_buttons(True))

    def _on_train(self):
        self._set_buttons(False)
        self._draw_empty_plot()
        self.rmse_var.set(LABEL_RMSE_EMPTY)
        self.mae_var.set(LABEL_MAE_EMPTY)
        params = {
            "ticker":      self.ticker_var.get(),
            "window_size": self.window_var.get(),
            "num_epochs":  self.epochs_var.get(),
            "batch_size":  self.batch_var.get(),
            "hidden_size": self.hidden_var.get(),
        }
        threading.Thread(target=self._train_worker, kwargs=params, daemon=True).start()

    def _train_worker(self, ticker, window_size, num_epochs, batch_size, hidden_size):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._log(LOG_DEVICE.format(device=device))

            df = load_from_db(ticker=ticker)
            if df.empty:
                self._log(LOG_NO_DATA.format(ticker=ticker))
                return

            X_train, y_train, X_val, y_val, _ = preprocess(df, window_size=window_size)
            n_features = X_train.shape[2]
            self._log(LOG_SHAPES.format(train=len(X_train), val=len(X_val), features=n_features))

            train_loader = DataLoader(VolatilityDataset(X_train, y_train),
                                      batch_size=batch_size, shuffle=True)
            val_loader   = DataLoader(VolatilityDataset(X_val, y_val),
                                      batch_size=batch_size, shuffle=False)

            model     = VolatilityLSTM(n_features, hidden_size).to(device)
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

            best_val_loss  = float("inf")
            best_state     = None
            no_improvement = 0

            for epoch in range(1, num_epochs + 1):
                model.train()
                for X_b, y_b in train_loader:
                    X_b, y_b = X_b.to(device), y_b.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(X_b), y_b)
                    loss.backward()
                    optimizer.step()

                model.eval()
                val_loss = 0.0
                preds_list, targets_list = [], []
                with torch.no_grad():
                    for X_b, y_b in val_loader:
                        X_b, y_b = X_b.to(device), y_b.to(device)
                        out = model(X_b)
                        val_loss += criterion(out, y_b).item() * X_b.size(0)
                        preds_list.extend(out.cpu().numpy())
                        targets_list.extend(y_b.cpu().numpy())
                val_loss /= len(val_loader.dataset)

                self._log(LOG_EPOCH.format(epoch=epoch, total=num_epochs, loss=val_loss))

                if val_loss < best_val_loss:
                    best_val_loss  = val_loss
                    best_state     = {k: v.clone() for k, v in model.state_dict().items()}
                    no_improvement = 0
                else:
                    no_improvement += 1
                    if no_improvement >= EARLY_STOP_PATIENCE:
                        self._log(LOG_EARLY_STOP.format(epoch=epoch))
                        break

            model.load_state_dict(best_state)
            model.eval()
            preds_list, targets_list = [], []
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    out = model(X_b.to(device))
                    preds_list.extend(out.cpu().numpy())
                    targets_list.extend(y_b.numpy())

            preds   = np.array(preds_list)
            targets = np.array(targets_list)
            rmse = np.sqrt(mean_squared_error(targets, preds))
            mae  = mean_absolute_error(targets, preds)

            self._log(LOG_METRICS.format(rmse=rmse, mae=mae))
            self.after(0, lambda: self.rmse_var.set(RMSE_TEMPLATE.format(value=rmse)))
            self.after(0, lambda: self.mae_var.set(MAE_TEMPLATE.format(value=mae)))
            self.after(0, lambda: self._draw_plot(preds, targets, ticker))

        except Exception as e:
            self._log(LOG_ERROR.format(error=e))
        finally:
            self.after(0, lambda: self._set_buttons(True))


if __name__ == "__main__":
    app = App()
    app.mainloop()