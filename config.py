"""
Configuration file for CAD Exchange Rate Prediction project.
"""

# ──────────────────────────────────────────────
# Data settings
# ──────────────────────────────────────────────
CURRENCY_PAIRS = {
    "USD": "FXUSDCAD",
    "EUR": "FXEURCAD",
    "CNY": "FXCNYCAD",
}

# Bank of Canada Valet API base URL
BOC_API_BASE = "https://www.bankofcanada.ca/valet/observations"

# Date range for historical data
START_DATE = "2015-01-01"
END_DATE = "2025-12-31"

# ──────────────────────────────────────────────
# Feature engineering
# ──────────────────────────────────────────────
LAG_DAYS = [1, 2, 3, 5, 7, 10, 14, 21]
ROLLING_WINDOWS = [5, 10, 21, 50]
FORECAST_HORIZON = 1  # predict N days ahead

# ──────────────────────────────────────────────
# Train / validation / test split
# ──────────────────────────────────────────────
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ──────────────────────────────────────────────
# Model hyperparameters
# ──────────────────────────────────────────────

# Support Vector Regression
SVR_PARAMS = {
    "kernel": "rbf",
    "C": 10.0,
    "epsilon": 0.01,
    "gamma": "scale",
}

# MLP (PyTorch)
MLP_PARAMS = {
    "hidden_sizes": [128, 64, 32],
    "dropout": 0.2,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "batch_size": 64,
    "epochs": 200,
    "patience": 20,  # early stopping patience
}

# LSTM (PyTorch)
LSTM_PARAMS = {
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "seq_len": 21,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "batch_size": 64,
    "epochs": 200,
    "patience": 20,
}

# SVR hyperparameter tuning grid
SVR_TUNING_GRID = {
    "C": [0.1, 1.0, 10.0, 100.0],
    "epsilon": [0.001, 0.01, 0.1],
    "gamma": ["scale", "auto"],
}

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
DATA_DIR = "data"
RESULTS_DIR = "results"
FIGURES_DIR = "results/figures"
