"""
Training routines for sklearn models and the PyTorch MLP/LSTM.
"""

import copy
from itertools import product

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from torch.utils.data import DataLoader

import config
from src.models import MLP, LSTMModel, ExchangeRateDataset, SequenceDataset


# ──────────────────────────────────────────────
# Scikit-learn (Linear Regression / SVR)
# ──────────────────────────────────────────────

def train_sklearn_model(model, X_train: np.ndarray, y_train: np.ndarray):
    """Fit a scikit-learn regression model and return it."""
    model.fit(X_train, y_train)
    return model


# ──────────────────────────────────────────────
# PyTorch MLP
# ──────────────────────────────────────────────

def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict | None = None,
) -> tuple[MLP, dict]:
    """Train the MLP with early stopping on validation loss.

    Returns the best model and a dict with training history.
    """
    if params is None:
        params = config.MLP_PARAMS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets & loaders
    train_ds = ExchangeRateDataset(X_train, y_train)
    val_ds = ExchangeRateDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=params["batch_size"], shuffle=False)

    # Model, loss, optimizer
    model = MLP(
        input_dim=X_train.shape[1],
        hidden_sizes=params["hidden_sizes"],
        dropout=params["dropout"],
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, params["epochs"] + 1):
        # ---- Train ----
        model.train()
        epoch_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * len(X_batch)
        epoch_train_loss /= len(train_ds)

        # ---- Validate ----
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                epoch_val_loss += loss.item() * len(X_batch)
        epoch_val_loss /= len(val_ds)

        scheduler.step(epoch_val_loss)
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:4d} | "
                f"train_loss={epoch_train_loss:.6f} | "
                f"val_loss={epoch_val_loss:.6f}"
            )

        if patience_counter >= params["patience"]:
            print(f"  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model, history


def predict_mlp(model: MLP, X: np.ndarray) -> np.ndarray:
    """Generate predictions from a trained MLP."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        preds = model(X_t).cpu().numpy()
    return preds


# ──────────────────────────────────────────────
# SVR hyperparameter tuning
# ──────────────────────────────────────────────

def tune_svr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    grid: dict | None = None,
) -> tuple[SVR, dict]:
    """Grid search over SVR hyperparameters using validation set.

    Returns the best model (refit on train) and the best params dict.
    """
    if grid is None:
        grid = config.SVR_TUNING_GRID

    param_names = list(grid.keys())
    param_values = list(grid.values())

    best_rmse = float("inf")
    best_params: dict = {}
    best_model: SVR | None = None

    combos = list(product(*param_values))
    print(f"  Tuning SVR: {len(combos)} combinations")

    for combo in combos:
        params = dict(zip(param_names, combo))
        model = SVR(kernel="rbf", **params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params
            best_model = model

    print(f"  Best SVR params: {best_params} (val RMSE={best_rmse:.6f})")
    return best_model, best_params


# ──────────────────────────────────────────────
# PyTorch LSTM
# ──────────────────────────────────────────────

def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict | None = None,
) -> tuple[LSTMModel, dict]:
    """Train the LSTM with early stopping on validation loss.

    Returns the best model and a dict with training history.
    """
    if params is None:
        params = config.LSTM_PARAMS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len = params["seq_len"]

    # Datasets & loaders
    train_ds = SequenceDataset(X_train, y_train, seq_len)
    val_ds = SequenceDataset(X_val, y_val, seq_len)
    train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=params["batch_size"], shuffle=False)

    # Model, loss, optimizer
    model = LSTMModel(
        input_dim=X_train.shape[1],
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, params["epochs"] + 1):
        # ---- Train ----
        model.train()
        epoch_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * len(X_batch)
        epoch_train_loss /= len(train_ds)

        # ---- Validate ----
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                epoch_val_loss += loss.item() * len(X_batch)
        epoch_val_loss /= len(val_ds)

        scheduler.step(epoch_val_loss)
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:4d} | "
                f"train_loss={epoch_train_loss:.6f} | "
                f"val_loss={epoch_val_loss:.6f}"
            )

        if patience_counter >= params["patience"]:
            print(f"  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model, history


def predict_lstm(
    model: LSTMModel, X: np.ndarray, seq_len: int | None = None
) -> np.ndarray:
    """Generate predictions from a trained LSTM.

    Returns predictions aligned with the last (len(X) - seq_len + 1) targets.
    """
    if seq_len is None:
        seq_len = config.LSTM_PARAMS["seq_len"]

    device = next(model.parameters()).device
    model.eval()

    ds = SequenceDataset(X, np.zeros(len(X)), seq_len)
    loader = DataLoader(ds, batch_size=256, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch)
            all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_preds)
