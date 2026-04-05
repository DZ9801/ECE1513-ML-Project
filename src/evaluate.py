"""
Evaluation metrics and visualisation utilities.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return MAE, RMSE, and R² for a set of predictions."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def print_metrics(name: str, metrics: dict[str, float]) -> None:
    print(f"  {name:20s} | MAE={metrics['MAE']:.6f} | RMSE={metrics['RMSE']:.6f} | R²={metrics['R2']:.4f}")


# ──────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────

def plot_predictions(
    dates: pd.Series,
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    currency: str,
    save_path: str | None = None,
) -> None:
    """Plot actual vs predicted exchange rates for each model."""
    plt.figure(figsize=(14, 5))
    plt.plot(dates.values, y_true, label="Actual", linewidth=1.5, color="black")
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, (model_name, preds) in enumerate(predictions.items()):
        plt.plot(
            dates.values,
            preds,
            label=model_name,
            linewidth=1.0,
            alpha=0.8,
            color=colors[i % len(colors)],
        )
    plt.xlabel("Date")
    plt.ylabel("Exchange Rate")
    plt.title(f"{currency}/CAD – Actual vs Predicted (Test Set)")
    plt.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"  Figure saved to {save_path}")
    plt.close()


def plot_learning_curve(
    history: dict[str, list[float]],
    currency: str,
    save_path: str | None = None,
    model_name: str = "MLP",
) -> None:
    """Plot training and validation loss curves."""
    plt.figure(figsize=(8, 4))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"{model_name} Learning Curve – {currency}/CAD")
    plt.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"  Figure saved to {save_path}")
    plt.close()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    currency: str,
    save_path: str | None = None,
) -> None:
    """Plot residual distribution for a model."""
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 4))
    plt.hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    plt.axvline(0, color="red", linestyle="--")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title(f"Residuals – {model_name} ({currency}/CAD)")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()


def results_table(all_metrics: dict[str, dict[str, dict[str, float]]]) -> pd.DataFrame:
    """Build a summary DataFrame from nested metrics dict.

    Expected structure: {currency: {model_name: {metric: value}}}
    """
    rows = []
    for currency, models in all_metrics.items():
        for model_name, metrics in models.items():
            rows.append({"Currency": currency, "Model": model_name, **metrics})
    return pd.DataFrame(rows)
