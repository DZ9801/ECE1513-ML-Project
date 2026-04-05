"""
Preprocessing: feature engineering and train/val/test splitting for
time-series exchange-rate data.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config


def build_features(
    df: pd.DataFrame,
    rate_col: str,
    lag_days: list[int] = config.LAG_DAYS,
    rolling_windows: list[int] = config.ROLLING_WINDOWS,
    horizon: int = config.FORECAST_HORIZON,
) -> pd.DataFrame:
    """Create lag features, rolling statistics, and the target column.

    Parameters
    ----------
    df : DataFrame with columns ['date', rate_col].
    rate_col : Name of the exchange-rate column.
    lag_days : List of lag offsets.
    rolling_windows : List of rolling-window sizes.
    horizon : Number of days ahead to predict.

    Returns
    -------
    DataFrame with engineered features and a 'target' column.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)

    # Lag features
    for lag in lag_days:
        df[f"lag_{lag}"] = df[rate_col].shift(lag)

    # Rolling mean / std
    for w in rolling_windows:
        df[f"roll_mean_{w}"] = df[rate_col].shift(1).rolling(window=w).mean()
        df[f"roll_std_{w}"] = df[rate_col].shift(1).rolling(window=w).std()

    # Percentage change features
    df["pct_change_1"] = df[rate_col].pct_change(1)
    df["pct_change_5"] = df[rate_col].pct_change(5)

    # Day-of-week (cyclical encoding)
    day_of_week = df["date"].dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * day_of_week / 5)
    df["dow_cos"] = np.cos(2 * np.pi * day_of_week / 5)

    # Month (cyclical encoding)
    month = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    # Target: future rate
    df["target"] = df[rate_col].shift(-horizon)

    # Drop rows with NaN produced by shifting / rolling
    df = df.dropna().reset_index(drop=True)

    return df


def split_time_series(
    df: pd.DataFrame,
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio: float = config.VAL_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train / validation / test split (no shuffling)."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    return train, val, test


def get_feature_target(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix X and target vector y from a DataFrame."""
    exclude = {"date", "target"}
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].values.astype(np.float64)
    y = df["target"].values.astype(np.float64)
    return X, y


def scale_data(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Fit a StandardScaler on training data and transform all splits."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_val_s, X_test_s, scaler


def scale_target(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Fit a StandardScaler on training targets and transform all splits."""
    scaler = StandardScaler()
    y_train_s = scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_s = scaler.transform(y_val.reshape(-1, 1)).ravel()
    y_test_s = scaler.transform(y_test.reshape(-1, 1)).ravel()
    return y_train_s, y_val_s, y_test_s, scaler


def inverse_transform_target(
    y: np.ndarray, scaler: StandardScaler
) -> np.ndarray:
    """Inverse-transform scaled targets back to original scale."""
    return scaler.inverse_transform(y.reshape(-1, 1)).ravel()
