"""
Main entry point – run the full pipeline:
  1. Download / load exchange-rate data
  2. Feature engineering
  3. Train Linear Regression, SVR, MLP
  4. Evaluate on test set and produce figures & summary table
"""

import os

import config
from src.data_loader import download_all
from src.preprocessing import build_features, split_time_series, get_feature_target, scale_data
from src.models import build_linear_regression, build_svr
from src.train import train_sklearn_model, train_mlp, predict_mlp
from src.evaluate import (
    compute_metrics,
    print_metrics,
    plot_predictions,
    plot_learning_curve,
    plot_residuals,
    results_table,
)
from src.utils import set_seed, ensure_dirs
from sklearn.preprocessing import StandardScaler


def run_pipeline() -> None:
    set_seed(42)
    ensure_dirs(config.DATA_DIR, config.RESULTS_DIR, config.FIGURES_DIR)

    # ── 1. Download data ─────────────────────────
    print("=" * 60)
    print("Step 1: Downloading / loading exchange-rate data")
    print("=" * 60)
    datasets = download_all()

    all_metrics: dict[str, dict[str, dict[str, float]]] = {}

    for currency, df_raw in datasets.items():
        print("\n" + "=" * 60)
        print(f"Processing {currency}/CAD  ({len(df_raw)} raw observations)")
        print("=" * 60)

        rate_col = config.CURRENCY_PAIRS[currency]

        # ── 2. Feature engineering ────────────────
        print("\nStep 2: Feature engineering")
        df_feat = build_features(df_raw, rate_col)
        print(f"  Features: {df_feat.shape[1] - 2} columns, {len(df_feat)} samples")

        # Identify feature columns (exclude date and target)
        feature_cols = [c for c in df_feat.columns if c not in {"date", "target", rate_col}]

        # ── 3. Train / val / test split ───────────
        print("\nStep 3: Splitting data (70/15/15)")
        train_df, val_df, test_df = split_time_series(df_feat)
        print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

        X_train, y_train = get_feature_target(train_df, feature_cols)
        X_val, y_val = get_feature_target(val_df, feature_cols)
        X_test, y_test = get_feature_target(test_df, feature_cols)

        # Scale features
        X_train_s, X_val_s, X_test_s, x_scaler = scale_data(X_train, X_val, X_test)

        # Scale targets
        y_scaler = StandardScaler()
        y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_val_s = y_scaler.transform(y_val.reshape(-1, 1)).ravel()
        y_test_s = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

        # ── 4. Train models ──────────────────────
        print("\nStep 4: Training models")
        predictions: dict[str, any] = {}
        currency_metrics: dict[str, dict[str, float]] = {}

        # Linear Regression
        print("\n  [Linear Regression]")
        lr_model = build_linear_regression()
        train_sklearn_model(lr_model, X_train_s, y_train_s)
        lr_preds_s = lr_model.predict(X_test_s)
        lr_preds = y_scaler.inverse_transform(lr_preds_s.reshape(-1, 1)).ravel()
        m = compute_metrics(y_test, lr_preds)
        currency_metrics["Linear Regression"] = m
        predictions["Linear Regression"] = lr_preds
        print_metrics("Linear Regression", m)

        # SVR
        print("\n  [SVR]")
        svr_model = build_svr()
        train_sklearn_model(svr_model, X_train_s, y_train_s)
        svr_preds_s = svr_model.predict(X_test_s)
        svr_preds = y_scaler.inverse_transform(svr_preds_s.reshape(-1, 1)).ravel()
        m = compute_metrics(y_test, svr_preds)
        currency_metrics["SVR"] = m
        predictions["SVR"] = svr_preds
        print_metrics("SVR", m)

        # MLP
        print("\n  [MLP]")
        mlp_model, history = train_mlp(X_train_s, y_train_s, X_val_s, y_val_s)
        mlp_preds_s = predict_mlp(mlp_model, X_test_s)
        mlp_preds = y_scaler.inverse_transform(mlp_preds_s.reshape(-1, 1)).ravel()
        m = compute_metrics(y_test, mlp_preds)
        currency_metrics["MLP"] = m
        predictions["MLP"] = mlp_preds
        print_metrics("MLP", m)

        all_metrics[currency] = currency_metrics

        # ── 5. Plots ─────────────────────────────
        print("\nStep 5: Generating figures")
        test_dates = test_df["date"].reset_index(drop=True)

        plot_predictions(
            test_dates, y_test, predictions, currency,
            save_path=os.path.join(config.FIGURES_DIR, f"{currency}_predictions.png"),
        )
        plot_learning_curve(
            history, currency,
            save_path=os.path.join(config.FIGURES_DIR, f"{currency}_learning_curve.png"),
        )
        for model_name, preds in predictions.items():
            safe_name = model_name.replace(" ", "_")
            plot_residuals(
                y_test, preds, model_name, currency,
                save_path=os.path.join(config.FIGURES_DIR, f"{currency}_{safe_name}_residuals.png"),
            )

    # ── 6. Summary table ─────────────────────────
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    summary = results_table(all_metrics)
    print(summary.to_string(index=False))
    summary_path = os.path.join(config.RESULTS_DIR, "results_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    run_pipeline()
