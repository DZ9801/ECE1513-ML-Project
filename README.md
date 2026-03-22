# ECE1513 ML Project – Predicting Canadian Dollar Exchange Rates

Predicting Canadian Dollar (CAD) exchange rates against major global currencies (USD, EUR, CNY) using machine learning. Daily exchange-rate data is sourced from the **Bank of Canada Valet API**.

## Project Structure

```
ECE1513-ML-Project/
├── main.py                    # Run the full pipeline (download → train → evaluate)
├── config.py                  # All hyperparameters & paths in one place
├── requirements.txt           # Python dependencies
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Download & cache data from Bank of Canada API
│   ├── preprocessing.py       # Feature engineering & train/val/test splitting
│   ├── models.py              # Model definitions (Linear Regression, SVR, MLP)
│   ├── train.py               # Training loops (sklearn + PyTorch)
│   ├── evaluate.py            # Metrics (MAE, RMSE, R²) and plotting
│   └── utils.py               # Seed setting, directory helpers
├── data/                      # Auto-created; cached CSV files
│   ├── USD_CAD.csv
│   ├── EUR_CAD.csv
│   └── CNY_CAD.csv
├── results/
│   ├── figures/               # Auto-created; prediction & residual plots
│   │   ├── USD_predictions.png
│   │   ├── EUR_predictions.png
│   │   ├── CNY_predictions.png
│   │   ├── USD_learning_curve.png
│   │   ├── EUR_learning_curve.png
│   │   ├── CNY_learning_curve.png
│   │   └── *_residuals.png    # Residual histograms for each model × currency
│   └── results_summary.csv    # Auto-created; metrics table
├── report/                    # LaTeX report (Overleaf-ready)
│   ├── report.tex             # Completed NeurIPS-format report
│   ├── neurips.sty            # NeurIPS style file
│   ├── figures/               # Copies of result figures for the report
│   └── report_overleaf.zip    # Ready-to-upload zip for Overleaf
└── reference doc/
    ├── main.tex               # Project proposal
    └── Report_Template.tex    # Report template (NeurIPS format)
```

## Problem Description

Exchange-rate forecasting is formulated as a **supervised regression** task:

- **Input**: 23 engineered features from historical daily rates (lagged observations, rolling means/stds, percentage changes, cyclical date encodings).
- **Output**: exchange rate *h* days ahead (default *h* = 1).

## Models

| Model | Description |
|---|---|
| **Linear Regression** | Baseline – ordinary least squares via scikit-learn |
| **SVR** | Support Vector Regression with RBF kernel (C=10, ε=0.01) |
| **MLP** | Multi-Layer Perceptron (PyTorch) with 3 hidden layers [128, 64, 32], BatchNorm, Dropout, and early stopping |

## Results

| Currency | Model | MAE | RMSE | R² |
|---|---|---|---|---|
| USD/CAD | Linear Regression | 0.0033 | 0.0045 | 0.9707 |
| USD/CAD | SVR | 0.0294 | 0.0369 | −0.9733 |
| USD/CAD | MLP | 0.0539 | 0.0636 | −4.8678 |
| EUR/CAD | Linear Regression | 0.0041 | 0.0053 | 0.9907 |
| EUR/CAD | SVR | 0.0243 | 0.0360 | 0.5684 |
| EUR/CAD | MLP | 0.0660 | 0.0797 | −1.1139 |
| CNY/CAD | Linear Regression | 0.0004 | 0.0006 | 0.9496 |
| CNY/CAD | SVR | 0.0023 | 0.0028 | −0.0455 |
| CNY/CAD | MLP | 0.0021 | 0.0025 | 0.1753 |

Linear Regression achieves the best performance (R² > 0.94) across all currency pairs by leveraging strong autocorrelation in exchange-rate time series.

## Evaluation Metrics

- **MAE** – Mean Absolute Error
- **RMSE** – Root Mean Squared Error
- **R²** – Coefficient of Determination

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python main.py
```

This will:
1. Download daily exchange-rate data from the Bank of Canada (cached to `data/`).
2. Engineer lag, rolling, and calendar features (23 features total).
3. Split chronologically into train (70%) / validation (15%) / test (15%).
4. Train Linear Regression, SVR, and MLP on each currency pair.
5. Print metrics and save plots to `results/figures/` and a summary CSV.

## Report

The completed NeurIPS-format report is in `report/`. To compile on Overleaf, upload `report/report_overleaf.zip` directly as a new project.

## Configuration

All key settings live in `config.py`:

- **Currency pairs** – add or remove pairs in `CURRENCY_PAIRS`.
- **Feature engineering** – adjust `LAG_DAYS`, `ROLLING_WINDOWS`, `FORECAST_HORIZON`.
- **SVR hyperparameters** – `SVR_PARAMS`.
- **MLP hyperparameters** – `MLP_PARAMS` (hidden sizes, learning rate, epochs, early stopping patience, etc.).

## Data Source

[Bank of Canada Valet API](https://www.bankofcanada.ca/valet/docs) – free, public, no API key required.

## Requirements

- Python ≥ 3.10
- See `requirements.txt` for package versions.
