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
│   ├── models.py              # Model definitions (Linear Regression, SVR, MLP, LSTM)
│   ├── train.py               # Training loops (sklearn + PyTorch)
│   ├── evaluate.py            # Metrics (MAE, RMSE, R²) and plotting
│   └── utils.py               # Seed setting, directory helpers
├── data/                      # Auto-created; cached CSV files
│   ├── USD_CAD.csv
│   ├── EUR_CAD.csv
│   └── CNY_CAD.csv
├── results/
│   ├── figures/               # Auto-created; prediction & residual plots
│   │   ├── *_predictions.png
│   │   ├── *_LSTM_predictions.png  # LSTM prediction plots
│   │   ├── *_MLP_learning_curve.png
│   │   ├── *_LSTM_learning_curve.png
│   │   └── *_residuals.png    # Residual histograms for each model × currency
│   ├── svr_grid_search/       # All SVR parameter combinations and validation metrics
│   │   ├── USD_svr_grid_search.csv
│   │   ├── EUR_svr_grid_search.csv
│   │   ├── CNY_svr_grid_search.csv
│   │   └── figures/           # C/epsilon heatmaps for validation RMSE and R²
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

- **Input**: 22 engineered features from historical daily rates (lagged observations, rolling means/stds, percentage changes, cyclical date encodings).
- **Output**: exchange rate *h* days ahead (default *h* = 1).

## Models

| Model | Description |
|---|---|
| **Persistence Baseline** | Predicts the next business-day rate as the current observed rate |
| **Linear Regression** | Ordinary least squares via scikit-learn |
| **SVR** | RBF Support Vector Regression with validation-set grid-search tuning |
| **MLP** | Multi-Layer Perceptron (PyTorch) with 3 hidden layers [128, 64, 32], BatchNorm, Dropout, and early stopping |
| **LSTM** | Long Short-Term Memory network (PyTorch) with 2 layers, hidden size 64, sequence length 21, and early stopping |

## Results

| Currency | Model | MAE | RMSE | R² |
|---|---|---|---|---|
| USD/CAD | Persistence Baseline | 0.0032 | 0.0044 | 0.9715 |
| USD/CAD | Linear Regression | 0.0033 | 0.0045 | 0.9707 |
| USD/CAD | SVR | 0.0298 | 0.0376 | −1.0490 |
| USD/CAD | MLP | 0.0062 | 0.0073 | 0.9225 |
| USD/CAD | LSTM | 0.0144 | 0.0168 | 0.5423 |
| EUR/CAD | Persistence Baseline | 0.0040 | 0.0052 | 0.9909 |
| EUR/CAD | Linear Regression | 0.0041 | 0.0053 | 0.9906 |
| EUR/CAD | SVR | 0.0269 | 0.0404 | 0.4557 |
| EUR/CAD | MLP | 0.0116 | 0.0140 | 0.9350 |
| EUR/CAD | LSTM | 0.0105 | 0.0139 | 0.9357 |
| CNY/CAD | Persistence Baseline | 0.0004 | 0.0006 | 0.9503 |
| CNY/CAD | Linear Regression | 0.0004 | 0.0006 | 0.9496 |
| CNY/CAD | SVR | 0.0007 | 0.0009 | 0.8923 |
| CNY/CAD | MLP | 0.0005 | 0.0007 | 0.9420 |
| CNY/CAD | LSTM | 0.0005 | 0.0007 | 0.9407 |

The persistence baseline achieves the best test performance across all three currency pairs, narrowly outperforming Linear Regression. This shows that the high level-prediction R² values are driven largely by strong day-to-day autocorrelation and that none of the learned models beats the simple random-walk forecast on this test period. MLP and LSTM remain competitive on EUR/CAD and CNY/CAD, while tuned SVR performs poorly on USD/CAD.

## SVR Grid Search Results

SVR hyperparameters are selected by minimizing validation-set RMSE over 24
combinations:

- `C`: 0.1, 1, 10, 100
- `epsilon`: 0.001, 0.01, 0.1
- `gamma`: `scale`, `auto`

| Currency | Best C | Best epsilon | Best gamma | Validation MAE | Validation RMSE | Validation R² |
|---|---:|---:|---|---:|---:|---:|
| USD/CAD | 1 | 0.1 | scale | 0.006669 | 0.008675 | 0.7432 |
| EUR/CAD | 1 | 0.1 | scale | 0.003799 | 0.004955 | 0.9238 |
| CNY/CAD | 100 | 0.1 | scale | 0.000916 | 0.001377 | 0.6493 |

These are **validation-set** metrics used for hyperparameter selection. The
metrics in the main Results table are computed later on the untouched test
set. In particular, USD/CAD falls from validation $R^2 = 0.7432$ to test
$R^2 = -1.0490$, indicating poor generalization across market periods.

Complete parameter-level results:

- [USD/CAD grid search](results/svr_grid_search/USD_svr_grid_search.csv)
- [EUR/CAD grid search](results/svr_grid_search/EUR_svr_grid_search.csv)
- [CNY/CAD grid search](results/svr_grid_search/CNY_svr_grid_search.csv)
- [RMSE and R² heatmaps](results/svr_grid_search/figures/)

## Evaluation Metrics

- **MAE** – Mean Absolute Error
- **RMSE** – Root Mean Squared Error
- **R²** – Coefficient of Determination

## Evaluation Protocol

- Data covers January 2017 through December 2025.
- Samples are split chronologically into training (70%), validation (15%), and
  test (15%) sets without shuffling.
- Feature and target scalers are fitted on the training set only.
- The validation set selects SVR hyperparameters and neural-network stopping
  points; the test set is reserved for final metrics.
- The current results use one chronological split. Walk-forward validation is
  documented in the report as a recommended extension but is not yet used in
  the reported metrics.

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
2. Engineer lag, rolling, and calendar features (22 features total).
3. Split chronologically into train (70%) / validation (15%) / test (15%).
4. Evaluate the persistence baseline and train Linear Regression, SVR, MLP, and LSTM on each currency pair.
5. Print metrics and save plots to `results/figures/` and a summary CSV.

All SVR grid-search combinations and validation metrics are saved as one CSV
per currency in `results/svr_grid_search/`.

## Report

The completed NeurIPS-format report is in `report/`. To compile on Overleaf, upload `report/report_overleaf.zip` directly as a new project.

## Configuration

All key settings live in `config.py`:

- **Currency pairs** – add or remove pairs in `CURRENCY_PAIRS`.
- **Feature engineering** – adjust `LAG_DAYS`, `ROLLING_WINDOWS`, `FORECAST_HORIZON`.
- **MLP hyperparameters** – `MLP_PARAMS` (hidden sizes, learning rate, epochs, early stopping patience, etc.).
- **LSTM hyperparameters** – `LSTM_PARAMS` (hidden size, num layers, sequence length, etc.).
- **SVR tuning grid** – `SVR_TUNING_GRID` for hyperparameter search.

## Data Source

[Bank of Canada Valet API](https://www.bankofcanada.ca/valet/docs) – free, public, no API key required.

## Requirements

- Python ≥ 3.10
- See `requirements.txt` for package versions.
