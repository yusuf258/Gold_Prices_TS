# Gold Price Forecasting | LSTM Time Series

LSTM-based deep learning model for forecasting daily gold prices using historical price sequences with a sliding window approach.

## Problem Statement
Predict future **gold closing prices** based on historical price sequences. The model learns temporal dependencies in price movements to generate next-day forecasts.

## Dataset
| Attribute | Detail |
|---|---|
| File | `gold_price_data.csv` |
| Records | 4,682 daily observations |
| Period | 2001 – 2019 (~18 years) |
| Features | Date, Open, High, Low, Close, Volume |
| Target | `Close` price |

## Methodology
1. **EDA & Visualization** — Price trend over time, volume analysis, rolling statistics
2. **Preprocessing** — `MinMaxScaler` normalization (0–1 range)
3. **Sequence Construction** — Sliding window (window size = 30 days) for LSTM input
4. **LSTM Architecture** — 2-layer LSTM with Dropout regularization
5. **Training** — Adam optimizer, MSE loss, `EarlyStopping` + `ModelCheckpoint`
6. **Evaluation** — MSE, RMSE, MAE; actual vs predicted price visualization

## Results
| Model | Metric | Value |
|---|---|---|
| **LSTM (2-layer, window=30)** | Best val_loss (MSE) | **1.897 × 10⁻⁴** |

> The model captures long-term price trends effectively; short-term volatility remains challenging.

## Technologies
`Python` · `TensorFlow/Keras` · `scikit-learn` · `Pandas` · `NumPy` · `Matplotlib` · `joblib`

## File Structure
```
16_Gold_Prices_TS/
├── project_notebook.ipynb    # Main notebook
├── gold_price_data.csv       # Dataset
└── models/                   # Saved LSTM model
```

## How to Run
```bash
cd 16_Gold_Prices_TS
jupyter notebook project_notebook.ipynb
```
