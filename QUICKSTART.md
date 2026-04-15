# QUICKSTART — BANKNIFTY Volatility Forecasting Pipeline

Get from raw NSE data to a live volatility signal in four steps.

---

## Prerequisites

**Python 3.10+**

```bash
pip install pandas numpy scipy arch xgboost scikit-learn shap \
            matplotlib seaborn joblib pyarrow openpyxl
```

For the dashboard:

```bash
pip install streamlit plotly
```

---

## Step 1 — Add raw data

Place NSE BANKNIFTY monthly Excel files in `BANKNIFTY/`.

Both NSE file formats are supported — you can mix them freely:

| Format | Example filename |
|--------|-----------------|
| Newer (string dates) | `BANK_NIFTY_AUG25.xlsx` |
| Older (Excel serial dates) | `BANK_NIFTY_April2023.xlsx` |

The pipeline auto-detects the date format per file. No configuration needed.

---

## Step 2 — Run the preprocessing pipeline

```bash
python preprocess.py
```

This takes 5–15 minutes (IV computation runs in parallel). When it finishes you'll have:

```
data/features/final_features.parquet   ← model input, 29 features
data/features/final_features.csv       ← same, CSV format
```

> After loading files, the pipeline prints a per-file date range table. Verify the dates look correct before proceeding.

---

## Step 3 — Train the XGBoost model

Open `xgboost_volatility_model.py` and set the split dates to match your data:

```python
TRAIN_END = "2025-12-31"   # adjust to your data range
VAL_END   = "2026-02-28"   # test set = everything after this
```

Then run:

```bash
python xgboost_volatility_model.py
```

Trained models are saved to `models/`. Plots go to `outputs/`.

---

## Step 4 — Generate daily signals (live use)

After market close each evening:

```bash
# 1. Format the raw NSE option chain CSV
python option_data_formating.py option-chain-ED-BANKNIFTY-28-Apr-2026.csv

# 2. Run the prediction
python daily_predict.py --chain data/option_chain_processed/option_chain_BANKNIFTY-28-Apr-2026.csv
```

Output:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  BANKNIFTY VOLATILITY SIGNAL
  Date: 2026-04-09
  GARCH Forecast:     0.892%   (ann. 14.16%)
  XGB Correction:    +0.031%
  Final Forecast:     0.923%   (ann. 14.65%)
  Direction:          UNDERESTIMATE  (conf: 68.4%)
  Signal:             BUY VOL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

| Signal | Action |
|--------|--------|
| BUY VOL | Buy options / go long volatility |
| SELL VOL | Sell options / go short volatility |
| HOLD | No edge — skip the trade |

---

## Dashboard (optional)

```bash
streamlit run vol_dashboard/app.py
```

Displays signal, forecast vs realized chart, SHAP-driven context, and recent signal accuracy.

---

## Add more data

Drop new monthly `.xlsx` files into `BANKNIFTY/` and rerun both scripts:

```bash
python preprocess.py
python xgboost_volatility_model.py   # update TRAIN_END / VAL_END first
```

No other changes needed. Both file formats are handled automatically.

---

## Inspect intermediate data

```bash
python parquet_viewer.py data/features/final_features.parquet
python parquet_viewer.py data/processed/master_with_iv.parquet
```

---

## Full documentation

See [DOCUMENTATION.md](DOCUMENTATION.md) for the complete reference: architecture, all configuration parameters, feature engineering details, model history, and troubleshooting.
