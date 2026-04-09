a# BANKNIFTY Volatility Forecasting Pipeline

A two-stage volatility forecasting system for BANKNIFTY options using **GARCH(1,1) + XGBoost**. The pipeline ingests raw NSE options data, computes implied volatility, builds a daily feature table, fits a rolling GARCH model, and then trains an XGBoost model to correct GARCH's systematic blind spots using options market signals.

**Author:** Divyansh Pathak — [divyansh.pathak129@gmail.com](mailto:divyansh.pathak129@gmail.com)

---

## How It Works

```
Stage 1 — GARCH(1,1)
  Spot returns  →  baseline volatility forecast

Stage 2 — XGBoost correction
  GARCH forecast + options signals (IV, skew, OI, volume, term structure)
  →  predict whether GARCH under/overestimates tomorrow's vol
  →  Final forecast = GARCH forecast ± XGBoost correction
```

GARCH only sees historical price returns. It is blind to options market signals — OI spikes, skew surges, expiry effects. XGBoost learns these patterns from the residuals GARCH leaves behind.

---

## Repository Structure

```
stockspipeline/
├── preprocess.py                    # Full data pipeline (Phases 1–7)
├── xgboost_volatility_model.py      # XGBoost model (Phases 0–8)
├── option_data_formating.py         # Format raw NSE option chain CSVs for daily use
├── daily_predict.py                 # Daily volatility signal generator (live trading)
├── xgboost_volatility_model_plan.md # Detailed implementation plan
├── parquet_viewer.py                # Utility to inspect .parquet files
├── BANKNIFTY/                       # Raw NSE monthly Excel files (input)
│   ├── BANK_NIFTY_AUG25.xlsx        # Dataset1 format (string dates)
│   ├── BANK_NIFTY_April2023.xlsx    # Dataset2 format (Excel serial dates)
│   └── ...
├── data/
│   ├── raw/                         # master_raw.parquet
│   ├── processed/                   # master_filtered.parquet, master_with_iv.parquet
│   ├── features/                    # final_features.parquet, final_features.csv
│   ├── option_chain_raw/            # Raw NSE option chain CSVs (input for formatter)
│   ├── option_chain_processed/      # Cleaned/formatted option chain CSVs
│   └── daily_state.json             # Persists yesterday's OI/volume/PCR for delta computation
├── vol_dashboard/                   # Streamlit dashboard (read-only, never re-runs pipeline)
│   ├── app.py                       # Entry point — run with: streamlit run vol_dashboard/app.py
│   ├── data_loader.py               # All file loading, caching, merging, unit normalisation
│   ├── utils.py                     # Shared vol formatting helpers
│   ├── requirements.txt             # Dashboard-specific dependencies
│   └── components/
│       ├── signal_hero.py           # Panel 1: BUY VOL / SELL VOL / HOLD hero
│       ├── forecast_cards.py        # Panel 2: GARCH / Correction / Final metric cards
│       ├── forecast_chart.py        # Panel 3: Forecast vs realized line chart
│       ├── signal_log.py            # Panel 4: Recent signal accuracy table
│       └── context_strip.py        # Panel 5: ATM IV / Skew / HV20 / PCR / DTE row
├── assets/                          # Output plots (committed for README display)
├── models/                          # Trained XGBoost models (.ubj, .pkl)
├── outputs/                         # Plots and forecast CSV (gitignored)
└── .gitignore
```

---

## Requirements

**Python 3.10+**

```bash
pip install pandas numpy scipy arch xgboost scikit-learn shap \
            matplotlib seaborn joblib pyarrow openpyxl
```

---

## Quickstart

### Step 1 — Add raw data

Place NSE BANKNIFTY options monthly Excel files in `BANKNIFTY/`. The pipeline supports two file formats that NSE has used across different time periods:

| Format | Example filename | Date column type |
|--------|-----------------|-----------------|
| Dataset1 (newer) | `BANK_NIFTY_AUG25.xlsx` | String `"01-08-2025"` |
| Dataset2 (older) | `BANK_NIFTY_April2023.xlsx` | Excel serial float e.g. `45139.0` |

You can mix both formats freely in the same folder — the pipeline auto-detects the date format per file.

Each file should contain these columns from NSE:

| Column | Description |
|--------|-------------|
| `Date` | Trading date (string, Excel serial, or datetime — all handled) |
| `CONTRACT_D` | Contract descriptor e.g. `OPTIDXBANKNIFTY01-AUG-2025CE45000` |
| `CLOSE_PRIC` | Option close price |
| `SETTLEMENT` | Settlement price (fallback if close is NaN) |
| `UNDRLNG_ST` | Underlying spot price |
| `OI_NO_CON` | Open interest (number of contracts) |
| `TRADED_QUA` | Traded quantity / volume (also accepts `TRADED_QTY`, `VOLUME`, `TRDNG_VALUE`) |

### Step 2 — Run the preprocessing pipeline

```bash
python preprocess.py
```

This runs all 7 preprocessing phases and writes:

| Output file | Description |
|-------------|-------------|
| `data/raw/master_raw.parquet` | All months merged, raw |
| `data/processed/master_filtered.parquet` | After liquidity filters |
| `data/processed/master_with_iv.parquet` | With computed implied volatility |
| `data/processed/daily_pre_lags.parquet` | Daily features + GARCH outputs |
| `data/features/final_features.parquet` | Final feature table (model input) |
| `data/features/final_features.csv` | Same, in CSV format |

> **Note:** IV computation runs in parallel using `joblib` and may take 5–15 minutes depending on your CPU and data size.

> **Date parsing:** After loading all files, the pipeline prints a date-range table showing the earliest date, latest date, and valid row count per source file. Verify these look correct before the pipeline proceeds. Any file with fewer than 100 valid-date rows will trigger a warning.

### Step 3 — Train the XGBoost model

```bash
python xgboost_volatility_model.py
```

### Step 4 — Run daily predictions (live use)

After the model is trained, use `option_data_formating.py` + `daily_predict.py` each evening to generate tomorrow's volatility signal. See the [Option Chain Formatting](#option-chain-formatting) and [Daily Prediction](#daily-prediction-live-signal) sections below for full details.

This runs all 8 model phases and writes:

| Output | Description |
|--------|-------------|
| `models/xgb_classifier.ubj` | Trained classifier (XGBoost native format) |
| `models/xgb_classifier.pkl` | Trained classifier (joblib pickle) |
| `models/xgb_regressor.ubj` | Trained regressor for correction magnitude |
| `models/xgb_regressor.pkl` | Trained regressor (joblib pickle) |
| `outputs/eda_signals.png` | ATM IV, GARCH vs realized vol, skew, GARCH error over time |
| `outputs/class_balance.png` | Target class distribution |
| `outputs/correlation_heatmap.png` | Feature correlation matrix |
| `outputs/feature_importances_baseline.png` | Baseline XGBoost feature importances |
| `outputs/shap_importance.png` | SHAP mean absolute importance (val set) |
| `outputs/shap_beeswarm.png` | SHAP beeswarm — direction and magnitude |
| `outputs/roc_curve.png` | ROC curve on test set |
| `outputs/forecast_comparison.png` | GARCH vs GARCH+XGBoost vs realized vol |
| `outputs/test_set_forecasts.csv` | Per-day forecasts with corrections on test set |

---

## Sample Outputs

### EDA — ATM IV, GARCH vs Realized Vol, Skew, GARCH Error

![EDA Signals](assets/eda_signals.png)

---

### Class Balance — GARCH Over vs Underestimation

![Class Balance](assets/class_balance.png)

---

### Feature Correlation Matrix

![Correlation Heatmap](assets/correlation_heatmap.png)

---

### Baseline XGBoost Feature Importances

![Feature Importances](assets/feature_importances_baseline.png)

---

### SHAP — Global Feature Importance

![SHAP Importance](assets/shap_importance.png)

---

### SHAP — Direction and Magnitude of Each Feature

![SHAP Beeswarm](assets/shap_beeswarm.png)

---

### ROC Curve — Test Set

![ROC Curve](assets/roc_curve.png)

---

### Forecast Comparison — GARCH vs GARCH+XGBoost vs Realized Vol

![Forecast Comparison](assets/forecast_comparison.png)

---

## Configuration

Key parameters live at the top of each script:

### `preprocess.py`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RISK_FREE_RATE` | `0.065` | RBI repo rate used in Black-Scholes IV calculation |
| `OI_MIN` | `50` | Minimum open interest to include a contract |
| `MONEYNESS_LOW` | `0.80` | Lower bound for strike/spot moneyness filter |
| `MONEYNESS_HIGH` | `1.20` | Upper bound for moneyness filter |
| `DTE_MIN` | `1` | Minimum days to expiry |
| `DTE_MAX` | `90` | Maximum days to expiry |
| `IV_MAX` | `2.0` | Cap IV at 200% annualized — higher is likely a data error |
| `GARCH_WARMUP` | `60` | Trading days used to warm up the rolling GARCH window |

### `xgboost_volatility_model.py`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TRAIN_END` | `"2025-12-31"` | Last date of training set |
| `VAL_END` | `"2026-02-28"` | Last date of validation set (test = everything after) |

Update `TRAIN_END` and `VAL_END` to match your actual data range whenever you add new monthly files.

---

## Features Used by the Model

The model uses 22 features derived from options data and GARCH outputs:

| Feature | Source | Description |
|---------|--------|-------------|
| `GARCH_Forecast` | GARCH | One-day-ahead volatility forecast |
| `GARCH_Residual_lag1/2` | GARCH | Lagged standardized residuals |
| `ATM_IV`, `ATM_IV_lag1/2` | Options | At-the-money implied volatility and lags |
| `ATM_IV_5d_mean` | Options | 5-day rolling mean of ATM IV |
| `Skew`, `Skew_lag1` | Options | Put IV (0.95m) minus Call IV (1.05m) |
| `TS_Slope` | Options | Term structure: far expiry IV minus near expiry IV |
| `IV_HV_Spread` | Derived | ATM IV minus 20-day historical volatility |
| `OI_Change`, `OI_Change_lag1` | Options | Daily change in total open interest |
| `PCR_OI` | Options | Put-call ratio by open interest |
| `Volume_Change` | Options | Daily change in total traded volume |
| `PCR_Volume` | Options | Put-call ratio by volume |
| `DTE_nearest` | Calendar | Days to nearest expiry |
| `Is_expiry_week` | Calendar | 1 if within 5 days of expiry, else 0 |
| `Days_since_last_expiry` | Calendar | Days elapsed since last expiry date |
| `HV_10`, `HV_20`, `HV_30` | Returns | Historical volatility over 10/20/30-day windows |

---

## Target Variable

**Classification (primary):** `Target_Binary`
- `1` — GARCH underestimated realized volatility (positive GARCH error)
- `0` — GARCH overestimated realized volatility

**Regression (secondary):** `Target_Regression`
- The exact signed error: `Realized_Vol_proxy − GARCH_Forecast`
- Used by the regressor to compute the correction magnitude

---

## Performance Notes

Model performance scales heavily with the amount of data. With more months of data:

| Training rows | Expected val accuracy |
|---------------|----------------------|
| ~40 rows | Overfits — unreliable |
| ~150 rows | 55–60% realistic |
| ~250+ rows | 60–65% achievable |

To add more data, drop new monthly files into `BANKNIFTY/` and rerun both scripts from scratch. Both Dataset1 (string dates) and Dataset2 (Excel serial dates) formats are handled automatically. No other changes needed.

A 60% classification accuracy on financial time series is considered excellent. The more meaningful metric for live trading use is **direction accuracy** — whether the combined forecast is closer to realized vol than GARCH alone.

---

## Using the Trained Model

Once you have run both scripts, the trained models sit in `models/`. Here is how to load them and get a forecast for a new day.

### What you need as input

To forecast for **tomorrow**, you need today's feature row — the same 22 columns the model was trained on. The easiest way is to pull the last row from `final_features.parquet` after running the pipeline on fresh data.

```python
import pandas as pd
import xgboost as xgb
import numpy as np

FEATURE_COLS = [
    "GARCH_Forecast", "GARCH_Residual_lag1", "GARCH_Residual_lag2",
    "ATM_IV", "ATM_IV_lag1", "ATM_IV_lag2", "ATM_IV_5d_mean",
    "Skew", "Skew_lag1", "TS_Slope", "IV_HV_Spread",
    "OI_Change", "OI_Change_lag1", "PCR_OI", "Volume_Change", "PCR_Volume",
    "DTE_nearest", "Is_expiry_week", "Days_since_last_expiry",
    "HV_10", "HV_20", "HV_30",
]

# Load today's features (last row of the pipeline output)
features = pd.read_parquet("data/features/final_features.parquet")
today = features[FEATURE_COLS].iloc[[-1]]  # shape (1, 22)
```

### Get the classification signal

The classifier tells you whether GARCH is likely **underestimating** tomorrow's volatility.

```python
clf = xgb.XGBClassifier()
clf.load_model("models/xgb_classifier.ubj")

predicted_class = clf.predict(today)[0]
probability     = clf.predict_proba(today)[0][1]

if predicted_class == 1:
    print(f"GARCH likely UNDERESTIMATES tomorrow's vol  (confidence: {probability:.1%})")
else:
    print(f"GARCH likely OVERESTIMATES tomorrow's vol  (confidence: {1 - probability:.1%})")
```

### Get the corrected volatility forecast

The regressor predicts the exact correction to apply on top of GARCH's forecast.

```python
reg = xgb.XGBRegressor()
reg.load_model("models/xgb_regressor.ubj")

garch_forecast  = features["GARCH_Forecast"].iloc[-1]
xgb_correction  = reg.predict(today)[0]
final_forecast  = garch_forecast + xgb_correction

print(f"GARCH forecast:          {garch_forecast*100:.3f}%")
print(f"XGBoost correction:      {xgb_correction*100:+.3f}%")
print(f"Final vol forecast:      {final_forecast*100:.3f}%")
print(f"Annualized (x sqrt252):  {final_forecast * np.sqrt(252) * 100:.2f}%")
```

### Practical workflow for live use

```
Every evening after market close (4:00 PM IST):
  4:00 PM  Market closes
  4:15 PM  Download today's NSE BANKNIFTY option chain CSV
           nseindia.com → Option Chain → BANKNIFTY → Download
  Step 1:  python option_data_formating.py <downloaded_chain_filename.csv>
  Step 2:  python preprocess.py
  Step 3:  python daily_predict.py --chain data/option_chain_processed/<formatted_chain.csv>
  
  Act:     Next morning if signal = ACT
```

> **Retrain periodically.** Re-run `xgboost_volatility_model.py` every month or two as you accumulate more data. Update `TRAIN_END` and `VAL_END` to reflect the new date range before retraining.

### Reading `outputs/test_set_forecasts.csv`

This file shows the model's performance day-by-day on the held-out test set:

| Column | Description |
|--------|-------------|
| `GARCH_Forecast` | Raw GARCH one-day-ahead vol forecast |
| `GARCH_Error` | How wrong GARCH was (realized − forecast) |
| `Realized_Vol_proxy` | Actual realized volatility for that day |
| `Target_Binary` | Ground truth (1 = GARCH underestimated) |
| `XGB_Predicted_Class` | Model's prediction (1 or 0) |
| `XGB_Pred_Probability` | Confidence score (0–1) for class 1 |
| `XGB_Correction` | Correction the regressor applied |
| `Final_Forecast` | `GARCH_Forecast + XGB_Correction` |
| `Correct_Direction` | 1 if the model predicted direction correctly |

---

## Option Chain Formatting

Before running `daily_predict.py`, the raw option chain CSV downloaded from NSE must be cleaned and standardised using `option_data_formating.py`.

### Step 1 — Place the raw file

Copy the downloaded NSE option chain CSV into `data/option_chain_raw/`. The filename must follow NSE's standard format, e.g.:

```
option-chain-ED-BANKNIFTY-28-Apr-2026.csv
```

### Step 2 — Run the formatter

```bash
python option_data_formating.py option-chain-ED-BANKNIFTY-28-Apr-2026.csv
```

The script:
- Strips the double-header NSE layout and renames all 21 columns to lowercase internal names
- Removes comma separators and converts all numeric columns
- Interpolates missing IV values along the vol smile (linear, both directions)
- Fills missing OI/volume with 0
- Drops extreme deep-OTM strikes with no data on either side
- Appends `asset` and `expiry_date` metadata columns extracted from the filename

Output is written to `data/option_chain_processed/option_chain_BANKNIFTY-28-Apr-2026.csv`.

---

## Daily Prediction (Live Signal)

`daily_predict.py` combines today's option chain with the historical GARCH and HV features to produce a single actionable volatility signal.

### Usage

```bash
# Minimal — spot auto-detected from put-call parity
python daily_predict.py --chain data/option_chain_processed/option_chain_BANKNIFTY-28-Apr-2026.csv

# With explicit spot price
python daily_predict.py --chain data/option_chain_processed/option_chain_BANKNIFTY-28-Apr-2026.csv --spot 52500

# With explicit next expiry
python daily_predict.py --chain data/option_chain_processed/option_chain_BANKNIFTY-28-Apr-2026.csv \
                        --spot 52500 --next-expiry 2026-04-30
```

### What it outputs

The script prints a structured signal block:

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

It also writes `data/daily_state.json` to persist OI/volume/PCR from today so that tomorrow's delta-based features (OI_Change, Volume_Change) are computed correctly.

### Signal interpretation

| Signal | Meaning |
|--------|---------|
| `BUY VOL` | Model predicts GARCH is underestimating — buy options / go long volatility |
| `SELL VOL` | Model predicts GARCH is overestimating — sell options / go short volatility |
| `HOLD` | Confidence below threshold — no actionable edge |

---

## Streamlit Dashboard

A personal daily-use dashboard that reads from pipeline output files and displays five panels: signal hero, forecast breakdown cards, forecast vs realized chart, recent signal log, and market context strip. It never re-runs the pipeline or model — it is read-only.

### Install dashboard dependencies

```bash
pip install streamlit plotly
```

Or install everything from the dashboard's own requirements file:

```bash
pip install -r vol_dashboard/requirements.txt
```

### Launch

Always run from the **project root**:

```bash
streamlit run vol_dashboard/app.py
```

### Panels

| Panel | What it shows |
|-------|--------------|
| **Signal Hero** | BUY VOL / SELL VOL / HOLD in large colored text, final annualised forecast, confidence bar |
| **Forecast Cards** | Three `st.metric` cards: GARCH forecast → XGB correction → Final forecast |
| **Forecast Chart** | Full-history Plotly line chart — Realized Vol (gold), GARCH (grey dashed), GARCH+XGB (blue). Range selector buttons: 1M / 3M / 6M / All |
| **Signal Log** | Last 15 test-period days with signal, confidence, and ✅/❌ direction accuracy |
| **Market Context** | ATM IV, Skew, HV 20, PCR (OI) with day-over-day delta, days to nearest expiry |

### Sidebar

- **Date picker** — defaults to the latest available trading day; browse any date in the full history
- **Training-period warning** — shown when the selected date has no XGB forecast (before the test split)
- **Stale data warning** — shown only if latest data is more than 7 days old; prompts you to add new monthly files to `BANKNIFTY/` and re-run `preprocess.py`
- **Refresh button** — clears the Streamlit data cache and reloads all parquet/CSV files

### Data flow

```
data/features/final_features.parquet  ──┐
                                         ├──  build_master()  ──  all 5 panels
outputs/test_set_forecasts.csv         ──┘
```

`build_master()` merges both files on date, fills `Realized_Vol_proxy` gaps from `HV_10`, and normalises all vol columns from decimal to percent (×100) in one place so no component needs to handle units.

### Signal colors

| Signal | Color |
|--------|-------|
| BUY VOL | Green |
| SELL VOL | Red |
| HOLD | Amber |
| NO XGB DATA | Grey (training-period date) |

---

## Inspect Parquet Files

To quickly inspect any intermediate dataset:

```bash
python parquet_viewer.py data/features/final_features.parquet
python parquet_viewer.py data/processed/master_with_iv.parquet
```

---

## License

MIT License — free to use, modify, and distribute with attribution.

---

## Contributing

Pull requests are welcome. If you're adding support for other indices (NIFTY 50, FINNIFTY) or alternate volatility models (EGARCH, GJR-GARCH), please open an issue first to discuss the approach.
