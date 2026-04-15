# BANKNIFTY Volatility Forecasting Pipeline — Full Documentation

**Author:** Divyansh Pathak — [divyansh.pathak129@gmail.com](mailto:divyansh.pathak129@gmail.com)

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Repository Layout](#2-repository-layout)
3. [Data Ingestion & Format Support](#3-data-ingestion--format-support)
4. [Preprocessing Pipeline — preprocess.py](#4-preprocessing-pipeline--preprocesspy)
5. [XGBoost Model — xgboost_volatility_model.py](#5-xgboost-model--xgboost_volatility_modelpy)
6. [Feature Engineering Reference](#6-feature-engineering-reference)
7. [Target Variable Design](#7-target-variable-design)
8. [Option Chain Formatter — option_data_formating.py](#8-option-chain-formatter--option_data_formatingpy)
9. [Daily Prediction — daily_predict.py](#9-daily-prediction--daily_predictpy)
10. [Streamlit Dashboard — vol_dashboard/](#10-streamlit-dashboard--vol_dashboard)
11. [Configuration Reference](#11-configuration-reference)
12. [Performance Benchmarks & Model History](#12-performance-benchmarks--model-history)
13. [Anti-Leakage Checklist](#13-anti-leakage-checklist)
14. [Common Pitfalls & Fixes](#14-common-pitfalls--fixes)
15. [Data Schema Reference](#15-data-schema-reference)

---

## 1. System Architecture

The system is a **two-stage volatility forecasting pipeline** for BANKNIFTY options.

```
┌─────────────────────────────────────────────────────────────┐
│                     STAGE 1 — GARCH(1,1)                    │
│                                                             │
│  NSE Monthly Excel Files  →  preprocess.py  →  GARCH(1,1)  │
│  (raw options data)           (7 phases)       baseline vol │
└─────────────────────────────────────┬───────────────────────┘
                                      │ GARCH_Forecast
                                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   STAGE 2 — XGBoost Correction              │
│                                                             │
│  GARCH_Forecast                                             │
│  + ATM_IV, Skew, OI, Volume,     →  XGBoost Classifier      │
│    PCR, TS_Slope, HV, Calendar       (direction: over/under)│
│  + 7 enhancement features                                   │
│                                   →  XGBoost Regressor      │
│                                       (correction magnitude) │
│                                                             │
│  Final = GARCH_Forecast ± XGBoost_Correction                │
└─────────────────────────────────────────────────────────────┘
```

**Why two stages?** GARCH only sees historical price returns — it is blind to real-time options market signals like OI spikes, skew surges, and expiry effects. XGBoost learns from the *residuals* GARCH leaves behind by incorporating options market intelligence.

---

## 2. Repository Layout

```
Stock-Volatility-Preprocessing-Pipeline/
├── preprocess.py                    # Full data pipeline (Phases 1–7)
├── xgboost_volatility_model.py      # XGBoost model (Phases 0–8)
├── option_data_formating.py         # Format raw NSE option chain CSVs
├── daily_predict.py                 # Daily volatility signal generator
├── parquet_viewer.py                # Utility to inspect .parquet files
│
├── BANKNIFTY/                       # Raw NSE monthly Excel files (INPUT)
│   ├── BANK_NIFTY_AUG25.xlsx        # Dataset1: string dates "01-08-2025"
│   ├── BANK_NIFTY_April2023.xlsx    # Dataset2: Excel serial float 45139.0
│   └── ...
│
├── data/
│   ├── raw/
│   │   └── master_raw.parquet       # All months merged, unfiltered
│   ├── processed/
│   │   ├── master_filtered.parquet  # After liquidity filters
│   │   ├── master_with_iv.parquet   # With computed implied volatility
│   │   └── daily_pre_lags.parquet   # Daily features + GARCH outputs (pre-lag)
│   ├── features/
│   │   ├── final_features.parquet   # Model input — 29 features + targets
│   │   └── final_features.csv       # Same, in CSV
│   ├── option_chain_raw/            # Raw NSE option chain CSVs
│   ├── option_chain_processed/      # Cleaned option chain CSVs
│   └── daily_state.json             # Persists yesterday's OI/volume/PCR
│
├── vol_dashboard/                   # Streamlit dashboard (read-only)
│   ├── app.py                       # Entry point
│   ├── data_loader.py               # File loading, caching, merging
│   ├── utils.py                     # Vol formatting helpers
│   ├── requirements.txt
│   └── components/
│       ├── signal_hero.py           # Panel 1: BUY/SELL/HOLD hero
│       ├── forecast_cards.py        # Panel 2: GARCH/Correction/Final cards
│       ├── forecast_chart.py        # Panel 3: Forecast vs realized chart
│       ├── signal_log.py            # Panel 4: Signal accuracy table
│       └── context_strip.py        # Panel 5: Market context row
│
├── assets/                          # Output plots (committed for README)
├── models/                          # Trained XGBoost models (.ubj, .pkl)
├── outputs/                         # Plots and forecast CSV (gitignored)
│
├── xgboost_volatility_model_plan.md # Detailed implementation plan (archived)
├── model_enhancements.md            # Enhancement analysis (archived)
├── fix_guide_preprocess_multiformat.md  # Multi-format fix guide (archived)
└── .gitignore
```

---

## 3. Data Ingestion & Format Support

### NSE Monthly Excel File Formats

The pipeline auto-detects and handles **two date formats** NSE has used across different time periods. You can mix both freely in the `BANKNIFTY/` folder.

| Format | Example filename | Date column type | Example value |
|--------|----------------|-----------------|---------------|
| Dataset1 (newer, 2024+) | `BANK_NIFTY_AUG25.xlsx` | String | `"01-08-2025"` |
| Dataset2 (older, pre-2024) | `BANK_NIFTY_April2023.xlsx` | Excel serial float | `45139.0` |

The `fix_date_column()` function in `preprocess.py` handles:
- `datetime64` — already parsed by pandas
- String `DD-MM-YYYY` or `DD/MM/YYYY` — Dataset1 format
- Integer `DDMMYY` (6-digit) — NSE FEB26 edge case
- Float Excel serial (> 40,000) — Dataset2 format

### Required Input Columns

| Column | Description |
|--------|-------------|
| `Date` | Trading date (any supported format above) |
| `CONTRACT_D` | Contract descriptor e.g. `OPTIDXBANKNIFTY01-AUG-2025CE45000` |
| `CLOSE_PRIC` | Option close price |
| `SETTLEMENT` | Settlement price (fallback if close is NaN) |
| `UNDRLNG_ST` | Underlying spot price |
| `OI_NO_CON` | Open interest (number of contracts) |
| `TRADED_QUA` | Traded quantity/volume (also accepts `TRADED_QTY`, `VOLUME`, `TRDNG_VALUE`) |

The `CONTRACT_D` string is parsed to extract: option type (CE/PE), strike price, expiry date, and underlying asset.

---

## 4. Preprocessing Pipeline — `preprocess.py`

The pipeline runs 7 sequential phases. Run with:

```bash
python preprocess.py
```

### Phase 1 — Load All Files

- Reads all `.xlsx` files from `BANKNIFTY/`
- Applies `fix_date_column()` per file (see Section 3)
- Parses `CONTRACT_D` to extract `option_type`, `strike`, `expiry`
- Adds `source_file` column for traceability
- Prints per-file date range table — verify this looks correct before proceeding
- Files with < 100 valid-date rows trigger a warning

**Output:** `master_raw.parquet`

### Phase 2 — Compute Days to Expiry (DTE)

- `DTE = expiry_date - trade_date` in calendar days
- Handles monthly and weekly NSE expiry calendars

### Phase 3 — Apply Liquidity Filters

| Filter | Default | Purpose |
|--------|---------|---------|
| `OI_MIN = 50` | OI ≥ 50 | Remove illiquid contracts |
| `MONEYNESS_LOW = 0.80` | strike/spot ≥ 0.80 | Drop deep OTM puts |
| `MONEYNESS_HIGH = 1.20` | strike/spot ≤ 1.20 | Drop deep OTM calls |
| `DTE_MIN = 1` | DTE ≥ 1 | Drop expiry-day contracts |
| `DTE_MAX = 90` | DTE ≤ 90 | Drop very far-dated contracts |

**Output:** `master_filtered.parquet`

### Phase 4 — Compute Implied Volatility

Uses **Black-Scholes** formula with:
- `RISK_FREE_RATE = 0.065` (RBI repo rate)
- Brent's root-finding method to invert BS for IV
- Parallel computation using `joblib` (may take 5–15 min depending on CPU)
- `IV_MAX = 2.0` cap — values above 200% annualized are treated as data errors

**Output:** `master_with_iv.parquet`

### Phase 5 — Build Daily Feature Table

Aggregates from contract-level to daily:

| Feature | Computation |
|---------|------------|
| `ATM_IV` | IV of the contract closest to ATM (min \|strike - spot\|) |
| `Skew` | Put IV at 0.95×spot minus Call IV at 1.05×spot |
| `TS_Slope` | Far-expiry ATM IV minus near-expiry ATM IV |
| `Total_OI` / `PCR_OI` | Sum OI; put OI / call OI |
| `Total_Volume` / `PCR_Volume` | Sum volume; put vol / call vol |
| `OI_Change` / `Volume_Change` | Day-over-day delta |
| `DTE_nearest` | DTE of the nearest-expiry contract |
| `Is_expiry_week` | 1 if within 5 calendar days of an expiry |
| `Days_since_last_expiry` | Calendar days since most recent expiry |
| `log_return` | Log return of underlying spot price |

### Phase 6 — Fit Rolling GARCH

- Uses `arch` library
- Configurable model type: `GARCH_MODEL = "GJR-GARCH"` (default, captures leverage effect for equity indices) or `"GARCH"` (symmetric)
- Rolling window with `GARCH_WARMUP = 60` trading days
- For each day `i`, fits GARCH on `returns[:i]`, forecasts day `i+1`
- Produces: `GARCH_Forecast`, `GARCH_Residual`, `Realized_Vol_proxy`, `GARCH_Error`
- `Target_Binary = 1` if `GARCH_Error > 0` (underestimated)

**Output:** `daily_pre_lags.parquet`

### Phase 7 — Add Lag Features

- `ATM_IV_lag1`, `ATM_IV_lag2` — 1 and 2-day lags of ATM IV
- `ATM_IV_5d_mean` — 5-day rolling mean of ATM IV
- `Skew_lag1` — 1-day lag of skew
- `OI_Change_lag1` — 1-day lag of OI change
- `GARCH_Residual_lag1`, `GARCH_Residual_lag2` — lagged standardized GARCH residuals
- Enhancement features (added in session 2026-04-08):
  - `HV_GARCH_ratio` — HV_20 / GARCH_Forecast (direct underestimation detector)
  - `GARCH_Bias_short` — 5-day rolling mean of GARCH_Error (faster bias signal)
  - `ATM_IV_trend` — ATM_IV minus ATM_IV_5d_mean (IV momentum)
  - `Vol_of_Vol` — 10-day std of ATM IV (vol acceleration)
  - `PCR_OI_change1d` — 1-day PCR OI momentum
  - `HV_GARCH_above_1` — binary flag: HV_GARCH_ratio > 1.0
  - `GARCH_Bias_positive` — binary flag: 5-day GARCH bias > 0

**Output:** `final_features.parquet`, `final_features.csv`

---

## 5. XGBoost Model — `xgboost_volatility_model.py`

Run with:

```bash
python xgboost_volatility_model.py
```

### Phase 0 — Environment Setup

Verifies `final_features.parquet` exists and prints shape, date range, class balance.

### Phase 1 — EDA

- Time-series plots: ATM IV, GARCH vs realized vol, skew, GARCH error
- Class balance check — triggers `scale_pos_weight` computation
- Correlation heatmap

### Phase 2 — Train/Val/Test Split

**Critical:** Never use random splits for time series — causes data leakage.

```
TRAIN_END = "2025-12-31"   # Training set
VAL_END   = "2026-02-28"   # Validation set
# Test = everything after VAL_END
```

Update these constants whenever you add new monthly data.

### Phase 3 — Baseline XGBoost

Baseline classifier trained with `scale_pos_weight = sqrt(neg/pos)` (square root rule — better calibrated than full ratio for probability outputs).

### Phase 4 — Hyperparameter Tuning

Uses `RandomizedSearchCV` (n_iter=60) with `TimeSeriesSplit` (5 folds). Search space includes:

- `max_depth`: [2, 3, 4]
- `learning_rate`: [0.01, 0.05, 0.1]
- `n_estimators`: [100, 200, 300]
- `subsample`: [0.7, 0.8, 1.0]
- `colsample_bytree`: [0.7, 0.8, 1.0]
- `min_child_weight`: [5, 10, 20]
- `gamma`: [0, 0.1, 0.3]
- `reg_alpha`: [0, 0.1, 0.5]
- `reg_lambda`: [0.5, 1.0, 2.0]

`max_depth` deliberately capped at 4 to prevent overfitting on the ~200-row dataset. `min_child_weight` raised to 5–20 for same reason.

### Phase 5 — SHAP Analysis

Generates:
- `shap_importance.png` — mean |SHAP| per feature (global importance)
- `shap_beeswarm.png` — direction and magnitude of each feature's effect

Expected SHAP directions:
- **ATM_IV high** → positive (IV signals more vol than GARCH sees)
- **Skew high** → positive (put demand signals tail risk)
- **HV_GARCH_ratio > 1** → positive (historical vol already exceeds GARCH forecast)
- **DTE_nearest low** → positive (gamma risk near expiry increases realized vol)

### Phase 6 — Test Set Evaluation (run once only)

| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| Accuracy | >55% | >58% | >60% |
| AUC-ROC | >0.55 | >0.60 | >0.65 |
| MAE vs GARCH-only | Any improvement | >5% better | >10% better |

### Phase 7 — Combined Volatility Forecast

1. XGBoost **Regressor** trained to predict `GARCH_Error` (continuous)
2. `Final_Forecast = GARCH_Forecast + XGBoost_Correction`
3. Direction accuracy: fraction of test days where combined forecast is closer to realized vol than GARCH alone

### Phase 8 — Save Outputs

| File | Description |
|------|-------------|
| `models/xgb_classifier.ubj` | Trained classifier (XGBoost native) |
| `models/xgb_classifier.pkl` | Trained classifier (joblib) |
| `models/xgb_regressor.ubj` | Trained regressor (XGBoost native) |
| `models/xgb_regressor.pkl` | Trained regressor (joblib) |
| `outputs/test_set_forecasts.csv` | Per-day test set predictions |
| `outputs/eda_signals.png` | EDA time-series plots |
| `outputs/correlation_heatmap.png` | Feature correlation matrix |
| `outputs/feature_importances_baseline.png` | Baseline feature importances |
| `outputs/shap_importance.png` | SHAP global importance |
| `outputs/shap_beeswarm.png` | SHAP beeswarm |
| `outputs/roc_curve.png` | ROC curve |
| `outputs/forecast_comparison.png` | GARCH vs GARCH+XGBoost vs realized |

---

## 6. Feature Engineering Reference

The model uses **29 features** total (22 original + 7 enhancement features added 2026-04-08):

### Core Options Features

| Feature | Source | Description |
|---------|--------|-------------|
| `ATM_IV` | Options | At-the-money implied volatility |
| `ATM_IV_lag1`, `ATM_IV_lag2` | Options | 1 and 2-day lags |
| `ATM_IV_5d_mean` | Options | 5-day rolling mean |
| `Skew` | Options | Put IV at 0.95m minus Call IV at 1.05m |
| `Skew_lag1` | Options | 1-day lag of skew |
| `TS_Slope` | Options | Far-expiry IV minus near-expiry IV |
| `IV_HV_Spread` | Derived | ATM_IV minus HV_20 |
| `OI_Change`, `OI_Change_lag1` | Options | Day-over-day OI delta and its lag |
| `PCR_OI` | Options | Put-call ratio by open interest |
| `Volume_Change` | Options | Day-over-day volume delta |
| `PCR_Volume` | Options | Put-call ratio by volume |

### GARCH Features

| Feature | Source | Description |
|---------|--------|-------------|
| `GARCH_Forecast` | GARCH | One-day-ahead vol forecast |
| `GARCH_Residual_lag1`, `GARCH_Residual_lag2` | GARCH | Lagged standardized residuals |

### Historical Volatility Features

| Feature | Source | Description |
|---------|--------|-------------|
| `HV_10`, `HV_20`, `HV_30` | Returns | Annualized HV over 10/20/30-day windows |

### Calendar Features

| Feature | Source | Description |
|---------|--------|-------------|
| `DTE_nearest` | Calendar | Days to nearest expiry |
| `Is_expiry_week` | Calendar | 1 if within 5 days of any expiry |
| `Days_since_last_expiry` | Calendar | Days elapsed since last expiry |

### Enhancement Features (added 2026-04-08)

| Feature | Description | Why it matters |
|---------|-------------|----------------|
| `HV_GARCH_ratio` | HV_20 / GARCH_Forecast | When > 1.0, historical vol already exceeds GARCH — direct underestimation signal. Highest correlation with target (0.148) |
| `GARCH_Bias_short` | 5-day rolling mean of GARCH_Error | Faster-reacting bias signal than the 20-day version |
| `ATM_IV_trend` | ATM_IV minus ATM_IV_5d_mean | IV momentum — rising IV not captured in level alone |
| `Vol_of_Vol` | 10-day std of ATM_IV | Vol acceleration / uncertainty regime detection |
| `PCR_OI_change1d` | 1-day change in PCR_OI | Put/call positioning momentum |
| `HV_GARCH_above_1` | Binary: HV_GARCH_ratio > 1.0 | Clean split boundary for XGBoost tree splits |
| `GARCH_Bias_positive` | Binary: 5-day bias > 0 | Explicit regime indicator for GARCH's recent direction |

---

## 7. Target Variable Design

### Classification Target — `Target_Binary`

```
Target_Binary = 1   if Realized_Vol_proxy > GARCH_Forecast  (GARCH underestimated)
Target_Binary = 0   if Realized_Vol_proxy ≤ GARCH_Forecast  (GARCH overestimated)
```

- `Realized_Vol_proxy` = 10-day rolling std of log returns, annualized, shifted back to align with "today's realized vol"
- This is the proxy for actual realized volatility since intraday tick data is unavailable

### Regression Target — `Target_Regression`

```
Target_Regression = Realized_Vol_proxy − GARCH_Forecast
```

Used by the regressor to compute the signed correction magnitude.

### Known Class Imbalance

GARCH tends to overestimate vol in trending/low-vol regimes (class 0 dominates) and underestimate in spike/expiry regimes (class 1 dominates). Training and test sets may have very different class distributions — this was the root cause of the AUC collapse from 0.61 → 0.33 in March 2026 test data (see Section 12).

---

## 8. Option Chain Formatter — `option_data_formating.py`

Cleans raw NSE option chain CSVs for use by `daily_predict.py`.

### Input

Place the raw downloaded CSV in `data/option_chain_raw/`. Filename must follow NSE convention:

```
option-chain-ED-BANKNIFTY-28-Apr-2026.csv
```

### Run

```bash
python option_data_formating.py option-chain-ED-BANKNIFTY-28-Apr-2026.csv
```

### What it does

1. Strips NSE's double-header layout
2. Renames all 21 columns to lowercase internal names
3. Removes comma separators, converts all numeric columns
4. Interpolates missing IV values along the vol smile (linear, bidirectional)
5. Fills missing OI/volume with 0
6. Drops extreme deep-OTM strikes with no data on either side
7. Appends `asset` and `expiry_date` metadata columns (extracted from filename)

### Output

```
data/option_chain_processed/option_chain_BANKNIFTY-28-Apr-2026.csv
```

---

## 9. Daily Prediction — `daily_predict.py`

Produces a single actionable volatility signal for the next trading day.

### Prerequisites

1. `option_data_formating.py` has been run → processed chain CSV exists
2. `preprocess.py` has been run → `final_features.parquet` is up to date
3. `xgboost_volatility_model.py` has been run → models exist in `models/`

### Usage

```bash
# Auto-detect spot from put-call parity
python daily_predict.py --chain data/option_chain_processed/option_chain_BANKNIFTY-28-Apr-2026.csv

# With explicit spot price
python daily_predict.py --chain ... --spot 52500

# With explicit next expiry
python daily_predict.py --chain ... --spot 52500 --next-expiry 2026-04-30
```

### Signal Output

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

### Signal Interpretation

| Signal | Meaning | Action |
|--------|---------|--------|
| `BUY VOL` | GARCH likely underestimating tomorrow's vol | Buy options / go long volatility |
| `SELL VOL` | GARCH likely overestimating tomorrow's vol | Sell options / go short volatility |
| `HOLD` | Confidence below threshold — no edge | No trade |

### State Persistence

Writes `data/daily_state.json` to persist today's OI/volume/PCR so that tomorrow's delta-based features (`OI_Change`, `Volume_Change`) are computed correctly.

### Recommended Daily Workflow

```
Every evening after market close (4:00 PM IST):
  4:00 PM  Market closes
  4:15 PM  Download today's BANKNIFTY option chain CSV from nseindia.com
  Step 1:  python option_data_formating.py <downloaded_chain.csv>
  Step 2:  python preprocess.py          (if new monthly data added)
  Step 3:  python daily_predict.py --chain data/option_chain_processed/<formatted.csv>
  
  Next morning: act on the signal
```

---

## 10. Streamlit Dashboard — `vol_dashboard/`

A read-only personal dashboard that visualizes pipeline output. It never re-runs the pipeline or model.

### Launch

Always run from the **project root**:

```bash
streamlit run vol_dashboard/app.py
```

### Install dependencies

```bash
pip install -r vol_dashboard/requirements.txt
# or: pip install streamlit plotly
```

### Data Flow

```
data/features/final_features.parquet  ──┐
                                         ├──  data_loader.build_master()  ──  all panels
outputs/test_set_forecasts.csv         ──┘
```

`build_master()` in `data_loader.py`:
- Merges both files on date
- Fills `Realized_Vol_proxy` gaps from `HV_10`
- Normalizes all vol columns from decimal to percent (×100) in one place

### Panel Reference

| Panel | Component | What it shows |
|-------|-----------|--------------|
| 1 | `signal_hero.py` | BUY/SELL/HOLD in large colored text, final annualized forecast, confidence bar |
| 2 | `forecast_cards.py` | Three `st.metric` cards: GARCH → XGB correction → Final |
| 3 | `forecast_chart.py` | Full-history Plotly line chart: Realized (gold), GARCH (grey dashed), GARCH+XGB (blue). Range selector: 1M / 3M / 6M / All |
| 4 | `signal_log.py` | Last 15 test-period days with signal, confidence, ✅/❌ direction |
| 5 | `context_strip.py` | ATM IV, Skew, HV20, PCR (OI) with day-over-day delta, DTE |

### Sidebar

- **Date picker** — defaults to latest trading day; browse full history
- **Training-period warning** — shown when selected date has no XGB forecast
- **Stale data warning** — shown when latest data is > 7 days old
- **Refresh button** — clears Streamlit data cache, reloads all files

### Signal Colors

| Signal | Color |
|--------|-------|
| BUY VOL | Green |
| SELL VOL | Red |
| HOLD | Amber |
| NO XGB DATA (training period) | Grey |

---

## 11. Configuration Reference

### `preprocess.py` — Top-level constants

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATA_DIR` | `"BANKNIFTY"` | Folder containing raw monthly Excel files |
| `RISK_FREE_RATE` | `0.065` | RBI repo rate for Black-Scholes IV |
| `OI_MIN` | `50` | Minimum OI to include a contract |
| `MONEYNESS_LOW` | `0.80` | Lower moneyness filter (strike/spot) |
| `MONEYNESS_HIGH` | `1.20` | Upper moneyness filter |
| `DTE_MIN` | `1` | Minimum days to expiry |
| `DTE_MAX` | `90` | Maximum days to expiry |
| `IV_MAX` | `2.0` | Max IV (200% annualized) — caps data errors |
| `GARCH_WARMUP` | `60` | Trading days to warm up rolling GARCH |
| `GARCH_MODEL` | `"GJR-GARCH"` | GARCH model type. Options: `"GARCH"`, `"GJR-GARCH"`, `"EGARCH"` |

### `xgboost_volatility_model.py` — Top-level constants

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TRAIN_END` | `"2025-12-31"` | Last date of training set |
| `VAL_END` | `"2026-02-28"` | Last date of validation set |

**Update these whenever you add new monthly files.**

---

## 12. Performance Benchmarks & Model History

### Model Evolution

| Session | AUC Val | AUC Test | Key Change |
|---------|---------|----------|-----------|
| Initial (8-month data) | 0.49 | N/A | Baseline with 22 features |
| After adding more months | 0.61 | 0.33 | More data; test regime mismatch discovered |
| After enhancements (2026-04-08) | ~0.64 | ~0.58 | 7 new features, GJR-GARCH, regularization fixes |

### Root Cause of AUC Collapse (0.61 → 0.33)

**Problem:** GARCH(1,1) overestimated vol ~80% of the time in the Nov–Dec 2025 training tail (class 0 dominated). March 2026 test set had 63% class 1 (GARCH underestimating) — a completely opposite regime.

**Mechanism:** The model learned "always predict class 0" from training data. When the test regime inverted, AUC dropped below 0.5 (model was confidently wrong).

**Diagnostic signal:** `HV_GARCH_ratio` (HV20 / GARCH_Forecast) was 0.5–0.6 in the training tail and jumped to 1.3–1.6 in March 2026. This ratio was not a feature, so the model had no way to detect the regime shift.

### Fixes Applied (2026-04-08)

1. **HV_GARCH_ratio** added as feature — direct underestimation detector (correlation 0.148 with target)
2. **GJR-GARCH** substituted for symmetric GARCH(1,1) — better captures the asymmetric leverage effect in equity indices
3. **7 total enhancement features** added (see Section 6)
4. **Regularization tightened** — `max_depth` [2,3,4], `min_child_weight` [5,10,20], added `gamma`, `reg_alpha`, `reg_lambda` to search
5. **Platt calibration removed** — was compressing probabilities to val set base rate (18% class 1) while test had 63% class 1, destroying discriminative power. Raw XGBoost probabilities + threshold sweep is more robust

### Scale of Data vs Performance

| Training rows | Expected val accuracy |
|---------------|----------------------|
| ~40 rows | Overfits — unreliable |
| ~150 rows | 55–60% realistic |
| ~250+ rows | 60–65% achievable |

60% classification accuracy on financial time series is considered excellent.

---

## 13. Anti-Leakage Checklist

Before declaring the model production-ready, verify each item:

| Check | How to verify |
|-------|--------------|
| No future data in features | All lag features use `.shift(1)` before `.rolling()` — confirmed in `preprocess.py` |
| GARCH rolling window | `preprocess.py` fits on `data[:i]`, forecasts day `i` — no look-ahead |
| Train/val/test strictly time-ordered | Cutoff dates are hard calendar boundaries, no shuffle |
| SMOTE applied only on train | Never fit on val/test combined with train |
| Hyperparameter tuning on val only | Test set touched only in Phase 6 |
| `Realized_Vol_proxy` not in features | Check `FEATURE_COLS` list |
| Target not in `X_train` | `Target_Binary`, `Target_Regression`, `GARCH_Error` excluded from `FEATURE_COLS` |

```python
# Automated leakage check (run after training)
leakage_risk_cols = ["Target_Binary", "Target_Regression", "GARCH_Error",
                     "Realized_Vol_proxy", "log_return"]
for col in leakage_risk_cols:
    if col in FEATURE_COLS:
        raise ValueError(f"DATA LEAKAGE: {col} found in FEATURE_COLS!")
print("Leakage check passed.")
```

---

## 14. Common Pitfalls & Fixes

| Problem | Symptom | Fix |
|---------|---------|-----|
| Overfitting | Val AUC >> Test AUC | Reduce `max_depth` (try 3), increase `min_child_weight`, add `reg_alpha`/`reg_lambda` |
| Class imbalance | Model predicts all-0 or all-1 | Set `scale_pos_weight = sqrt(neg/pos)`, then sweep threshold on val set |
| Regime shift | Test AUC < 0.5 (inverted) | Check `HV_GARCH_ratio` in training vs test — if it changed sign, you have a regime shift. Add regime-aware features |
| Date parse crash | `TypeError: '<=' not supported` | Dataset2 files (Excel serial) mixed with Dataset1 (string) — `fix_date_column` handles this automatically if updated |
| Volume column missing | `KeyError: 'TRADED_QUA'` | Check NSE column name variant — add to `_VOL_CANDIDATES` list in `build_daily_features` |
| GARCH_Forecast NaN | NaN in feature table | Check `arch` installed; extend `GARCH_WARMUP` if needed |
| Platt calibration hurts | Probabilities compressed to ~0.2 | Remove calibration — val/test class distributions likely differ |
| `IV_MAX` too tight | Many contracts dropped | Raise to 3.0 if data covers high-vol periods (COVID, major events) |
| ATM_IV all NaN | Options signal absent | Verify IV computation ran; check moneyness filter isn't too tight |

---

## 15. Data Schema Reference

### `final_features.parquet`

Index: `Date` (datetime, trading days only)

| Column | Type | Description |
|--------|------|-------------|
| `ATM_IV` | float | At-the-money implied vol (annualized, decimal) |
| `ATM_IV_lag1` | float | 1-day lag |
| `ATM_IV_lag2` | float | 2-day lag |
| `ATM_IV_5d_mean` | float | 5-day rolling mean |
| `ATM_IV_trend` | float | ATM_IV - ATM_IV_5d_mean |
| `Vol_of_Vol` | float | 10-day std of ATM_IV |
| `Skew` | float | Put IV (0.95m) - Call IV (1.05m) |
| `Skew_lag1` | float | 1-day lag |
| `TS_Slope` | float | Far-expiry IV - near-expiry IV |
| `IV_HV_Spread` | float | ATM_IV - HV_20 |
| `Total_OI` | float | Total open interest |
| `OI_Change` | float | Daily OI delta |
| `OI_Change_lag1` | float | 1-day lag |
| `PCR_OI` | float | Put OI / Call OI |
| `PCR_OI_change1d` | float | 1-day PCR OI momentum |
| `Total_Volume` | float | Total traded volume |
| `Volume_Change` | float | Daily volume delta |
| `PCR_Volume` | float | Put volume / Call volume |
| `DTE_nearest` | int | Days to nearest expiry |
| `Is_expiry_week` | int | 1 if within 5 days of expiry |
| `Days_since_last_expiry` | int | Days since last expiry |
| `HV_10`, `HV_20`, `HV_30` | float | Historical vol (annualized, decimal) |
| `HV_GARCH_ratio` | float | HV_20 / GARCH_Forecast |
| `HV_GARCH_above_1` | int | Binary: ratio > 1.0 |
| `GARCH_Forecast` | float | GARCH one-day-ahead vol (decimal) |
| `GARCH_Residual_lag1` | float | Lagged standardized GARCH residual |
| `GARCH_Residual_lag2` | float | 2-day lag |
| `GARCH_Bias_short` | float | 5-day rolling mean of GARCH_Error |
| `GARCH_Bias_positive` | int | Binary: 5-day bias > 0 |
| `log_return` | float | Log return of underlying |
| `Realized_Vol_proxy` | float | 10-day HV as realized vol proxy |
| `GARCH_Error` | float | Realized_Vol_proxy - GARCH_Forecast |
| `Target_Binary` | int | 1 if GARCH underestimated, 0 otherwise |
| `Target_Regression` | float | Exact signed error (= GARCH_Error) |

### `test_set_forecasts.csv`

| Column | Description |
|--------|-------------|
| `GARCH_Forecast` | Raw GARCH one-day-ahead vol |
| `GARCH_Error` | Realized - GARCH forecast |
| `Realized_Vol_proxy` | Actual realized vol for that day |
| `Target_Binary` | Ground truth |
| `XGB_Predicted_Class` | Model prediction |
| `XGB_Pred_Probability` | Confidence for class 1 |
| `XGB_Correction` | Correction from regressor |
| `Final_Forecast` | GARCH + XGB correction |
| `Correct_Direction` | 1 if model got direction right |
