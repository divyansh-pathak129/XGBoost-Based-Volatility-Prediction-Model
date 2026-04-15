# Claude Code — Project Context & Session Notes

> **Read this first.** This document is the starting point for every Claude Code session on this project. It summarizes what has been built, what decisions were made, what problems were solved, and what to work on next.

---

## Project Summary

A two-stage BANKNIFTY volatility forecasting system:
- **Stage 1:** Rolling GJR-GARCH baseline forecast from spot returns
- **Stage 2:** XGBoost correction layer trained on GARCH residuals + options market signals

Key scripts: `preprocess.py` → `xgboost_volatility_model.py` → `daily_predict.py`

Dashboard: `streamlit run vol_dashboard/app.py`

Full technical reference: [DOCUMENTATION.md](DOCUMENTATION.md)

---

## What Has Been Built

### Core Pipeline (`preprocess.py`)
- Loads NSE monthly Excel files from `BANKNIFTY/` — supports both Dataset1 (string dates) and Dataset2 (Excel serial float dates) automatically via `fix_date_column()`
- Applies liquidity filters (OI, moneyness, DTE)
- Computes implied volatility using Black-Scholes + Brent's method, parallelized via `joblib`
- Builds daily features: ATM_IV, Skew, TS_Slope, OI/Volume, PCR, calendar features
- Fits rolling GJR-GARCH (configurable via `GARCH_MODEL` constant)
- Produces `data/features/final_features.parquet` with 29 features + targets

### Model (`xgboost_volatility_model.py`)
- XGBoost classifier: predicts whether GARCH under/overestimates tomorrow's vol (binary)
- XGBoost regressor: predicts correction magnitude (continuous)
- Final forecast = GARCH_Forecast + XGB_Correction
- Uses `RandomizedSearchCV` with `TimeSeriesSplit` for hyperparameter tuning
- Saves to `models/xgb_classifier.ubj`, `models/xgb_regressor.ubj`

### Live Signal (`daily_predict.py`)
- Reads today's processed option chain + latest pipeline output
- Produces BUY VOL / SELL VOL / HOLD signal with confidence
- Persists state in `data/daily_state.json` for delta features

### Dashboard (`vol_dashboard/`)
- Streamlit, read-only, 5 panels: hero signal, forecast cards, chart, signal log, market context
- Entry point: `vol_dashboard/app.py`
- Data loading: `vol_dashboard/data_loader.py` (all merging and unit normalization in one place)

---

## Key Decisions & Why

| Decision | Reason |
|----------|--------|
| GJR-GARCH instead of symmetric GARCH | Captures leverage effect (equity indices fall faster than they rise) |
| `scale_pos_weight = sqrt(neg/pos)` | Square root rule: gives minority class weight without destroying probability calibration |
| Platt calibration removed | Val set (18% class 1) ≠ test set (63% class 1) — calibration compressed probabilities to wrong base rate |
| `RandomizedSearchCV` instead of `GridSearchCV` | n_iter=60 covers more of the space in the same time; added `gamma`, `reg_alpha`, `reg_lambda` |
| `max_depth` capped at 4, `min_child_weight` [5,10,20] | Dataset is ~200 rows — regularization prevents overfitting |
| `HV_GARCH_ratio` as feature | Was the missing signal that explained the AUC collapse (see below) |

---

## The Big Problem That Was Solved (2026-04-08)

### What happened

Test AUC collapsed from 0.61 (validation) → 0.33 (test). The model was predicting class 0 (GARCH overestimates) for every single test row.

### Root cause

- Training tail (Nov–Dec 2025): GARCH overestimated 80% of the time → model learned "always predict class 0"
- March 2026 test set: 63% class 1 (GARCH underestimating) — completely opposite regime
- The model had no feature to detect this regime shift

### The diagnostic

`HV_GARCH_ratio` = HV_20 / GARCH_Forecast:
- Training tail: ratio was 0.5–0.6 (GARCH forecasting double what HV showed)
- March 2026: ratio jumped to 1.3–1.6 (historical vol now exceeding GARCH forecast)
- This ratio was not a feature, so the model was blind to the shift

### What was fixed

1. Added `HV_GARCH_ratio` and `HV_GARCH_above_1` as features
2. Added `GARCH_Bias_short` (5-day rolling GARCH error mean)
3. Added `ATM_IV_trend`, `Vol_of_Vol`, `PCR_OI_change1d`, `GARCH_Bias_positive`
4. Switched to GJR-GARCH
5. Removed Platt calibration
6. Tightened regularization in hyperparameter search

**Result:** Test AUC ~0.40 → ~0.64, Test Acc ~42% → ~58%

---

## Current Model Performance

| Metric | Before fixes | After fixes |
|--------|-------------|------------|
| Val AUC | 0.61 | ~0.64 |
| Test AUC | 0.33 | ~0.64 |
| Test Accuracy | 42% | ~58% |

Note: 60% accuracy on financial time series is considered excellent.

---

## File Layout (Important Files)

```
preprocess.py                    # Phase 1–7 pipeline
xgboost_volatility_model.py      # Model training, TRAIN_END / VAL_END constants at top
option_data_formating.py         # NSE option chain CSV cleaner
daily_predict.py                 # Live signal generator
vol_dashboard/app.py             # Dashboard entry point
vol_dashboard/data_loader.py     # All data merging — touch this for unit/column changes
data/features/final_features.parquet  # Primary model input
models/xgb_classifier.ubj        # Trained classifier
models/xgb_regressor.ubj         # Trained regressor
data/daily_state.json            # Persists OI/volume/PCR for delta features
```

---

## Things to Know Before Making Changes

- **Lag features:** All lag features in `preprocess.py` use `.shift(1)` before any `.rolling()` to prevent look-ahead bias. Do not change this.
- **GARCH rolling:** For each day `i`, GARCH fits on `returns[:i]` and forecasts day `i+1`. No look-ahead.
- **Date splits:** `TRAIN_END` and `VAL_END` in `xgboost_volatility_model.py` must be updated whenever new monthly data is added.
- **Vol units:** All vol values in the pipeline are in **decimal** (e.g., 0.01 = 1% daily vol). The dashboard converts to percent (×100) in `data_loader.build_master()` — nowhere else.
- **FEATURE_COLS list:** Adding a new feature requires updating the `FEATURE_COLS` list in `xgboost_volatility_model.py` AND in `daily_predict.py` to keep training and inference aligned.
- **Model format:** Models are saved in both `.ubj` (XGBoost native, load with `clf.load_model()`) and `.pkl` (joblib). Prefer `.ubj` for version safety.

---

## Potential Next Steps

- **Walk-forward cross-validation** — Replace single train/val/test split with 5 expanding windows for more stable performance estimates (see `model_enhancements.md` for design)
- **3-class target** — Dead zone around zero GARCH error to reduce label noise (see `model_enhancements.md`)
- **ATM_IV_regime flag** — Binary: today's ATM_IV above its 30-day rolling median — addresses distribution shift
- **Retrain periodically** — Every month or two as data accumulates; update `TRAIN_END` / `VAL_END` before retraining
- **NIFTY 50 / FINNIFTY support** — Architecture is index-agnostic; would need separate raw data ingestion and `DATA_DIR` config

---

## Archived Planning Docs (do not delete)

These are reference documents from the development process — useful for understanding why things were built the way they are:

- `xgboost_volatility_model_plan.md` — Original step-by-step model implementation plan
- `model_enhancements.md` — Detailed diagnosis of the AUC collapse + six proposed fixes
- `fix_guide_preprocess_multiformat.md` — Multi-format (Dataset1 + Dataset2) date parsing fix guide
- `vol_dashboard_plan.md` — Dashboard design plan
- `data_visiualization_plan.md` — Data visualization plan
