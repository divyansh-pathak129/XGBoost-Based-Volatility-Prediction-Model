# XGBoost Volatility Prediction Model — Step-by-Step Execution Plan

**Dataset:** `data/features/final_features.parquet`  
**Goal:** Predict whether GARCH underestimates next-day realized volatility (binary classification), then produce a corrected volatility forecast via `GARCH_Forecast + XGBoost_Correction`.

---

## Overview of the Two-Stage Architecture

```
Stage 1 (Done in pipeline):  Spot Returns → GARCH(1,1) → GARCH_Forecast
Stage 2 (This plan):         GARCH_Forecast + Options Signals → XGBoost → Correction
                              ↓
                         Final_Vol_Forecast = GARCH_Forecast ± XGBoost_Correction
```

The pipeline in `preprocess.py` has already produced:
- `GARCH_Forecast` — one-day-ahead volatility forecast from rolling GARCH(1,1)
- `GARCH_Error` — realized vol proxy minus GARCH forecast (continuous target)
- `Target_Binary` — 1 if GARCH underestimated, 0 if overestimated
- ~23 options/microstructure/calendar features with lags

---

## Phase 0: Environment Setup

### Step 0.1 — Install dependencies

```bash
pip install xgboost scikit-learn shap matplotlib seaborn pandas numpy pyarrow imbalanced-learn
```

### Step 0.2 — Verify dataset is ready

```python
import pandas as pd

df = pd.read_parquet("data/features/final_features.parquet")
print(df.shape)
print(df.index.min(), "→", df.index.max())
print(df.columns.tolist())
print(df[["Target_Binary", "GARCH_Forecast", "GARCH_Error"]].describe())
print(df["Target_Binary"].value_counts(normalize=True))  # check class balance
```

**Expected:** ~180–250 rows, date range Apr 2025 – Feb 2026, 25–30 columns.

---

## Phase 1: Exploratory Data Analysis (EDA)

### Step 1.1 — Load and inspect

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_parquet("data/features/final_features.parquet")
df = df.sort_index()  # ensure chronological order

# Basic info
print(df.dtypes)
print(df.isnull().sum().sort_values(ascending=False))
```

### Step 1.2 — Plot key signals over time

```python
fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)

df["ATM_IV"].plot(ax=axes[0], title="ATM IV Over Time")
df["GARCH_Forecast"].plot(ax=axes[1], title="GARCH Forecast vs Realized Vol Proxy", label="GARCH Forecast")
df["Realized_Vol_proxy"].plot(ax=axes[1], label="Realized Vol Proxy", alpha=0.6)
axes[1].legend()
df["Skew"].plot(ax=axes[2], title="IV Skew (Put - Call)")
df["GARCH_Error"].plot(ax=axes[3], title="GARCH Error (Target - Regression)")
axes[3].axhline(0, color="red", linestyle="--")

plt.tight_layout()
plt.savefig("outputs/eda_signals.png", dpi=150)
plt.show()
```

### Step 1.3 — Class balance check

```python
balance = df["Target_Binary"].value_counts()
print(balance)
print(f"\nClass ratio: {balance[1]/balance[0]:.2f}")
# If ratio > 1.5 or < 0.67 → use scale_pos_weight in XGBoost
```

### Step 1.4 — Correlation heatmap

```python
FEATURE_COLS = [
    "GARCH_Forecast", "GARCH_Residual_lag1", "GARCH_Residual_lag2",
    "ATM_IV", "ATM_IV_lag1", "ATM_IV_lag2", "ATM_IV_5d_mean",
    "Skew", "Skew_lag1", "TS_Slope", "IV_HV_Spread",
    "OI_Change", "OI_Change_lag1", "PCR_OI", "Volume_Change", "PCR_Volume",
    "DTE_nearest", "Is_expiry_week", "Days_since_last_expiry",
    "HV_10", "HV_20", "HV_30",
]
# Keep only columns that exist in df
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]

corr = df[FEATURE_COLS + ["Target_Binary"]].corr()
plt.figure(figsize=(16, 12))
sns.heatmap(corr, annot=False, cmap="RdBu_r", center=0)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png", dpi=150)
plt.show()

# Top features correlated with target
print(corr["Target_Binary"].sort_values(ascending=False).head(15))
```

---

## Phase 2: Prepare Data for Modeling

### Step 2.1 — Drop NaN rows

```python
# Drop rows where any feature or target is NaN
model_df = df[FEATURE_COLS + ["Target_Binary", "Target_Regression", "GARCH_Forecast",
                               "GARCH_Error", "Realized_Vol_proxy", "log_return"]].copy()
before = len(model_df)
model_df = model_df.dropna(subset=FEATURE_COLS + ["Target_Binary"])
print(f"Dropped {before - len(model_df)} rows with NaN → {len(model_df)} usable rows")
print(f"Date range: {model_df.index.min()} → {model_df.index.max()}")
```

### Step 2.2 — Time-based train/validation/test split

> **Critical:** Never use random splits for time series — it causes data leakage.

```python
# Adjust cutoff dates based on your actual date range
TRAIN_END    = "2025-10-31"   # Training: Apr 2025 – Oct 2025 (~150 days)
VAL_END      = "2026-01-31"   # Validation: Nov 2025 – Jan 2026 (~60 days)
# Test: Feb 2026 → end of dataset (~40 days)

train_df = model_df[model_df.index <= TRAIN_END]
val_df   = model_df[(model_df.index > TRAIN_END) & (model_df.index <= VAL_END)]
test_df  = model_df[model_df.index > VAL_END]

print(f"Train: {len(train_df)} rows | {train_df.index.min()} → {train_df.index.max()}")
print(f"Val:   {len(val_df)} rows | {val_df.index.min()} → {val_df.index.max()}")
print(f"Test:  {len(test_df)} rows | {test_df.index.min()} → {test_df.index.max()}")

X_train = train_df[FEATURE_COLS]
y_train = train_df["Target_Binary"]

X_val = val_df[FEATURE_COLS]
y_val = val_df["Target_Binary"]

X_test = test_df[FEATURE_COLS]
y_test = test_df["Target_Binary"]
```

### Step 2.3 — Handle class imbalance (if needed)

```python
from collections import Counter

ratio = Counter(y_train)
print(f"Train class distribution: {ratio}")
neg, pos = ratio[0], ratio[1]
scale_pos_weight = neg / pos   # pass to XGBoost
print(f"scale_pos_weight = {scale_pos_weight:.2f}")

# Option A: Use scale_pos_weight in XGBoost (preferred for time series)
# Option B: SMOTE — only on training data, never on val/test
# from imblearn.over_sampling import SMOTE
# sm = SMOTE(random_state=42)
# X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
```

---

## Phase 3: Baseline XGBoost Model

### Step 3.1 — Fit baseline model

```python
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix

baseline_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1,
)

baseline_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

# Evaluate on validation set
y_val_pred  = baseline_model.predict(X_val)
y_val_proba = baseline_model.predict_proba(X_val)[:, 1]

print("=== Validation Set Results (Baseline) ===")
print(f"Accuracy:  {accuracy_score(y_val, y_val_pred):.4f}")
print(f"AUC-ROC:   {roc_auc_score(y_val, y_val_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))
```

**Target baseline:** ≥55% accuracy, AUC-ROC ≥ 0.55 on validation set.

### Step 3.2 — Feature importance check

```python
import pandas as pd

importances = pd.Series(
    baseline_model.feature_importances_,
    index=FEATURE_COLS
).sort_values(ascending=False)

print("\nTop 15 Feature Importances:")
print(importances.head(15).to_string())

# GARCH_Forecast should be in top 10 — if not, investigate data leakage or stale features
assert "GARCH_Forecast" in importances.head(15).index, \
    "WARNING: GARCH_Forecast not in top 15 — check for issues"

importances.head(20).plot(kind="barh", figsize=(10, 8), title="XGBoost Feature Importances")
plt.tight_layout()
plt.savefig("outputs/feature_importances_baseline.png", dpi=150)
plt.show()
```

---

## Phase 4: Hyperparameter Tuning

### Step 4.1 — TimeSeriesSplit cross-validation

> Use `TimeSeriesSplit` (not standard K-Fold) to respect temporal ordering.

```python
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

tscv = TimeSeriesSplit(n_splits=5)

# Grid search parameters
param_grid = {
    "max_depth":        [3, 4, 5, 6],
    "learning_rate":    [0.01, 0.05, 0.1],
    "n_estimators":     [100, 200, 300],
    "subsample":        [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
}

# Lighter search to start — expand if compute allows
param_grid_light = {
    "max_depth":     [3, 4, 5],
    "learning_rate": [0.05, 0.1],
    "n_estimators":  [100, 200, 300],
    "subsample":     [0.8, 1.0],
}

base_xgb = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    use_label_encoder=False,
    random_state=42,
    n_jobs=1,  # keep 1 here since GridSearchCV parallelizes across folds
)

# Combine train+val for CV (split controls leakage)
X_trainval = pd.concat([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])

grid_search = GridSearchCV(
    base_xgb,
    param_grid_light,
    cv=tscv,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=1,
    refit=True,
)
grid_search.fit(X_trainval, y_trainval)

print(f"\nBest params: {grid_search.best_params_}")
print(f"Best CV AUC: {grid_search.best_score_:.4f}")
```

### Step 4.2 — Retrain best model with early stopping

```python
best_params = grid_search.best_params_

tuned_model = xgb.XGBClassifier(
    **best_params,
    scale_pos_weight=scale_pos_weight,
    n_estimators=500,        # high ceiling; early stopping will find optimum
    eval_metric="logloss",
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=30,
)

tuned_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

print(f"Best iteration: {tuned_model.best_iteration}")

y_val_pred_tuned  = tuned_model.predict(X_val)
y_val_proba_tuned = tuned_model.predict_proba(X_val)[:, 1]

print("\n=== Validation Set Results (Tuned) ===")
print(f"Accuracy:  {accuracy_score(y_val, y_val_pred_tuned):.4f}")
print(f"AUC-ROC:   {roc_auc_score(y_val, y_val_proba_tuned):.4f}")
print(classification_report(y_val, y_val_pred_tuned))
```

---

## Phase 5: SHAP Value Analysis

### Step 5.1 — Compute and plot SHAP values

```python
import shap

explainer = shap.TreeExplainer(tuned_model)
shap_values = explainer.shap_values(X_val)

# Summary plot — shows global feature importance + directionality
plt.figure()
shap.summary_plot(shap_values, X_val, plot_type="bar", show=False)
plt.title("SHAP Feature Importance (Validation Set)")
plt.tight_layout()
plt.savefig("outputs/shap_importance.png", dpi=150, bbox_inches="tight")
plt.show()

# Beeswarm — shows direction of effect
plt.figure()
shap.summary_plot(shap_values, X_val, show=False)
plt.title("SHAP Beeswarm (Validation Set)")
plt.tight_layout()
plt.savefig("outputs/shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.show()
```

### Step 5.2 — Validate SHAP makes financial sense

Expected sensible behavior:
- **ATM_IV high** → SHAP positive (more likely GARCH underestimates; IV signals more vol)
- **Skew high** (puts expensive) → SHAP positive (put demand signals tail risk, vol underpriced)
- **GARCH_Forecast high** → SHAP varies (already a high baseline, may mean revert)
- **DTE_nearest low** (near expiry) → SHAP positive (gamma/pin risk increases realized vol)
- **HV features** → inversely related to error (high HV already in GARCH)

```python
# Check mean absolute SHAP for top features
shap_df = pd.DataFrame(np.abs(shap_values), columns=FEATURE_COLS)
top_shap = shap_df.mean().sort_values(ascending=False)
print("Top 10 features by mean |SHAP|:")
print(top_shap.head(10).to_string())
```

---

## Phase 6: Final Test Set Evaluation

> **Only run this once** — after all tuning decisions are finalized. Do not tune further based on test set results.

### Step 6.1 — Evaluate on held-out test set

```python
y_test_pred  = tuned_model.predict(X_test)
y_test_proba = tuned_model.predict_proba(X_test)[:, 1]

print("=" * 50)
print("=== TEST SET RESULTS (Final, Feb–Mar 2026) ===")
print("=" * 50)
print(f"Accuracy:  {accuracy_score(y_test, y_test_pred):.4f}")
print(f"AUC-ROC:   {roc_auc_score(y_test, y_test_proba):.4f}")
print(classification_report(y_test, y_test_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))
```

**Success criteria (from blueprint):**
| Metric | Target |
|--------|--------|
| Accuracy | 56–62% |
| AUC-ROC | ≥ 0.58 |
| Better direction accuracy than raw GARCH | Yes |

---

## Phase 7: Combined Volatility Forecast

### Step 7.1 — Regression model for continuous correction

```python
# Train regression XGBoost to predict the exact GARCH error magnitude
y_train_reg = train_df["Target_Regression"]
y_val_reg   = val_df["Target_Regression"]
y_test_reg  = test_df["Target_Regression"]

reg_model = xgb.XGBRegressor(
    **best_params,
    n_estimators=tuned_model.best_iteration,
    scale_pos_weight=1,
    eval_metric="rmse",
    random_state=42,
    n_jobs=-1,
)

reg_model.fit(
    X_train, y_train_reg,
    eval_set=[(X_val, y_val_reg)],
    verbose=False,
)
```

### Step 7.2 — Compute corrected forecast

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Predict correction on test set
xgb_correction = reg_model.predict(X_test)

# Combine with GARCH
final_forecast = test_df["GARCH_Forecast"] + xgb_correction
realized        = test_df["Realized_Vol_proxy"]
garch_only      = test_df["GARCH_Forecast"]

# Metrics
mae_combined = mean_absolute_error(realized, final_forecast)
mae_garch    = mean_absolute_error(realized, garch_only)
rmse_combined = np.sqrt(mean_squared_error(realized, final_forecast))
rmse_garch    = np.sqrt(mean_squared_error(realized, garch_only))

print("=== Forecast Comparison (Test Set) ===")
print(f"GARCH only  — MAE: {mae_garch:.6f}  RMSE: {rmse_garch:.6f}")
print(f"GARCH+XGBoost — MAE: {mae_combined:.6f}  RMSE: {rmse_combined:.6f}")
print(f"Improvement: MAE {(mae_garch - mae_combined)/mae_garch*100:.1f}%  "
      f"RMSE {(rmse_garch - rmse_combined)/rmse_garch*100:.1f}%")
```

### Step 7.3 — Direction accuracy

```python
# Direction accuracy: does the combined forecast correct the GARCH error direction?
garch_direction_correct    = ((garch_only > realized) == (garch_only > realized)).sum()
# Proper: is the combined forecast closer to realized than GARCH alone?
combined_better = (np.abs(final_forecast - realized) < np.abs(garch_only - realized)).mean()
print(f"\nTest rows where combined forecast is closer to realized: {combined_better:.1%}")
```

### Step 7.4 — Visualization

```python
plt.figure(figsize=(14, 5))
plt.plot(realized.index, realized.values, label="Realized Vol Proxy", color="black", linewidth=1.5)
plt.plot(garch_only.index, garch_only.values, label="GARCH Forecast", color="blue", linestyle="--", alpha=0.8)
plt.plot(final_forecast.index, final_forecast.values, label="GARCH+XGBoost", color="orange", linewidth=1.5)
plt.title("Test Set: GARCH vs GARCH+XGBoost vs Realized Volatility")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/forecast_comparison.png", dpi=150)
plt.show()
```

---

## Phase 8: Save Model and Outputs

### Step 8.1 — Save models

```python
import joblib
import os

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

joblib.dump(tuned_model, "models/xgb_classifier.pkl")
joblib.dump(reg_model, "models/xgb_regressor.pkl")
print("Models saved to models/")

# Also save in XGBoost native format (faster load, version-safe)
tuned_model.save_model("models/xgb_classifier.ubj")
reg_model.save_model("models/xgb_regressor.ubj")
```

### Step 8.2 — Save forecast outputs

```python
results = test_df[["GARCH_Forecast", "GARCH_Error", "Realized_Vol_proxy",
                    "Target_Binary"]].copy()
results["XGB_Predicted_Class"]   = y_test_pred
results["XGB_Pred_Probability"]  = y_test_proba
results["XGB_Correction"]        = xgb_correction
results["Final_Forecast"]        = final_forecast.values

results.to_csv("outputs/test_set_forecasts.csv")
print(results.tail(10).to_string())
```

---

## Phase 9: Anti-Leakage Checklist

Before declaring the model production-ready, verify each item:

| Check | How to verify |
|-------|--------------|
| No future data in features | All lag features shift forward (`lag1` = yesterday's value); rolling features use `.shift(1)` before `.rolling()` — confirmed in `preprocess.py` |
| GARCH rolling window | `preprocess.py` fits on `data[:i]`, forecasts day `i` — confirmed no look-ahead |
| Train/val/test are strictly time-ordered | Cutoff dates are hard calendar boundaries, no shuffle |
| SMOTE applied only on train | Never fit SMOTE on val/test combined with train |
| Hyperparameter tuning used val set only | Test set only touched in Phase 6 |
| `Realized_Vol_proxy` not in feature set | Check `FEATURE_COLS` — it must not appear there |
| Target not in `X_train` | `Target_Binary` and `Target_Regression` excluded from `FEATURE_COLS` |

```python
# Automated leakage check
leakage_risk_cols = ["Target_Binary", "Target_Regression", "GARCH_Error",
                     "Realized_Vol_proxy", "log_return"]
for col in leakage_risk_cols:
    if col in FEATURE_COLS:
        raise ValueError(f"DATA LEAKAGE: {col} found in FEATURE_COLS!")
print("Leakage check passed.")
```

---

## Phase 10: Common Pitfalls and Fixes

| Problem | Symptom | Fix |
|---------|---------|-----|
| Overfitting | Val accuracy >> Test accuracy | Reduce `max_depth` (try 3), increase `min_child_weight`, add `reg_alpha`/`reg_lambda` |
| Class imbalance ignored | Model predicts all-0 or all-1 | Set `scale_pos_weight = neg/pos`, or use SMOTE on train only |
| GARCH_Forecast missing | Feature NaN everywhere | Check `preprocess.py` ran with `arch` installed; warmup period may need extending |
| Skew/TS_Slope NaN heavy | >20% rows have NaN | Forward-fill with `df[col].fillna(method='ffill')` only on train, then propagate |
| Validation accuracy <50% | Worse than random | Check target variable — `Target_Binary` flipped? Verify `GARCH_Error = Realized - Forecast`, not the reverse |
| SHAP magnitudes all equal | No dominant features | Probably too few trees or too shallow; increase `n_estimators`, check data has signal |
| ATM_IV not top SHAP feature | Options signal absent | Verify IV computation succeeded in pipeline; check ATM_IV NaN count |

---

## Recommended File Structure

```
stockspipeline/
├── preprocess.py                  ← pipeline (Phases 1–7 done)
├── xgboost_volatility_model.py    ← model script (implement from this plan)
├── data/
│   ├── raw/master_raw.parquet
│   ├── processed/
│   │   ├── master_filtered.parquet
│   │   ├── master_with_iv.parquet
│   │   └── daily_pre_lags.parquet
│   └── features/
│       ├── final_features.parquet  ← input to this plan
│       └── final_features.csv
├── models/
│   ├── xgb_classifier.ubj
│   └── xgb_regressor.ubj
└── outputs/
    ├── eda_signals.png
    ├── correlation_heatmap.png
    ├── feature_importances_baseline.png
    ├── shap_importance.png
    ├── shap_beeswarm.png
    ├── forecast_comparison.png
    └── test_set_forecasts.csv
```

---

## Execution Order Summary

```
Step 0.1  Install deps
Step 0.2  Verify dataset
Step 1.1  Load & inspect
Step 1.2  Plot signals
Step 1.3  Class balance
Step 1.4  Correlation heatmap
Step 2.1  Drop NaN rows
Step 2.2  Train/val/test split (time-based)
Step 2.3  Handle class imbalance → compute scale_pos_weight
Step 3.1  Fit baseline XGBoost
Step 3.2  Check feature importances
Step 4.1  TimeSeriesSplit grid search
Step 4.2  Retrain best model with early stopping
Step 5.1  SHAP summary plots
Step 5.2  Validate SHAP financial logic
Step 6.1  Evaluate on test set (ONCE, no further tuning)
Step 7.1  Train regression XGBoost on GARCH_Error
Step 7.2  Combine GARCH + XGBoost correction
Step 7.3  Direction accuracy
Step 7.4  Plot forecast comparison
Step 8.1  Save models
Step 8.2  Save forecast CSV
Step 9    Leakage checklist
```

---

## Success Criteria

| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| Test Accuracy (classification) | >55% | >58% | >60% |
| Test AUC-ROC | >0.55 | >0.60 | >0.65 |
| GARCH+XGBoost MAE vs GARCH-only | Any improvement | >5% better | >10% better |
| SHAP dominant feature | ATM_IV or Skew in top 5 | GARCH_Forecast in top 3 | All top features financially intuitive |
| No look-ahead bias | Leakage check passes | — | — |

> **Note:** 60% accuracy in financial time series prediction is considered excellent. Do not tune aggressively to chase higher numbers — that path leads to overfitting on a small (~200 row) dataset.
