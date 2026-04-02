"""
XGBoost Volatility Prediction Model
BANKNIFTY Options -- GARCH Residual Correction via XGBoost
Input:  data/features/final_features.parquet  (produced by preprocess.py)
Output: models/xgb_classifier.ubj, models/xgb_regressor.ubj
        outputs/test_set_forecasts.csv, outputs/*.png

Enhancements applied (see model_enhancements.md):
  Fix A -- scale_pos_weight = sqrt(neg/pos); decision threshold tuned on val set
  Fix B -- Replace HV_10, HV_30 with HV_ratio = HV_10 / HV_30
  Fix C -- Add ATM_IV_regime binary flag (above/below 30-day rolling median)
  Fix D -- Add GARCH_Bias_rolling (20-day rolling mean of past GARCH errors)
  Fix E -- Dead-zone target: drop rows where |GARCH_Error| < 0.003 from train/val
  Fix F -- Walk-forward CV with 5 expanding windows for hyperparameter tuning
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

warnings.filterwarnings("ignore")

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# --------------------------------------------------------------
# FEATURE COLUMNS
# Fix B: HV_10 and HV_30 removed; HV_ratio added
# Fix C: ATM_IV_regime added
# Fix D: GARCH_Bias_rolling added
# --------------------------------------------------------------

FEATURE_COLS = [
    "GARCH_Forecast", "GARCH_Residual_lag1", "GARCH_Residual_lag2",
    "ATM_IV", "ATM_IV_lag1", "ATM_IV_lag2", "ATM_IV_5d_mean",
    "Skew", "Skew_lag1", "TS_Slope", "IV_HV_Spread",
    "OI_Change", "OI_Change_lag1", "PCR_OI", "Volume_Change", "PCR_Volume",
    "DTE_nearest", "Is_expiry_week", "Days_since_last_expiry",
    "HV_20", "HV_ratio",          # Fix B: HV_10/HV_30 replaced by HV_20 + ratio
    "ATM_IV_regime",               # Fix C
    "GARCH_Bias_rolling",          # Fix D
]

# Dead-zone threshold (Fix E)
DEAD_ZONE = 0.003

# Time-based split dates
TRAIN_END = "2025-12-31"
VAL_END   = "2026-02-28"


# ==============================================================
# PHASE 0 -- LOAD & VALIDATE DATASET
# ==============================================================

def load_data():
    print("=" * 60)
    print("PHASE 0: Loading dataset")
    print("=" * 60)

    df = pd.read_parquet("data/features/final_features.parquet")
    df = df.sort_index()

    # ----------------------------------------------------------
    # Fix B: HV_ratio = HV_10 / HV_30
    # Short-term vol elevated vs long-term => compression signal
    # ----------------------------------------------------------
    if "HV_10" in df.columns and "HV_30" in df.columns:
        df["HV_ratio"] = df["HV_10"] / df["HV_30"].replace(0, np.nan)
        df["HV_ratio"] = df["HV_ratio"].fillna(1.0)

    # ----------------------------------------------------------
    # Fix C: ATM_IV_regime = 1 if ATM_IV > its 30-day rolling median
    # Tells XGBoost which vol environment it is currently in
    # ----------------------------------------------------------
    if "ATM_IV" in df.columns:
        rolling_med = df["ATM_IV"].rolling(30, min_periods=10).median()
        df["ATM_IV_regime"] = (df["ATM_IV"] > rolling_med).astype(int)

    # ----------------------------------------------------------
    # Fix D: GARCH_Bias_rolling = 20-day rolling mean of PAST GARCH errors
    # Shift by 1 to avoid leaking today's error into today's feature
    # ----------------------------------------------------------
    if "GARCH_Error" in df.columns:
        df["GARCH_Bias_rolling"] = (
            df["GARCH_Error"].shift(1)
                             .rolling(20, min_periods=5)
                             .mean()
        )

    # Restrict FEATURE_COLS to what is actually present
    global FEATURE_COLS
    FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]

    missing_orig = [c for c in [
        "GARCH_Forecast", "ATM_IV", "Skew", "HV_20",
        "OI_Change", "PCR_OI", "Target_Binary",
    ] if c not in df.columns]
    if missing_orig:
        print(f"  WARNING -- missing base columns: {missing_orig}")

    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df.index.min().date()} -> {df.index.max().date()}")
    print(f"  Features used: {len(FEATURE_COLS)}")
    print(f"  NaN in features: {df[FEATURE_COLS].isnull().sum().sum()}")

    balance = df["Target_Binary"].value_counts(normalize=True)
    print(f"\n  Class balance (full dataset):")
    print(f"    Class 0 (GARCH overestimated): {balance.get(0, 0):.1%}")
    print(f"    Class 1 (GARCH underestimated): {balance.get(1, 0):.1%}")

    return df


# ==============================================================
# PHASE 1 -- EDA
# ==============================================================

def run_eda(df):
    print("\n" + "=" * 60)
    print("PHASE 1: Exploratory Data Analysis")
    print("=" * 60)

    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)

    df["ATM_IV"].plot(ax=axes[0], color="steelblue", linewidth=1.2)
    axes[0].set_title("ATM Implied Volatility Over Time")
    axes[0].set_ylabel("IV (annualized)")

    if "GARCH_Forecast" in df.columns and "Realized_Vol_proxy" in df.columns:
        df["GARCH_Forecast"].plot(ax=axes[1], label="GARCH Forecast",
                                   color="navy", linewidth=1.2)
        df["Realized_Vol_proxy"].plot(ax=axes[1], label="Realized Vol Proxy",
                                      color="tomato", linewidth=1.0, alpha=0.7)
    axes[1].set_title("GARCH Forecast vs Realized Volatility")
    axes[1].set_ylabel("Volatility")
    axes[1].legend()

    df["Skew"].plot(ax=axes[2], color="purple", linewidth=1.2)
    axes[2].set_title("IV Skew (Put IV - Call IV)")
    axes[2].axhline(0, color="black", linestyle="--", linewidth=0.7)
    axes[2].set_ylabel("Skew")

    if "GARCH_Error" in df.columns:
        df["GARCH_Error"].plot(ax=axes[3], color="darkorange", linewidth=1.2)
        axes[3].axhline(0,            color="red",  linestyle="--", linewidth=0.8)
        axes[3].axhline( DEAD_ZONE,   color="gray", linestyle=":",  linewidth=0.8,
                         label=f"+{DEAD_ZONE} dead-zone")
        axes[3].axhline(-DEAD_ZONE,   color="gray", linestyle=":",  linewidth=0.8,
                         label=f"-{DEAD_ZONE} dead-zone")
    axes[3].set_title("GARCH Error (dead-zone band shown in gray)")
    axes[3].set_ylabel("Error")
    axes[3].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("outputs/eda_signals.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/eda_signals.png")

    # Class balance bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["Target_Binary"].value_counts()
    ax.bar(["Class 0\n(GARCH overestimated)", "Class 1\n(GARCH underestimated)"],
           [counts.get(0, 0), counts.get(1, 0)],
           color=["steelblue", "tomato"], edgecolor="white")
    for i, v in enumerate([counts.get(0, 0), counts.get(1, 0)]):
        ax.text(i, v + 0.5, str(v), ha="center", fontsize=11, fontweight="bold")
    ax.set_title("Class Distribution (Target_Binary)")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig("outputs/class_balance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/class_balance.png")

    # Correlation heatmap
    corr_cols = FEATURE_COLS + ["Target_Binary"]
    corr_cols = [c for c in corr_cols if c in df.columns]
    corr = df[corr_cols].corr()
    plt.figure(figsize=(16, 13))
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, linewidths=0.4, annot_kws={"size": 7},
                cbar_kws={"shrink": 0.8})
    plt.title("Feature Correlation Matrix", fontsize=13)
    plt.xticks(fontsize=7, rotation=45, ha="right")
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig("outputs/correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/correlation_heatmap.png")

    target_corr = corr["Target_Binary"].drop("Target_Binary").sort_values(
        key=abs, ascending=False)
    print(f"\n  Top 10 features by |corr| with Target_Binary:")
    print(target_corr.head(10).to_string())


# ==============================================================
# PHASE 2 -- PREPARE DATA
# ==============================================================

def prepare_data(df):
    print("\n" + "=" * 60)
    print("PHASE 2: Preparing Data")
    print("=" * 60)

    extra_cols = ["Target_Binary", "Target_Regression",
                  "GARCH_Error", "Realized_Vol_proxy", "log_return"]
    needed_cols = list(dict.fromkeys(FEATURE_COLS + extra_cols))
    model_df = df[[c for c in needed_cols if c in df.columns]].copy()

    before = len(model_df)
    model_df = model_df.dropna(subset=FEATURE_COLS + ["Target_Binary"])
    print(f"  Dropped {before - len(model_df)} NaN rows -> {len(model_df)} total rows")

    # Time-based split
    train_val_df = model_df[model_df.index <= VAL_END]
    test_df      = model_df[model_df.index > VAL_END]
    train_df_raw = model_df[model_df.index <= TRAIN_END]
    val_df_raw   = model_df[(model_df.index > TRAIN_END) & (model_df.index <= VAL_END)]

    # ----------------------------------------------------------
    # Fix E: Dead-zone filter -- drop ambiguous rows from train & val
    # Keep test set intact so evaluation is realistic
    # ----------------------------------------------------------
    if "GARCH_Error" in train_df_raw.columns:
        n_before_train = len(train_df_raw)
        n_before_val   = len(val_df_raw)
        train_df = train_df_raw[train_df_raw["GARCH_Error"].abs() >= DEAD_ZONE]
        val_df   = val_df_raw[val_df_raw["GARCH_Error"].abs() >= DEAD_ZONE]
        print(f"\n  Fix E -- Dead-zone filter (|GARCH_Error| < {DEAD_ZONE}):")
        print(f"    Train: {n_before_train} -> {len(train_df)} rows "
              f"(removed {n_before_train - len(train_df)})")
        print(f"    Val:   {n_before_val} -> {len(val_df)} rows "
              f"(removed {n_before_val - len(val_df)})")
        print(f"    Test:  {len(test_df)} rows (kept all -- realistic evaluation)")
    else:
        train_df = train_df_raw
        val_df   = val_df_raw

    print(f"\n  Train:  {len(train_df)} rows  "
          f"[{train_df.index.min().date()} -> {train_df.index.max().date()}]")
    print(f"  Val:    {len(val_df)} rows  "
          f"[{val_df.index.min().date()} -> {val_df.index.max().date()}]")
    print(f"  Test:   {len(test_df)} rows  "
          f"[{test_df.index.min().date()} -> {test_df.index.max().date()}]")

    X_train = train_df[FEATURE_COLS].reset_index(drop=True)
    y_train = train_df["Target_Binary"].reset_index(drop=True)
    X_val   = val_df[FEATURE_COLS].reset_index(drop=True)
    y_val   = val_df["Target_Binary"].reset_index(drop=True)
    X_test  = test_df[FEATURE_COLS].reset_index(drop=True)
    y_test  = test_df["Target_Binary"].reset_index(drop=True)

    # ----------------------------------------------------------
    # Fix A: scale_pos_weight = sqrt(neg / pos)
    # Square-root rule -- enough weight for minority class without
    # destroying probability calibration
    # ----------------------------------------------------------
    counts   = Counter(y_train)
    neg, pos = counts[0], counts.get(1, 1)
    scale_pos_weight = np.sqrt(neg / pos)
    print(f"\n  Train class distribution: {dict(counts)}")
    print(f"  Fix A -- scale_pos_weight = sqrt({neg}/{pos}) = {scale_pos_weight:.3f}")

    # Leakage check
    leakage_risk = ["Target_Binary", "Target_Regression", "GARCH_Error",
                    "Realized_Vol_proxy", "log_return"]
    for col in leakage_risk:
        if col in FEATURE_COLS:
            raise ValueError(f"DATA LEAKAGE: {col} found in FEATURE_COLS!")
    print("  Leakage check: PASSED")

    return (X_train, y_train, X_val, y_val, X_test, y_test,
            train_df, val_df, test_df, scale_pos_weight)


# ==============================================================
# PHASE 3 -- BASELINE MODEL
# ==============================================================

def train_baseline(X_train, y_train, X_val, y_val, scale_pos_weight):
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix

    print("\n" + "=" * 60)
    print("PHASE 3: Baseline XGBoost Model")
    print("=" * 60)

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_proba = model.predict_proba(X_val)[:, 1]
    y_pred  = model.predict(X_val)

    val_auc = roc_auc_score(y_val, y_proba)
    print(f"\n  Validation Accuracy:  {accuracy_score(y_val, y_pred):.4f}")
    print(f"  Validation AUC-ROC:   {val_auc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_val, y_pred,
                                 target_names=["GARCH Over (0)", "GARCH Under (1)"]))
    print("  Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    importances = pd.Series(model.feature_importances_,
                             index=FEATURE_COLS).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["tomato" if i >= len(importances) - 5 else "steelblue"
              for i in range(len(importances))]
    importances.plot(kind="barh", ax=ax, color=colors)
    ax.set_title("Baseline XGBoost -- Feature Importances (Gain)", fontsize=12)
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("outputs/feature_importances_baseline.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  Saved: outputs/feature_importances_baseline.png")
    print(f"\n  Top 5 features: "
          f"{importances.sort_values(ascending=False).head(5).to_dict()}")

    return model, val_auc


# ==============================================================
# PHASE 4 -- HYPERPARAMETER TUNING  (Fix F: walk-forward CV)
# ==============================================================

def tune_threshold(model, X_val, y_val):
    """Fix A: sweep threshold 0.30->0.70 and pick best macro F1."""
    from sklearn.metrics import f1_score
    y_proba = model.predict_proba(X_val)[:, 1]
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.arange(0.30, 0.71, 0.05):
        preds = (y_proba >= thr).astype(int)
        f1 = f1_score(y_val, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    print(f"  Fix A -- best threshold: {best_thr:.2f}  (macro F1={best_f1:.4f})")
    return best_thr


def tune_model(X_train, y_train, X_val, y_val, scale_pos_weight):
    import xgboost as xgb
    from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
    from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score

    print("\n" + "=" * 60)
    print("PHASE 4: Hyperparameter Tuning  (Fix F: 5-fold walk-forward CV)")
    print("=" * 60)

    # Fix F: 5 expanding walk-forward folds over train+val combined
    tscv = TimeSeriesSplit(n_splits=5)

    param_grid = {
        "max_depth":        [3, 4, 5],
        "learning_rate":    [0.05, 0.1],
        "n_estimators":     [100, 200, 300],
        "subsample":        [0.7, 0.9],
        "colsample_bytree": [0.7, 0.9],
        "min_child_weight": [1, 3],
    }

    base_xgb = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
    )

    # Train + val combined for walk-forward CV
    X_trainval = pd.concat([X_train, X_val]).reset_index(drop=True)
    y_trainval = pd.concat([y_train, y_val]).reset_index(drop=True)

    grid = GridSearchCV(
        base_xgb, param_grid,
        cv=tscv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
        refit=True,
        error_score=0.0,   # folds with single class -> score 0 instead of NaN
    )
    grid.fit(X_trainval, y_trainval)

    print(f"\n  Best params:  {grid.best_params_}")
    print(f"  Best CV AUC (avg over 5 folds): {grid.best_score_:.4f}")

    # Retrain with early stopping on val set using best params
    best_p = {k: v for k, v in grid.best_params_.items() if k != "n_estimators"}
    tuned = xgb.XGBClassifier(
        **best_p,
        scale_pos_weight=scale_pos_weight,
        n_estimators=500,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
    )
    tuned.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"  Best iteration (early stopping): {tuned.best_iteration}")

    # Fix A: tune decision threshold on validation set
    best_threshold = tune_threshold(tuned, X_val, y_val)

    y_proba = tuned.predict_proba(X_val)[:, 1]
    y_pred  = (y_proba >= best_threshold).astype(int)
    val_auc = roc_auc_score(y_val, y_proba)

    print(f"\n  Tuned Val Accuracy:  {accuracy_score(y_val, y_pred):.4f}")
    print(f"  Tuned Val AUC-ROC:   {val_auc:.4f}")
    print("\n  Classification Report (Tuned, threshold adjusted):")
    print(classification_report(y_val, y_pred,
                                 target_names=["GARCH Over (0)", "GARCH Under (1)"]))

    return tuned, val_auc, best_threshold


# ==============================================================
# PHASE 5 -- SHAP VALUES
# ==============================================================

def run_shap(model, X_val):
    try:
        import shap
    except ImportError:
        print("  shap not installed -- skipping SHAP analysis")
        return None

    print("\n" + "=" * 60)
    print("PHASE 5: SHAP Value Analysis")
    print("=" * 60)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_val, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (mean |SHAP|) -- Validation Set", fontsize=11)
    plt.tight_layout()
    plt.savefig("outputs/shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/shap_importance.png")

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_val, show=False)
    plt.title("SHAP Beeswarm -- Direction & Magnitude of Features", fontsize=11)
    plt.tight_layout()
    plt.savefig("outputs/shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/shap_beeswarm.png")

    shap_df  = pd.DataFrame(np.abs(shap_values), columns=FEATURE_COLS)
    top_shap = shap_df.mean().sort_values(ascending=False)
    print("\n  Top 10 features by mean |SHAP|:")
    print(top_shap.head(10).to_string())

    print("\n  Financial sanity checks:")
    top3 = top_shap.head(3).index.tolist()
    print(f"    Top 3 SHAP features: {top3}")
    expected = {"ATM_IV", "Skew", "GARCH_Forecast", "IV_HV_Spread",
                "GARCH_Residual_lag1", "HV_20", "HV_ratio",
                "ATM_IV_regime", "GARCH_Bias_rolling"}
    sensible = [f for f in top3 if f in expected]
    print(f"    Financially intuitive in top 3: {sensible}  "
          f"({'OK' if len(sensible) >= 2 else 'CHECK -- unexpected features dominant'})")

    return shap_values


# ==============================================================
# PHASE 6 -- TEST SET EVALUATION
# ==============================================================

def evaluate_test(model, X_test, y_test, threshold=0.5):
    from sklearn.metrics import (accuracy_score, classification_report,
                                  roc_auc_score, confusion_matrix, roc_curve)

    print("\n" + "=" * 60)
    print("PHASE 6: Final Test Set Evaluation  (run once -- no further tuning)")
    print("=" * 60)
    print(f"  Using decision threshold: {threshold:.2f}  (Fix A)")

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    def grade(val, t1, t2):
        if val > t1:   return "[OK] EXCELLENT"
        if val > t2:   return "[OK] GOOD"
        return "[X] Below target"

    print(f"\n  Test Accuracy:  {acc:.4f}  {grade(acc, 0.60, 0.55)}")
    print(f"  Test AUC-ROC:   {auc:.4f}  {grade(auc, 0.65, 0.55)}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=["GARCH Over (0)", "GARCH Under (1)"]))
    print("  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="tomato", linewidth=2,
             label=f"XGBoost (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.500)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve -- Test Set")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  Saved: outputs/roc_curve.png")

    return y_pred, y_proba, acc, auc


# ==============================================================
# PHASE 7 -- COMBINED VOLATILITY FORECAST
# ==============================================================

def combined_forecast(X_train, y_train_reg, X_val, y_val_reg,
                       X_test, test_df, best_params, scale_pos_weight):
    import xgboost as xgb
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    print("\n" + "=" * 60)
    print("PHASE 7: Combined Volatility Forecast  (GARCH + XGBoost)")
    print("=" * 60)

    reg_params = {k: v for k, v in best_params.items() if k != "n_estimators"}
    reg_model = xgb.XGBRegressor(
        **reg_params,
        n_estimators=300,
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
    )
    reg_model.fit(X_train, y_train_reg,
                  eval_set=[(X_val, y_val_reg)],
                  verbose=False)
    print(f"  Regression best iteration: {reg_model.best_iteration}")

    xgb_correction = reg_model.predict(X_test)
    garch_forecast  = test_df["GARCH_Forecast"].values
    realized        = test_df["Realized_Vol_proxy"].values
    final_forecast  = garch_forecast + xgb_correction

    mae_garch    = mean_absolute_error(realized, garch_forecast)
    mae_combined = mean_absolute_error(realized, final_forecast)
    rmse_garch   = np.sqrt(mean_squared_error(realized, garch_forecast))
    rmse_combined= np.sqrt(mean_squared_error(realized, final_forecast))

    improvement_mae  = (mae_garch - mae_combined) / mae_garch * 100
    improvement_rmse = (rmse_garch - rmse_combined) / rmse_garch * 100

    print(f"\n  Forecast Comparison (Test Set):")
    print(f"    GARCH only      -- MAE: {mae_garch:.6f}  RMSE: {rmse_garch:.6f}")
    print(f"    GARCH + XGBoost -- MAE: {mae_combined:.6f}  RMSE: {rmse_combined:.6f}")
    print(f"    MAE  improvement: {improvement_mae:+.1f}%  "
          f"{'[OK]' if improvement_mae > 0 else '[X]'}")
    print(f"    RMSE improvement: {improvement_rmse:+.1f}%  "
          f"{'[OK]' if improvement_rmse > 0 else '[X]'}")

    combined_closer = (
        np.abs(final_forecast - realized) < np.abs(garch_forecast - realized)
    ).mean()
    print(f"    Test rows where combined is closer to realized: {combined_closer:.1%}")

    dates = test_df.index
    plt.figure(figsize=(14, 5))
    plt.plot(dates, realized,       label="Realized Vol Proxy",  color="black",     linewidth=1.8)
    plt.plot(dates, garch_forecast, label="GARCH Forecast",      color="steelblue", linestyle="--", linewidth=1.3)
    plt.plot(dates, final_forecast, label="GARCH + XGBoost",     color="tomato",    linewidth=1.5)
    plt.fill_between(dates,
                     np.minimum(garch_forecast, final_forecast),
                     np.maximum(garch_forecast, final_forecast),
                     alpha=0.15, color="orange", label="Correction band")
    plt.title("Test Set: GARCH vs GARCH+XGBoost vs Realized Volatility")
    plt.ylabel("Volatility")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/forecast_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  Saved: outputs/forecast_comparison.png")

    return reg_model, xgb_correction, final_forecast, improvement_mae


# ==============================================================
# PHASE 8 -- SAVE MODELS & RESULTS
# ==============================================================

def save_outputs(clf_model, reg_model, test_df, y_pred_cls, y_proba_cls,
                 xgb_correction, final_forecast, threshold):
    import joblib

    print("\n" + "=" * 60)
    print("PHASE 8: Saving Models & Outputs")
    print("=" * 60)

    joblib.dump(clf_model, "models/xgb_classifier.pkl")
    joblib.dump(reg_model, "models/xgb_regressor.pkl")
    clf_model.save_model("models/xgb_classifier.ubj")
    reg_model.save_model("models/xgb_regressor.ubj")
    print("  Saved: models/xgb_classifier.ubj + xgb_regressor.ubj")
    print("  Saved: models/xgb_classifier.pkl + xgb_regressor.pkl")

    results = test_df[["GARCH_Forecast", "GARCH_Error",
                        "Realized_Vol_proxy", "Target_Binary"]].copy()
    results["XGB_Predicted_Class"]  = y_pred_cls
    results["XGB_Pred_Probability"] = y_proba_cls
    results["XGB_Threshold"]        = threshold
    results["XGB_Correction"]       = xgb_correction
    results["Final_Forecast"]       = final_forecast
    results["Correct_Direction"]    = (
        results["XGB_Predicted_Class"] == results["Target_Binary"]
    ).astype(int)

    results.to_csv("outputs/test_set_forecasts.csv")
    print("  Saved: outputs/test_set_forecasts.csv")

    print("\n  Forecast sample (last 5 rows):")
    print(results.tail(5).to_string())


# ==============================================================
# SUMMARY REPORT
# ==============================================================

def print_summary(df, X_train, X_val, X_test,
                  val_auc_base, val_auc_tuned, val_auc_final,
                  test_acc, test_auc, improvement_mae, threshold):
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    print(f"""
  Dataset:         {len(df)} rows  ({df.index.min().date()} -> {df.index.max().date()})
  Features:        {len(FEATURE_COLS)}  (after Fix B/C/D enhancements)
  Train/Val/Test:  {len(X_train)} / {len(X_val)} / {len(X_test)} rows
  Decision threshold: {threshold:.2f}  (Fix A -- tuned on val macro F1)

  -- Enhancement Fixes Applied --------------------------------
  Fix A: scale_pos_weight = sqrt(neg/pos); threshold sweep on val
  Fix B: HV_10 + HV_30 -> HV_20 + HV_ratio  (reduce multicollinearity)
  Fix C: ATM_IV_regime flag  (regime-aware predictions)
  Fix D: GARCH_Bias_rolling  (model remembers GARCH's recent bias)
  Fix E: Dead-zone |GARCH_Error| < {DEAD_ZONE} rows dropped from train/val
  Fix F: 5-fold walk-forward CV for hyperparameter tuning

  -- Classification Results -----------------------------------
  Baseline Val AUC: {val_auc_base:.4f}
  Tuned    Val AUC: {val_auc_tuned:.4f}
  Final    Val AUC: {val_auc_final:.4f}
  Test     Accuracy: {test_acc:.4f}   AUC: {test_auc:.4f}
  Target:  >55% acc, >0.55 AUC  (60%+ = excellent in finance)

  -- Regression Results ---------------------------------------
  MAE improvement over plain GARCH: {improvement_mae:+.1f}%

  -- Outputs --------------------------------------------------
  outputs/eda_signals.png
  outputs/class_balance.png
  outputs/correlation_heatmap.png
  outputs/feature_importances_baseline.png
  outputs/shap_importance.png
  outputs/shap_beeswarm.png
  outputs/roc_curve.png
  outputs/forecast_comparison.png
  outputs/test_set_forecasts.csv
  models/xgb_classifier.ubj
  models/xgb_regressor.ubj
""")


# ==============================================================
# MAIN
# ==============================================================

def main():
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    # Phase 0
    df = load_data()

    # Phase 1
    run_eda(df)

    # Phase 2
    (X_train, y_train, X_val, y_val, X_test, y_test,
     train_df, val_df, test_df, scale_pos_weight) = prepare_data(df)

    # Phase 3 -- baseline
    baseline_model, val_auc_base = train_baseline(
        X_train, y_train, X_val, y_val, scale_pos_weight)

    # Phase 4 -- tuning with walk-forward CV + threshold sweep
    tuned_model, val_auc_tuned, best_threshold = tune_model(
        X_train, y_train, X_val, y_val, scale_pos_weight)

    # Pick better model on val AUC
    if val_auc_tuned >= val_auc_base:
        best_model = tuned_model
        val_auc_final = val_auc_tuned
        print(f"\n  Using TUNED model "
              f"(val AUC {val_auc_tuned:.4f} vs baseline {val_auc_base:.4f})")
    else:
        best_model = baseline_model
        val_auc_final = val_auc_base
        # Re-tune threshold with baseline model
        best_threshold = tune_threshold(baseline_model, X_val, y_val)
        print(f"\n  Using BASELINE model "
              f"(val AUC {val_auc_base:.4f} vs tuned {val_auc_tuned:.4f})")

    # Phase 5 -- SHAP
    run_shap(best_model, X_val)

    # Phase 6 -- test evaluation with tuned threshold
    y_test_pred, y_test_proba, test_acc, test_auc = evaluate_test(
        best_model, X_test, y_test, threshold=best_threshold)

    # Phase 7 -- combined forecast
    best_params = best_model.get_params()
    reg_safe_keys = {"max_depth", "learning_rate", "n_estimators", "subsample",
                     "colsample_bytree", "min_child_weight", "reg_alpha", "reg_lambda"}
    reg_params = {k: v for k, v in best_params.items()
                  if k in reg_safe_keys and v is not None}

    reg_model, xgb_correction, final_forecast, improvement_mae = combined_forecast(
        X_train, train_df["Target_Regression"].reset_index(drop=True),
        X_val,   val_df["Target_Regression"].reset_index(drop=True),
        X_test.reset_index(drop=True), test_df, reg_params, scale_pos_weight,
    )

    # Phase 8 -- save
    save_outputs(best_model, reg_model, test_df,
                 y_test_pred, y_test_proba, xgb_correction,
                 final_forecast, best_threshold)

    # Summary
    print_summary(df, X_train, X_val, X_test,
                  val_auc_base, val_auc_tuned, val_auc_final,
                  test_acc, test_auc, improvement_mae, best_threshold)


if __name__ == "__main__":
    main()
