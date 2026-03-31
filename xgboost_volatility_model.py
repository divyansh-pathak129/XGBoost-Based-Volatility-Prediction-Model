"""
XGBoost Volatility Prediction Model
BANKNIFTY Options -- GARCH Residual Correction via XGBoost
Input:  data/features/final_features.parquet  (produced by preprocess.py)
Output: models/xgb_classifier.ubj, models/xgb_regressor.ubj
        outputs/test_set_forecasts.csv, outputs/*.png

Plan: xgboost_volatility_model_plan.md
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
# FEATURE COLUMNS (matches preprocess.py FINAL_FEATURES)
# --------------------------------------------------------------

FEATURE_COLS = [
    "GARCH_Forecast", "GARCH_Residual_lag1", "GARCH_Residual_lag2",
    "ATM_IV", "ATM_IV_lag1", "ATM_IV_lag2", "ATM_IV_5d_mean",
    "Skew", "Skew_lag1", "TS_Slope", "IV_HV_Spread",
    "OI_Change", "OI_Change_lag1", "PCR_OI", "Volume_Change", "PCR_Volume",
    "DTE_nearest", "Is_expiry_week", "Days_since_last_expiry",
    "HV_10", "HV_20", "HV_30",
]

# --------------------------------------------------------------
# TIME-BASED SPLIT DATES  (adjusted for actual data range)
# Data: Nov 2025 - Feb 2026 -> 80 usable rows after warmup
# Train: Nov-Dec 2025  (~39 rows)
# Val:   Jan 2026      (~20 rows)
# Test:  Feb 2026      (~21 rows)
# --------------------------------------------------------------

TRAIN_END = "2025-12-31"
VAL_END   = "2026-01-31"


# ==============================================================
# PHASE 0 -- LOAD & VALIDATE DATASET
# ==============================================================

def load_data():
    print("=" * 60)
    print("PHASE 0: Loading dataset")
    print("=" * 60)

    df = pd.read_parquet("data/features/final_features.parquet")
    df = df.sort_index()

    # Keep only columns present in data
    global FEATURE_COLS
    FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in [
        "GARCH_Forecast", "GARCH_Residual_lag1", "GARCH_Residual_lag2",
        "ATM_IV", "ATM_IV_lag1", "ATM_IV_lag2", "ATM_IV_5d_mean",
        "Skew", "Skew_lag1", "TS_Slope", "IV_HV_Spread",
        "OI_Change", "OI_Change_lag1", "PCR_OI", "Volume_Change", "PCR_Volume",
        "DTE_nearest", "Is_expiry_week", "Days_since_last_expiry",
        "HV_10", "HV_20", "HV_30",
    ] if c not in df.columns]
    if missing:
        print(f"  Missing from dataset (excluded): {missing}")

    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df.index.min().date()} -> {df.index.max().date()}")
    print(f"  Features: {len(FEATURE_COLS)}")
    print(f"  NaN in features: {df[FEATURE_COLS].isnull().sum().sum()}")

    # Class balance
    balance = df["Target_Binary"].value_counts(normalize=True)
    print(f"\n  Class balance:")
    print(f"    Class 0 (GARCH overestimated): {balance.get(0,0):.1%}")
    print(f"    Class 1 (GARCH underestimated): {balance.get(1,0):.1%}")

    return df


# ==============================================================
# PHASE 1 -- EDA
# ==============================================================

def run_eda(df):
    print("\n" + "=" * 60)
    print("PHASE 1: Exploratory Data Analysis")
    print("=" * 60)

    # 1.1 Signal plots
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)

    df["ATM_IV"].plot(ax=axes[0], color="steelblue", linewidth=1.2)
    axes[0].set_title("ATM Implied Volatility Over Time")
    axes[0].set_ylabel("IV (annualized)")

    df["GARCH_Forecast"].plot(ax=axes[1], label="GARCH Forecast", color="navy", linewidth=1.2)
    df["Realized_Vol_proxy"].plot(ax=axes[1], label="Realized Vol Proxy", color="tomato",
                                  linewidth=1.0, alpha=0.7)
    axes[1].set_title("GARCH Forecast vs Realized Volatility")
    axes[1].set_ylabel("Volatility")
    axes[1].legend()

    df["Skew"].plot(ax=axes[2], color="purple", linewidth=1.2)
    axes[2].set_title("IV Skew (Put IV − Call IV)")
    axes[2].axhline(0, color="black", linestyle="--", linewidth=0.7)
    axes[2].set_ylabel("Skew")

    df["GARCH_Error"].plot(ax=axes[3], color="darkorange", linewidth=1.2)
    axes[3].axhline(0, color="red", linestyle="--", linewidth=0.8)
    axes[3].set_title("GARCH Error = Realized − Forecast  (XGBoost regression target)")
    axes[3].set_ylabel("Error")

    plt.tight_layout()
    plt.savefig("outputs/eda_signals.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/eda_signals.png")

    # 1.2 Class balance bar chart
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

    # 1.3 Correlation heatmap
    corr_cols = FEATURE_COLS + ["Target_Binary"]
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

    # Top 10 features correlated with target
    target_corr = corr["Target_Binary"].drop("Target_Binary").sort_values(key=abs, ascending=False)
    print(f"\n  Top 10 features by |corr| with Target_Binary:")
    print(target_corr.head(10).to_string())


# ==============================================================
# PHASE 2 -- PREPARE DATA
# ==============================================================

def prepare_data(df):
    print("\n" + "=" * 60)
    print("PHASE 2: Preparing Data")
    print("=" * 60)

    # Drop the one NaN row (GARCH_Residual_lag2)
    # De-duplicate: GARCH_Forecast is already in FEATURE_COLS
    extra_cols = ["Target_Binary", "Target_Regression",
                  "GARCH_Error", "Realized_Vol_proxy", "log_return"]
    needed_cols = list(dict.fromkeys(FEATURE_COLS + extra_cols))  # preserves order, no dupes
    model_df = df[[c for c in needed_cols if c in df.columns]].copy()

    before = len(model_df)
    model_df = model_df.dropna(subset=FEATURE_COLS + ["Target_Binary"])
    print(f"  Dropped {before - len(model_df)} NaN rows -> {len(model_df)} usable rows")

    # Time-based split
    train_df = model_df[model_df.index <= TRAIN_END]
    val_df   = model_df[(model_df.index > TRAIN_END) & (model_df.index <= VAL_END)]
    test_df  = model_df[model_df.index > VAL_END]

    print(f"\n  Train:  {len(train_df)} rows  [{train_df.index.min().date()} -> {train_df.index.max().date()}]")
    print(f"  Val:    {len(val_df)} rows  [{val_df.index.min().date()} -> {val_df.index.max().date()}]")
    print(f"  Test:   {len(test_df)} rows  [{test_df.index.min().date()} -> {test_df.index.max().date()}]")

    # Reset index -- XGBoost 3.x chokes on datetime index in DataFrames
    X_train = train_df[FEATURE_COLS].reset_index(drop=True)
    y_train = train_df["Target_Binary"].reset_index(drop=True)
    X_val   = val_df[FEATURE_COLS].reset_index(drop=True)
    y_val   = val_df["Target_Binary"].reset_index(drop=True)
    X_test  = test_df[FEATURE_COLS].reset_index(drop=True)
    y_test  = test_df["Target_Binary"].reset_index(drop=True)

    # Class imbalance
    counts   = Counter(y_train)
    neg, pos = counts[0], counts[1]
    scale_pos_weight = neg / pos if pos > 0 else 1.0
    print(f"\n  Train class distribution: {dict(counts)}")
    print(f"  scale_pos_weight = {scale_pos_weight:.2f}")

    if abs(neg / len(y_train) - pos / len(y_train)) > 0.20:
        print("  WARNING: Class imbalance >20% -- using scale_pos_weight in XGBoost")

    # Leakage check
    leakage_risk = ["Target_Binary", "Target_Regression", "GARCH_Error",
                    "Realized_Vol_proxy", "log_return"]
    for col in leakage_risk:
        if col in FEATURE_COLS:
            raise ValueError(f"DATA LEAKAGE: {col} found in FEATURE_COLS!")
    print("  Leakage check: PASSED")

    return X_train, y_train, X_val, y_val, X_test, y_test, train_df, val_df, test_df, scale_pos_weight


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

    y_pred  = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    print(f"\n  Validation Accuracy:  {accuracy_score(y_val, y_pred):.4f}")
    print(f"  Validation AUC-ROC:   {roc_auc_score(y_val, y_proba):.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_val, y_pred, target_names=["GARCH Over (0)", "GARCH Under (1)"]))
    print("  Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    # Feature importance plot
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)
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

    top5 = importances.sort_values(ascending=False).head(5)
    print(f"\n  Top 5 features: {top5.to_dict()}")

    if "GARCH_Forecast" not in importances.sort_values(ascending=False).head(15).index:
        print("  WARNING: GARCH_Forecast not in top 15 -- investigate.")

    return model


# ==============================================================
# PHASE 4 -- HYPERPARAMETER TUNING
# ==============================================================

def tune_model(X_train, y_train, X_val, y_val, scale_pos_weight):
    import xgboost as xgb
    from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
    from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

    print("\n" + "=" * 60)
    print("PHASE 4: Hyperparameter Tuning (TimeSeriesSplit CV)")
    print("=" * 60)

    # Use n_splits=3 -- appropriate for ~60 training rows
    tscv = TimeSeriesSplit(n_splits=3)

    param_grid = {
        "max_depth":        [3, 4, 5],
        "learning_rate":    [0.05, 0.1],
        "n_estimators":     [100, 200, 300],
        "subsample":        [0.7, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.9],
        "min_child_weight": [1, 3],
    }

    base_xgb = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
            random_state=42,
        n_jobs=1,
    )

    # Combine train + val for CV
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])

    grid = GridSearchCV(
        base_xgb, param_grid,
        cv=tscv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
        refit=True,
    )
    grid.fit(X_trainval, y_trainval)

    print(f"\n  Best params:  {grid.best_params_}")
    print(f"  Best CV AUC:  {grid.best_score_:.4f}")

    # Retrain best model with early stopping on val set
    # Remove n_estimators from best_p -- we override it with a high ceiling for early stopping
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

    tuned.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=False)

    print(f"  Best iteration (early stopping): {tuned.best_iteration}")

    y_pred  = tuned.predict(X_val)
    y_proba = tuned.predict_proba(X_val)[:, 1]

    print(f"\n  Tuned Val Accuracy:  {accuracy_score(y_val, y_pred):.4f}")
    print(f"  Tuned Val AUC-ROC:   {roc_auc_score(y_val, y_proba):.4f}")
    print("\n  Classification Report (Tuned):")
    print(classification_report(y_val, y_pred, target_names=["GARCH Over (0)", "GARCH Under (1)"]))

    return tuned


# ==============================================================
# PHASE 5 -- SHAP VALUES
# ==============================================================

def run_shap(model, X_val):
    try:
        import shap
    except ImportError:
        print("  shap not installed -- skipping SHAP analysis")
        return

    print("\n" + "=" * 60)
    print("PHASE 5: SHAP Value Analysis")
    print("=" * 60)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)

    # Summary bar plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_val, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (mean |SHAP|) -- Validation Set", fontsize=11)
    plt.tight_layout()
    plt.savefig("outputs/shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/shap_importance.png")

    # Beeswarm
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_val, show=False)
    plt.title("SHAP Beeswarm -- Direction & Magnitude of Features", fontsize=11)
    plt.tight_layout()
    plt.savefig("outputs/shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/shap_beeswarm.png")

    # Top features by mean |SHAP|
    shap_df = pd.DataFrame(np.abs(shap_values), columns=FEATURE_COLS)
    top_shap = shap_df.mean().sort_values(ascending=False)
    print("\n  Top 10 features by mean |SHAP|:")
    print(top_shap.head(10).to_string())

    # Financial sanity checks
    print("\n  Financial sanity checks:")
    top3 = top_shap.head(3).index.tolist()
    print(f"    Top 3 SHAP features: {top3}")
    expected = {"ATM_IV", "Skew", "GARCH_Forecast", "IV_HV_Spread",
                "GARCH_Residual_lag1", "HV_20", "HV_10"}
    sensible = [f for f in top3 if f in expected]
    print(f"    Financially intuitive in top 3: {sensible}  "
          f"({'OK' if len(sensible) >= 2 else 'CHECK -- unexpected features dominant'})")

    return shap_values


# ==============================================================
# PHASE 6 -- TEST SET EVALUATION
# ==============================================================

def evaluate_test(model, X_test, y_test):
    from sklearn.metrics import (accuracy_score, classification_report,
                                  roc_auc_score, confusion_matrix)

    print("\n" + "=" * 60)
    print("PHASE 6: Final Test Set Evaluation  (run once -- no further tuning)")
    print("=" * 60)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm  = confusion_matrix(y_test, y_pred)

    print(f"\n  Test Accuracy:  {acc:.4f}  {'[OK] EXCELLENT' if acc > 0.60 else '[OK] GOOD' if acc > 0.56 else '[OK] OK' if acc > 0.52 else '[X] Below target'}")
    print(f"  Test AUC-ROC:   {auc:.4f}  {'[OK] EXCELLENT' if auc > 0.65 else '[OK] GOOD' if auc > 0.60 else '[OK] OK' if auc > 0.55 else '[X] Below target'}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["GARCH Over (0)", "GARCH Under (1)"]))
    print("  Confusion Matrix:")
    print(cm)

    # ROC curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="tomato", linewidth=2, label=f"XGBoost (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.500)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve -- Test Set")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  Saved: outputs/roc_curve.png")

    return y_pred, y_proba


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

    # Regression model on GARCH_Error
    # Strip n_estimators -- we set it explicitly below
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

    # Predict correction on test set
    xgb_correction = reg_model.predict(X_test)

    garch_forecast = test_df["GARCH_Forecast"].values
    realized       = test_df["Realized_Vol_proxy"].values
    final_forecast = garch_forecast + xgb_correction

    mae_garch    = mean_absolute_error(realized, garch_forecast)
    mae_combined = mean_absolute_error(realized, final_forecast)
    rmse_garch   = np.sqrt(mean_squared_error(realized, garch_forecast))
    rmse_combined= np.sqrt(mean_squared_error(realized, final_forecast))

    improvement_mae  = (mae_garch - mae_combined) / mae_garch * 100
    improvement_rmse = (rmse_garch - rmse_combined) / rmse_garch * 100

    print(f"\n  Forecast Comparison (Test Set):")
    print(f"    GARCH only     -- MAE: {mae_garch:.6f}  RMSE: {rmse_garch:.6f}")
    print(f"    GARCH + XGBoost -- MAE: {mae_combined:.6f}  RMSE: {rmse_combined:.6f}")
    print(f"    MAE  improvement: {improvement_mae:+.1f}%  {'[OK]' if improvement_mae > 0 else '[X]'}")
    print(f"    RMSE improvement: {improvement_rmse:+.1f}%  {'[OK]' if improvement_rmse > 0 else '[X]'}")

    # Direction accuracy
    combined_closer = (
        np.abs(final_forecast - realized) < np.abs(garch_forecast - realized)
    ).mean()
    print(f"    Test rows where combined is closer to realized: {combined_closer:.1%}")

    # Forecast comparison plot
    dates = test_df.index
    plt.figure(figsize=(14, 5))
    plt.plot(dates, realized, label="Realized Vol Proxy", color="black", linewidth=1.8)
    plt.plot(dates, garch_forecast, label="GARCH Forecast", color="steelblue",
             linestyle="--", linewidth=1.3)
    plt.plot(dates, final_forecast, label="GARCH + XGBoost", color="tomato", linewidth=1.5)
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

    return reg_model, xgb_correction, final_forecast


# ==============================================================
# PHASE 8 -- SAVE MODELS & RESULTS
# ==============================================================

def save_outputs(clf_model, reg_model, test_df, y_pred_cls, y_proba_cls,
                 xgb_correction, final_forecast):
    import joblib

    print("\n" + "=" * 60)
    print("PHASE 8: Saving Models & Outputs")
    print("=" * 60)

    # Models
    joblib.dump(clf_model, "models/xgb_classifier.pkl")
    joblib.dump(reg_model, "models/xgb_regressor.pkl")
    clf_model.save_model("models/xgb_classifier.ubj")
    reg_model.save_model("models/xgb_regressor.ubj")
    print("  Saved: models/xgb_classifier.ubj + xgb_regressor.ubj")
    print("  Saved: models/xgb_classifier.pkl + xgb_regressor.pkl")

    # Forecast CSV
    results = test_df[["GARCH_Forecast", "GARCH_Error",
                        "Realized_Vol_proxy", "Target_Binary"]].copy()
    results["XGB_Predicted_Class"]  = y_pred_cls
    results["XGB_Pred_Probability"] = y_proba_cls
    results["XGB_Correction"]       = xgb_correction
    results["Final_Forecast"]       = final_forecast
    results["Correct_Direction"]    = (results["XGB_Predicted_Class"] == results["Target_Binary"]).astype(int)

    results.to_csv("outputs/test_set_forecasts.csv")
    print("  Saved: outputs/test_set_forecasts.csv")

    print("\n  Forecast sample (last 5 rows):")
    print(results.tail(5).to_string())


# ==============================================================
# SUMMARY REPORT
# ==============================================================

def print_summary(df, X_train, X_val, X_test, val_acc, val_auc,
                  test_acc, test_auc, improvement_mae):
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    print(f"""
  Dataset:         {len(df)} rows  ({df.index.min().date()} -> {df.index.max().date()})
  Features:        {len(FEATURE_COLS)}
  Train/Val/Test:  {len(X_train)} / {len(X_val)} / {len(X_test)} rows

  -- Classification Results ------------------------------
  Validation  Accuracy: {val_acc:.4f}   AUC: {val_auc:.4f}
  Test        Accuracy: {test_acc:.4f}   AUC: {test_auc:.4f}
  Target:     >55% acc, >0.55 AUC  (60%+ = excellent in finance)

  -- Regression Results ----------------------------------
  MAE improvement over plain GARCH: {improvement_mae:+.1f}%

  -- Outputs ---------------------------------------------
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
    baseline_model = train_baseline(X_train, y_train, X_val, y_val, scale_pos_weight)
    val_pred_base  = baseline_model.predict(X_val)
    val_proba_base = baseline_model.predict_proba(X_val)[:, 1]
    val_acc_base   = accuracy_score(y_val, val_pred_base)
    val_auc_base   = roc_auc_score(y_val, val_proba_base)

    # Phase 4 -- tuning
    tuned_model = tune_model(X_train, y_train, X_val, y_val, scale_pos_weight)
    val_pred    = tuned_model.predict(X_val)
    val_proba   = tuned_model.predict_proba(X_val)[:, 1]
    val_acc     = accuracy_score(y_val, val_pred)
    val_auc     = roc_auc_score(y_val, val_proba)

    # Use whichever model is better on val set
    if val_auc >= val_auc_base:
        best_model = tuned_model
        print(f"\n  Using TUNED model (val AUC {val_auc:.4f} vs baseline {val_auc_base:.4f})")
    else:
        best_model = baseline_model
        val_acc, val_auc = val_acc_base, val_auc_base
        print(f"\n  Using BASELINE model (val AUC {val_auc_base:.4f} vs tuned {val_auc:.4f})")

    # Phase 5 -- SHAP
    run_shap(best_model, X_val)

    # Phase 6 -- test evaluation
    y_test_pred, y_test_proba = evaluate_test(best_model, X_test, y_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)

    # Phase 7 -- combined forecast
    best_params = (tuned_model.get_params() if best_model is tuned_model
                   else baseline_model.get_params())
    # Strip non-regressor keys
    reg_safe_keys = {"max_depth", "learning_rate", "n_estimators", "subsample",
                     "colsample_bytree", "min_child_weight", "reg_alpha", "reg_lambda"}
    reg_params = {k: v for k, v in best_params.items()
                  if k in reg_safe_keys and v is not None}

    reg_model, xgb_correction, final_forecast = combined_forecast(
        X_train, train_df["Target_Regression"].reset_index(drop=True),
        X_val,   val_df["Target_Regression"].reset_index(drop=True),
        X_test.reset_index(drop=True),  test_df, reg_params, scale_pos_weight
    )

    # MAE improvement for summary
    realized     = test_df["Realized_Vol_proxy"].values
    garch_only   = test_df["GARCH_Forecast"].values
    mae_garch    = mean_absolute_error(realized, garch_only)
    mae_combined = mean_absolute_error(realized, final_forecast)
    improvement_mae = (mae_garch - mae_combined) / mae_garch * 100

    # Phase 8 -- save
    save_outputs(best_model, reg_model, test_df,
                 y_test_pred, y_test_proba, xgb_correction, final_forecast)

    # Summary
    print_summary(df, X_train, X_val, X_test,
                  val_acc, val_auc, test_acc, test_auc, improvement_mae)


if __name__ == "__main__":
    main()
