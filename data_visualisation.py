"""
Data Visualization & Pre-Modeling Analysis
BANKNIFTY Options -- Volatility Forecasting Pipeline
Input:  data/features/final_features.parquet
Output: outputs/viz_*.png  +  outputs/pre_modeling_checklist.txt

Follows: data_visiualization_plan.md  (Steps 1-10)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.stats import skew as scipy_skew
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.nonparametric.smoothers_lowess import lowess

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

# ---------------------------------------------------------------
# SPLIT DATES (must match xgboost_volatility_model.py)
# ---------------------------------------------------------------
TRAIN_END = "2025-12-31"
VAL_END   = "2026-02-28"

FEATURE_COLS = [
    "GARCH_Forecast", "GARCH_Residual_lag1", "GARCH_Residual_lag2",
    "ATM_IV", "ATM_IV_lag1", "ATM_IV_lag2", "ATM_IV_5d_mean",
    "Skew", "Skew_lag1", "TS_Slope", "IV_HV_Spread",
    "OI_Change", "OI_Change_lag1", "PCR_OI", "Volume_Change", "PCR_Volume",
    "DTE_nearest", "Is_expiry_week", "Days_since_last_expiry",
    "HV_10", "HV_20", "HV_30",
]

# ---------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------
def load_data():
    df = pd.read_parquet("data/features/final_features.parquet")
    df = df.sort_index()
    # Only keep feature cols present in dataset
    global FEATURE_COLS
    FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]
    return df


# ==============================================================
# STEP 1: DATASET HEALTH CHECK
# ==============================================================
def step1_health_check(df):
    print("\n[Step 1] Dataset Health Check")

    # --- Summary table ---
    nan_counts = df[FEATURE_COLS].isnull().sum()
    nan_pct    = nan_counts / len(df) * 100

    print(f"  Rows          : {len(df)}")
    print(f"  Date range    : {df.index.min().date()} -> {df.index.max().date()}")
    print(f"  Trading days  : {len(df)}")
    print(f"  Features      : {len(FEATURE_COLS)}")
    print(f"  Total NaN (features): {df[FEATURE_COLS].isnull().sum().sum()}")
    print("\n  NaN per column:")
    for col in FEATURE_COLS:
        flag = " *** >15%" if nan_pct[col] > 15 else ""
        print(f"    {col:<30} {nan_counts[col]:>4}  ({nan_pct[col]:.1f}%){flag}")

    # --- Missing value heatmap ---
    viz_cols = FEATURE_COLS + ["Target_Binary", "GARCH_Error", "log_return"]
    viz_cols = [c for c in viz_cols if c in df.columns]

    missing_matrix = df[viz_cols].isnull().astype(int)

    fig, ax = plt.subplots(figsize=(18, 6))
    sns.heatmap(
        missing_matrix.T,
        ax=ax,
        cmap=["#1f77b4", "white"],   # blue=present, white=missing
        cbar=False,
        yticklabels=True,
        xticklabels=False,
        linewidths=0,
    )
    ax.set_title("Missing Value Map  (white = missing, blue = present)", fontsize=13)
    ax.set_xlabel("Date (each column = one trading day)")
    ax.set_ylabel("Feature")
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    plt.savefig("outputs/viz_01_missing_values.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/viz_01_missing_values.png")

    # Columns above 15% NaN threshold
    high_nan = nan_pct[nan_pct > 15].index.tolist()
    return high_nan, nan_pct


# ==============================================================
# STEP 2: TARGET VARIABLE ANALYSIS
# ==============================================================
def step2_target_analysis(df):
    print("\n[Step 2] Target Variable Analysis")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Chart 1: Class balance bar chart ---
    counts = df["Target_Binary"].value_counts().sort_index()
    total  = len(df)
    bars = axes[0].bar(
        ["Class 0\n(GARCH over-\nestimated)", "Class 1\n(GARCH under-\nestimated)"],
        [counts.get(0, 0), counts.get(1, 0)],
        color=["steelblue", "tomato"], edgecolor="white", width=0.5
    )
    for bar, cnt in zip(bars, [counts.get(0, 0), counts.get(1, 0)]):
        pct = cnt / total * 100
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{cnt}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=11, fontweight="bold")
    axes[0].set_title("Class Distribution (Target_Binary)", fontsize=12)
    axes[0].set_ylabel("Count")
    axes[0].set_ylim(0, max(counts) * 1.2)

    # --- Chart 2: Target over time scatter ---
    c0 = df[df["Target_Binary"] == 0]
    c1 = df[df["Target_Binary"] == 1]
    axes[1].scatter(c0.index, c0["Target_Binary"], color="steelblue", alpha=0.7, s=40, label="Class 0")
    axes[1].scatter(c1.index, c1["Target_Binary"], color="tomato",    alpha=0.7, s=40, label="Class 1")
    axes[1].set_title("Target_Binary Over Time", fontsize=12)
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Class")
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["0 (Over)", "1 (Under)"])
    axes[1].legend(fontsize=9)
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30, ha="right")

    # --- Chart 3: GARCH_Error histogram ---
    err = df["GARCH_Error"].dropna()
    axes[2].hist(err, bins=30, color="darkorange", edgecolor="white", alpha=0.85)
    axes[2].axvline(0,          color="red",    linestyle="--", linewidth=1.5, label="Zero")
    axes[2].axvline(err.mean(), color="navy",   linestyle="-",  linewidth=1.5, label=f"Mean={err.mean():.4f}")
    axes[2].axvline(err.median(), color="green", linestyle="-.", linewidth=1.5, label=f"Median={err.median():.4f}")
    axes[2].set_title("GARCH_Error Distribution", fontsize=12)
    axes[2].set_xlabel("GARCH Error (Realized - Forecast)")
    axes[2].set_ylabel("Frequency")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("outputs/viz_02_target_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/viz_02_target_analysis.png")

    class_ratio  = counts.get(0, 0) / max(counts.get(1, 1), 1)
    imbalanced   = abs(counts.get(0, 0) / total - counts.get(1, 0) / total) > 0.20
    skew_threshold_zero   = abs(err.mean()) < abs(err.median())   # median closer to 0
    print(f"  Class 0: {counts.get(0,0)} ({counts.get(0,0)/total:.1%})  Class 1: {counts.get(1,0)} ({counts.get(1,0)/total:.1%})")
    print(f"  Imbalanced (>20%): {imbalanced}  -> scale_pos_weight = {class_ratio:.2f}")
    print(f"  GARCH_Error mean={err.mean():.5f}  median={err.median():.5f}")

    return counts, imbalanced, class_ratio, err


# ==============================================================
# STEP 3: FEATURE DISTRIBUTIONS
# ==============================================================
def step3_feature_distributions(df):
    print("\n[Step 3] Feature Distributions")

    n_features  = len(FEATURE_COLS)
    ncols       = 5
    nrows       = (n_features + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4 * nrows))
    axes_flat = axes.flatten()

    skewness_results = {}
    for i, col in enumerate(FEATURE_COLS):
        ax  = axes_flat[i]
        data = df[col].dropna()
        sk   = scipy_skew(data)
        skewness_results[col] = sk

        color = "darkorange" if abs(sk) > 2 else "steelblue"
        ax.hist(data, bins=30, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(data.mean(), color="red", linestyle="--", linewidth=1.2)
        ax.set_title(col, fontsize=8, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.text(0.97, 0.93, f"skew={sk:.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=7,
                color="darkorange" if abs(sk) > 2 else "black")

    # Hide unused subplots
    for j in range(n_features, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Feature Distributions  (orange = |skew| > 2, red line = mean)", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("outputs/viz_03_feature_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/viz_03_feature_distributions.png")

    high_skew = {k: v for k, v in skewness_results.items() if abs(v) > 2}
    print(f"  Features with |skew| > 2: {list(high_skew.keys())}")
    for col, sk in high_skew.items():
        print(f"    {col:<30} skew={sk:.3f}")

    return skewness_results, high_skew


# ==============================================================
# STEP 4: OUTLIER DETECTION
# ==============================================================
def step4_outlier_detection(df):
    print("\n[Step 4] Outlier Detection")

    scaler   = StandardScaler()
    feat_df  = df[FEATURE_COLS].copy()
    scaled   = pd.DataFrame(scaler.fit_transform(feat_df.fillna(feat_df.median())),
                             index=feat_df.index, columns=FEATURE_COLS)

    # --- Box plots (standardized) ---
    fig, ax = plt.subplots(figsize=(20, 6))
    scaled.boxplot(ax=ax, rot=45, fontsize=8)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.7)
    ax.set_title("Feature Box Plots (StandardScaler normalized)", fontsize=13)
    ax.set_ylabel("Z-score")
    plt.tight_layout()
    plt.savefig("outputs/viz_04a_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/viz_04a_boxplots.png")

    # --- Z-score time series for top 5 features ---
    top5 = ["ATM_IV", "Skew", "OI_Change", "GARCH_Forecast", "PCR_Volume"]
    top5 = [c for c in top5 if c in df.columns]

    fig, axes = plt.subplots(len(top5), 1, figsize=(16, 3 * len(top5)), sharex=True)
    if len(top5) == 1:
        axes = [axes]

    shock_dates = set()
    for ax, col in zip(axes, top5):
        z = scaled[col]
        ax.plot(z.index, z, color="steelblue", linewidth=1.0)
        ax.axhline( 3, color="red", linestyle="--", linewidth=1.0, label="+3")
        ax.axhline(-3, color="red", linestyle="--", linewidth=1.0, label="-3")
        ax.fill_between(z.index, -3, 3, alpha=0.05, color="green")
        ax.set_ylabel(f"{col}\nZ-score", fontsize=9)
        ax.set_title(col, fontsize=9)
        # Collect shock dates
        shocks = z.index[z.abs() > 3]
        for d in shocks:
            shock_dates.add(d)

    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Z-score Time Series  (red lines = +-3 sigma)", fontsize=13)
    plt.tight_layout()
    plt.savefig("outputs/viz_04b_zscore_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/viz_04b_zscore_timeseries.png")

    # --- Add Is_market_shock flag ---
    z_all = scaled.abs()
    df["Is_market_shock"] = (z_all.max(axis=1) > 3).astype(int)
    n_shock = df["Is_market_shock"].sum()
    print(f"  Market shock days (any feature Z > 3): {n_shock}")
    if n_shock > 0:
        print(f"  Shock dates: {[d.date() for d in df[df['Is_market_shock']==1].index.tolist()]}")

    return df, shock_dates


# ==============================================================
# STEP 5: FEATURE CORRELATIONS
# ==============================================================
def step5_correlations(df):
    print("\n[Step 5] Feature Correlations")

    corr_cols = FEATURE_COLS + ["Target_Binary"]
    corr_cols = [c for c in corr_cols if c in df.columns]
    corr = df[corr_cols].corr()

    # --- Chart 1: Full correlation heatmap ---
    fig, ax = plt.subplots(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, ax=ax, mask=mask,
        annot=True, fmt=".2f", cmap="RdBu_r",
        vmin=-1, vmax=1, center=0,
        linewidths=0.3, annot_kws={"size": 7},
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Feature Correlation Matrix", fontsize=13)
    ax.tick_params(axis="x", labelsize=8, rotation=45)
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    plt.savefig("outputs/viz_05a_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/viz_05a_correlation_heatmap.png")

    # --- Chart 2: Feature-to-target correlation bar chart ---
    target_corr = corr["Target_Binary"].drop("Target_Binary").sort_values()
    colors = ["green" if abs(v) >= 0.1 else "gray" for v in target_corr]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(target_corr.index, target_corr.values, color=colors, edgecolor="white")
    ax.axvline(0,    color="black", linewidth=0.8)
    ax.axvline( 0.1, color="green", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(-0.1, color="green", linestyle="--", linewidth=0.8, alpha=0.6)
    for bar, val in zip(bars, target_corr.values):
        x = bar.get_width() + (0.005 if val >= 0 else -0.005)
        ax.text(x, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left" if val >= 0 else "right", fontsize=8)
    ax.set_title("Feature Correlation with Target_Binary\n(green = |corr| >= 0.10)", fontsize=12)
    ax.set_xlabel("Pearson Correlation")
    plt.tight_layout()
    plt.savefig("outputs/viz_05b_feature_target_corr.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/viz_05b_feature_target_corr.png")

    # --- Identify highly correlated pairs ---
    high_corr_pairs = []
    feat_only = [c for c in corr_cols if c != "Target_Binary"]
    for i in range(len(feat_only)):
        for j in range(i + 1, len(feat_only)):
            c = corr.loc[feat_only[i], feat_only[j]]
            if abs(c) > 0.85:
                high_corr_pairs.append((feat_only[i], feat_only[j], c))

    low_target_corr = target_corr[target_corr.abs() < 0.05].index.tolist()

    print(f"  Highly correlated pairs (|r|>0.85): {len(high_corr_pairs)}")
    for f1, f2, c in high_corr_pairs:
        print(f"    {f1} <-> {f2}  r={c:.3f}")
    print(f"  Features with |corr to target| < 0.05: {low_target_corr}")

    return target_corr, high_corr_pairs, low_target_corr


# ==============================================================
# STEP 6: TIME SERIES STATIONARITY & STABILITY
# ==============================================================
def step6_stationarity(df):
    print("\n[Step 6] Time Series Stationarity and Stability")

    # --- Chart 1: ATM_IV rolling mean and std band ---
    fig, ax = plt.subplots(figsize=(16, 5))
    iv = df["ATM_IV"].dropna()
    roll_mean = iv.rolling(20).mean()
    roll_std  = iv.rolling(20).std()

    ax.plot(iv.index, iv,          color="steelblue", linewidth=0.9, alpha=0.7, label="ATM_IV")
    ax.plot(roll_mean.index, roll_mean, color="navy", linewidth=2.0, label="20-day rolling mean")
    ax.fill_between(roll_mean.index,
                    roll_mean - roll_std,
                    roll_mean + roll_std,
                    alpha=0.2, color="navy", label="+/- 1 std")
    ax.set_title("ATM_IV: Level, 20-day Rolling Mean, and +/-1 Std Band", fontsize=12)
    ax.set_ylabel("IV (annualized)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("outputs/viz_06a_atm_iv_rolling.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/viz_06a_atm_iv_rolling.png")

    # --- Chart 2: 30-day rolling mean of 4 key features ---
    features_to_plot = [c for c in ["ATM_IV", "Skew", "HV_20", "GARCH_Forecast"] if c in df.columns]
    colors_map = {"ATM_IV": "steelblue", "Skew": "tomato", "HV_20": "green", "GARCH_Forecast": "purple"}

    fig, ax = plt.subplots(figsize=(16, 5))
    for feat in features_to_plot:
        series = df[feat].dropna()
        roll   = series.rolling(30).mean()
        # Normalize to 0-1 for multi-scale display
        rng = roll.max() - roll.min()
        if rng > 0:
            norm_roll = (roll - roll.min()) / rng
        else:
            norm_roll = roll * 0
        ax.plot(norm_roll.index, norm_roll, linewidth=1.8,
                color=colors_map.get(feat, "gray"), label=feat)

    ax.set_title("30-day Rolling Mean of Key Features (normalized 0-1 for scale comparison)", fontsize=12)
    ax.set_ylabel("Normalized Value")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("outputs/viz_06b_rolling_means.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/viz_06b_rolling_means.png")

    # --- Chart 3: ACF of ATM_IV and log_return ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    acf_cols = [("ATM_IV", "ATM Implied Volatility"), ("log_return", "Log Return")]
    lag1_sig = {}
    for ax, (col, title) in zip(axes, acf_cols):
        if col in df.columns:
            series = df[col].dropna()
            plot_acf(series, ax=ax, lags=20, alpha=0.05, zero=False)
            ax.set_title(f"ACF: {title}", fontsize=11)
            ax.set_xlabel("Lag (trading days)")
            # Check lag-1 significance (95% CI ~ 1.96/sqrt(n))
            n  = len(series)
            ci = 1.96 / np.sqrt(n)
            acf_val = series.autocorr(lag=1)
            lag1_sig[col] = abs(acf_val) > ci

    plt.tight_layout()
    plt.savefig("outputs/viz_06c_acf.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/viz_06c_acf.png")

    print(f"  ATM_IV lag-1 autocorr significant: {lag1_sig.get('ATM_IV', False)}")
    print(f"  log_return lag-1 autocorr significant: {lag1_sig.get('log_return', False)}")

    return lag1_sig


# ==============================================================
# STEP 7: GARCH MODEL QUALITY CHECK
# ==============================================================
def step7_garch_quality(df):
    print("\n[Step 7] GARCH Model Quality Check")

    garch_df = df[["GARCH_Forecast", "Realized_Vol_proxy", "GARCH_Error",
                   "GARCH_Residual_lag1"]].dropna()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # --- Chart 1: GARCH Forecast vs Realized Vol ---
    ax = axes[0, 0]
    ax.plot(garch_df.index, garch_df["GARCH_Forecast"],   color="navy",   linewidth=1.2, label="GARCH Forecast")
    ax.plot(garch_df.index, garch_df["Realized_Vol_proxy"], color="tomato", linewidth=1.0, alpha=0.8, label="Realized Vol Proxy")
    ax.set_title("GARCH Forecast vs Realized Volatility", fontsize=11)
    ax.set_ylabel("Volatility")
    ax.legend(fontsize=9)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # --- Chart 2: GARCH_Error histogram + normal overlay + Q-Q ---
    ax = axes[0, 1]
    err = garch_df["GARCH_Error"]
    mu, sigma = err.mean(), err.std()
    kurt = stats.kurtosis(err)

    n_bins = 25
    counts_h, bin_edges = np.histogram(err, bins=n_bins, density=True)
    ax.bar(bin_edges[:-1], counts_h, width=np.diff(bin_edges),
           color="darkorange", edgecolor="white", alpha=0.8, label="GARCH Error")
    x_range = np.linspace(err.min(), err.max(), 200)
    ax.plot(x_range, stats.norm.pdf(x_range, mu, sigma),
            color="navy", linewidth=2, label="Normal curve")
    ax.set_title(f"GARCH Error Distribution\nmean={mu:.5f}  std={sigma:.5f}  kurtosis={kurt:.2f}", fontsize=10)
    ax.set_xlabel("GARCH Error")
    ax.legend(fontsize=8)

    # --- Chart 3: Q-Q plot of GARCH_Error ---
    ax = axes[1, 0]
    stats.probplot(err, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot of GARCH_Error (vs Normal)", fontsize=11)
    ax.get_lines()[0].set(markersize=4, alpha=0.7, color="steelblue")
    ax.get_lines()[1].set(color="red", linewidth=1.5)

    # --- Chart 4: ACF of squared GARCH Residuals ---
    ax = axes[1, 1]
    # Use GARCH_Residual_lag1 as a proxy for residuals (actual residual column)
    residual_col = None
    for c in ["GARCH_Residual", "GARCH_Residual_lag1"]:
        if c in df.columns:
            residual_col = c
            break
    if residual_col:
        residuals = df[residual_col].dropna()
        sq_resid  = residuals ** 2
        plot_acf(sq_resid, ax=ax, lags=15, alpha=0.05, zero=False)
        ax.set_title(f"ACF of Squared GARCH Residuals ({residual_col}^2)", fontsize=10)
        ax.set_xlabel("Lag")
    else:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig("outputs/viz_07_garch_quality.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/viz_07_garch_quality.png")

    garch_bias_mean = err.mean()
    garch_bias_systematic = abs(garch_bias_mean) > 0.002
    print(f"  GARCH Error mean={garch_bias_mean:.6f}  systematic bias: {garch_bias_systematic}")
    print(f"  GARCH Error kurtosis={kurt:.3f}  (fat tails if > 3)")

    return garch_bias_mean, garch_bias_systematic, kurt


# ==============================================================
# STEP 8: FEATURE-TARGET RELATIONSHIP (NON-LINEAR)
# ==============================================================
def step8_nonlinear_relationships(df, target_corr):
    print("\n[Step 8] Feature-Target Non-linear Relationships")

    # Top 6 features by absolute correlation with target
    top6 = target_corr.abs().sort_values(ascending=False).head(6).index.tolist()
    top6 = [c for c in top6 if c in df.columns and c != "Target_Binary"][:6]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()

    for i, col in enumerate(top6):
        ax = axes_flat[i]
        sub = df[[col, "GARCH_Error", "Target_Binary"]].dropna()
        x   = sub[col].values
        y   = sub["GARCH_Error"].values
        cls = sub["Target_Binary"].values

        # Scatter
        ax.scatter(x[cls == 0], y[cls == 0], color="steelblue", alpha=0.6, s=30, label="Class 0")
        ax.scatter(x[cls == 1], y[cls == 1], color="tomato",    alpha=0.6, s=30, label="Class 1")

        # LOWESS smoothing line
        sorted_idx = np.argsort(x)
        xs = x[sorted_idx]
        ys = y[sorted_idx]
        if len(xs) > 10:
            smooth = lowess(ys, xs, frac=0.4, return_sorted=True)
            ax.plot(smooth[:, 0], smooth[:, 1], color="black", linewidth=2.0, label="LOWESS")

        ax.axhline(0, color="gray", linestyle="--", linewidth=0.7)
        ax.set_title(f"{col} vs GARCH_Error", fontsize=10, fontweight="bold")
        ax.set_xlabel(col, fontsize=9)
        ax.set_ylabel("GARCH_Error", fontsize=9)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=8)

    # Hide unused axes
    for j in range(len(top6), 6):
        axes_flat[j].set_visible(False)

    fig.suptitle("Top 6 Features vs GARCH_Error  (color = Target class, black = LOWESS trend)",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig("outputs/viz_08_nonlinear_relationships.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/viz_08_nonlinear_relationships.png")
    print(f"  Top 6 features plotted: {top6}")


# ==============================================================
# STEP 9: TRAIN / VAL / TEST SPLIT VISUALIZATION
# ==============================================================
def step9_split_visualization(df):
    print("\n[Step 9] Train/Val/Test Split Visualization")

    fig, ax = plt.subplots(figsize=(16, 4))

    iv = df["ATM_IV"].dropna()
    ax.plot(iv.index, iv, color="black", linewidth=1.2, label="ATM_IV", zorder=3)

    t_start = iv.index.min()
    t_end   = iv.index.max()
    train_end_dt = pd.Timestamp(TRAIN_END)
    val_end_dt   = pd.Timestamp(VAL_END)

    ax.axvspan(t_start,       train_end_dt, alpha=0.15, color="steelblue", label="Train")
    ax.axvspan(train_end_dt,  val_end_dt,   alpha=0.20, color="orange",    label="Validation")
    ax.axvspan(val_end_dt,    t_end,        alpha=0.20, color="green",     label="Test")

    ax.axvline(train_end_dt, color="steelblue", linestyle="--", linewidth=1.5)
    ax.axvline(val_end_dt,   color="darkorange", linestyle="--", linewidth=1.5)

    ymin, ymax = iv.min(), iv.max()
    mid = (ymin + ymax) / 2
    ax.text(t_start + (train_end_dt - t_start) * 0.5, ymax * 0.97,
            "TRAIN", ha="center", fontsize=11, color="steelblue", fontweight="bold")
    ax.text(train_end_dt + (val_end_dt - train_end_dt) * 0.5, ymax * 0.97,
            "VAL",   ha="center", fontsize=11, color="darkorange", fontweight="bold")
    ax.text(val_end_dt + (t_end - val_end_dt) * 0.5, ymax * 0.97,
            "TEST",  ha="center", fontsize=11, color="green", fontweight="bold")

    ax.set_title("Train / Validation / Test Split  (ATM_IV as reference)", fontsize=12)
    ax.set_ylabel("ATM IV (annualized)")
    ax.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    plt.savefig("outputs/viz_09_train_val_test_split.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: outputs/viz_09_train_val_test_split.png")

    # Check if test IV is dramatically different from train IV
    train_iv_mean = iv[iv.index <= TRAIN_END].mean()
    test_iv_mean  = iv[iv.index > VAL_END].mean()
    ood_flag = abs(test_iv_mean - train_iv_mean) / train_iv_mean > 0.20
    print(f"  Train ATM_IV mean: {train_iv_mean:.4f}  Test ATM_IV mean: {test_iv_mean:.4f}")
    print(f"  Out-of-distribution risk (>20% IV shift): {ood_flag}")
    return ood_flag, train_iv_mean, test_iv_mean


# ==============================================================
# STEP 10: FINAL PRE-MODELING CHECKLIST
# ==============================================================
def step10_checklist(df, nan_pct, high_nan, counts, imbalanced, class_ratio,
                     high_skew, high_corr_pairs, low_target_corr,
                     lag1_sig, garch_bias_mean, garch_bias_systematic, kurt,
                     shock_dates, ood_flag, train_iv_mean, test_iv_mean):
    print("\n[Step 10] Final Pre-Modeling Checklist")

    total = len(df)
    lines = []
    lines.append("=" * 68)
    lines.append("  PRE-MODELING CHECKLIST")
    lines.append("=" * 68)

    lines.append(f"\n1. Dataset size")
    lines.append(f"   Rows in final dataset  : {total}")
    lines.append(f"   Above 150 threshold    : {'YES' if total >= 150 else 'NO -- WARNING: too few rows'}")

    lines.append(f"\n2. Class balance")
    c0, c1 = counts.get(0, 0), counts.get(1, 0)
    lines.append(f"   Class 0 (GARCH over)   : {c0} ({c0/total:.1%})")
    lines.append(f"   Class 1 (GARCH under)  : {c1} ({c1/total:.1%})")
    lines.append(f"   Imbalanced (>20% gap)  : {'YES -- use scale_pos_weight' if imbalanced else 'NO'}")
    lines.append(f"   Recommended scale_pos_weight: {class_ratio:.2f}")

    lines.append(f"\n3. Features needing log transformation (|skew| > 2)")
    if high_skew:
        for col, sk in high_skew.items():
            lines.append(f"   {col:<32} skew={sk:.3f}")
    else:
        lines.append("   None")

    lines.append(f"\n4. Highly correlated feature pairs (|r| > 0.85)")
    if high_corr_pairs:
        for f1, f2, c in high_corr_pairs:
            lines.append(f"   {f1} <-> {f2}  r={c:.3f}")
    else:
        lines.append("   None detected")

    lines.append(f"\n5. Features with near-zero target correlation (|r| < 0.05)")
    if low_target_corr:
        lines.append(f"   {low_target_corr}")
        lines.append("   ACTION: Do not drop yet -- check SHAP values after training")
    else:
        lines.append("   None")

    lines.append(f"\n6. GARCH Error bias")
    lines.append(f"   Mean GARCH Error       : {garch_bias_mean:.6f}")
    lines.append(f"   Systematic bias        : {'YES -- add GARCH_Bias_rolling feature' if garch_bias_systematic else 'NO -- unbiased'}")
    lines.append(f"   Kurtosis               : {kurt:.3f}  ({'fat tails -- t-dist correct' if kurt > 3 else 'near-normal'})")

    lines.append(f"\n7. ATM_IV lag-1 autocorrelation (confirms lag features valid)")
    lines.append(f"   ATM_IV lag-1 significant : {'YES -- lag features validated' if lag1_sig.get('ATM_IV') else 'NO'}")
    lines.append(f"   log_return lag-1 significant : {'YES -- caution, returns have memory' if lag1_sig.get('log_return') else 'NO -- efficient market confirmed'}")

    lines.append(f"\n8. Market shock days (any feature Z-score > 3)")
    lines.append(f"   Count                  : {len(shock_dates)}")
    if shock_dates:
        lines.append(f"   Dates                  : {sorted([d.date() for d in shock_dates])}")
        lines.append("   ACTION: Is_market_shock column added to dataset")
    else:
        lines.append("   None detected")

    lines.append(f"\n9. Columns with >15% NaN")
    if high_nan:
        for col in high_nan:
            lines.append(f"   {col:<32} {nan_pct[col]:.1f}%")
        lines.append("   ACTION: Drop or forward-fill these columns")
    else:
        lines.append("   None -- all columns within threshold")

    lines.append(f"\n10. Train/Test distribution shift")
    lines.append(f"    Train ATM_IV mean      : {train_iv_mean:.4f}")
    lines.append(f"    Test  ATM_IV mean      : {test_iv_mean:.4f}")
    lines.append(f"    OOD risk (>20% shift)  : {'YES -- acknowledge as model limitation' if ood_flag else 'NO -- test period representative'}")

    lines.append("\n" + "=" * 68)
    lines.append("  OUTPUTS SAVED")
    lines.append("=" * 68)
    output_files = [
        "outputs/viz_01_missing_values.png",
        "outputs/viz_02_target_analysis.png",
        "outputs/viz_03_feature_distributions.png",
        "outputs/viz_04a_boxplots.png",
        "outputs/viz_04b_zscore_timeseries.png",
        "outputs/viz_05a_correlation_heatmap.png",
        "outputs/viz_05b_feature_target_corr.png",
        "outputs/viz_06a_atm_iv_rolling.png",
        "outputs/viz_06b_rolling_means.png",
        "outputs/viz_06c_acf.png",
        "outputs/viz_07_garch_quality.png",
        "outputs/viz_08_nonlinear_relationships.png",
        "outputs/viz_09_train_val_test_split.png",
        "outputs/pre_modeling_checklist.txt",
    ]
    for f in output_files:
        lines.append(f"  {f}")
    lines.append("=" * 68)

    checklist_text = "\n".join(lines)
    print(checklist_text)

    with open("outputs/pre_modeling_checklist.txt", "w", encoding="utf-8") as fh:
        fh.write(checklist_text + "\n")
    print("\n  Saved: outputs/pre_modeling_checklist.txt")


# ==============================================================
# MAIN
# ==============================================================
def main():
    print("=" * 60)
    print("Data Visualization & Pre-Modeling Analysis")
    print("BANKNIFTY Volatility Forecasting Pipeline")
    print("=" * 60)

    df = load_data()

    high_nan, nan_pct           = step1_health_check(df)
    counts, imbalanced, class_ratio, err = step2_target_analysis(df)
    skewness_results, high_skew = step3_feature_distributions(df)
    df, shock_dates             = step4_outlier_detection(df)
    target_corr, high_corr_pairs, low_target_corr = step5_correlations(df)
    lag1_sig                    = step6_stationarity(df)
    garch_bias_mean, garch_bias_systematic, kurt = step7_garch_quality(df)
    step8_nonlinear_relationships(df, target_corr)
    ood_flag, train_iv_mean, test_iv_mean = step9_split_visualization(df)
    step10_checklist(
        df, nan_pct, high_nan, counts, imbalanced, class_ratio,
        high_skew, high_corr_pairs, low_target_corr,
        lag1_sig, garch_bias_mean, garch_bias_systematic, kurt,
        shock_dates, ood_flag, train_iv_mean, test_iv_mean
    )

    print("\nAll visualizations complete.")


if __name__ == "__main__":
    main()
