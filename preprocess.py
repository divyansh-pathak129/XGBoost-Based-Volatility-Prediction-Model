"""
BANKNIFTY Options Data Preprocessing Pipeline
Produces final feature dataset ready for GARCH + XGBoost volatility forecasting model.
Blueprint: volatility forecasting model Blueprint.docx
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR = "BANKNIFTY"
OUT_DIR = "data"
RISK_FREE_RATE = 0.065          # RBI repo rate ~6.5%
OI_MIN = 50                     # minimum OI contracts
MONEYNESS_LOW = 0.80
MONEYNESS_HIGH = 1.20
DTE_MIN = 1
DTE_MAX = 90
IV_MAX = 2.0                    # cap at 200% annualized
MIN_ROWS_PER_DAY = 10
GARCH_WARMUP = 60               # trading days for GARCH warmup
N_JOBS = -1                     # parallel jobs for IV computation

os.makedirs(f"{OUT_DIR}/raw", exist_ok=True)
os.makedirs(f"{OUT_DIR}/processed", exist_ok=True)
os.makedirs(f"{OUT_DIR}/features", exist_ok=True)


# ─────────────────────────────────────────────
# PHASE 1: LOAD & MERGE
# ─────────────────────────────────────────────

def fix_date_column(df, fname):
    """Handle multiple date formats found across NSE monthly files:
    - datetime64: already parsed (most files)
    - int/float DDMMYY: e.g. 10226 → 01/02/26 → 2026-02-01  (FEB26 file)
    - int Excel serial: e.g. 44197 → 2021-01-01 (fallback)
    """
    col = df["Date"]
    if pd.api.types.is_datetime64_any_dtype(col):
        # Already proper datetime
        return df

    # Try DDMMYY integer format first (NSE quirk seen in FEB26 file)
    # Values look like 10226, 20226, ..., 270226  (max 6 digits for DDMMYY)
    non_null = col.dropna()
    if len(non_null) and non_null.max() <= 9999999:
        try:
            # Zero-pad to 6 chars → DDMMYY
            date_strs = non_null.astype(float).astype("Int64").astype(str).str.zfill(6)
            test = pd.to_datetime(date_strs, format="%d%m%y", errors="coerce")
            valid_pct = test.notna().mean()
            if valid_pct > 0.8:
                df["Date"] = pd.to_datetime(
                    col.dropna().astype(float).astype("Int64").astype(str).str.zfill(6),
                    format="%d%m%y", errors="coerce"
                ).reindex(df.index)
                sample = df["Date"].dropna().iloc[0]
                print(f"  [{fname}] Converted DDMMYY integer dates. Sample: {sample.date()}")
                return df
        except Exception:
            pass

    # Fallback: Excel serial date (days since 1899-12-30)
    try:
        df["Date"] = pd.to_datetime(col, unit="D", origin="1899-12-30", errors="coerce")
        sample = df["Date"].dropna().iloc[0]
        print(f"  [{fname}] Converted Excel serial dates. Sample: {sample.date()}")
    except Exception as e:
        df["Date"] = pd.to_datetime(col, errors="coerce")
        print(f"  [{fname}] Date fallback parse. Error was: {e}")
    return df


def load_all_files():
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".xlsx")])
    frames = []
    for f in files:
        path = os.path.join(DATA_DIR, f)
        print(f"Loading {f}...")
        df = pd.read_excel(path)
        df["source_file"] = f
        df = fix_date_column(df, f)
        print(f"  Rows: {len(df):,}  |  Cols: {list(df.columns)}")
        frames.append(df)

    # Standardize column names (strip whitespace)
    frames = [df.rename(columns=lambda c: c.strip()) for df in frames]

    # Verify all files have the same columns
    col_sets = [set(df.columns) for df in frames]
    if len(set(frozenset(s) for s in col_sets)) > 1:
        print("WARNING: Column mismatch across files!")
        for f, cs in zip(files, col_sets):
            print(f"  {f}: {cs}")

    master = pd.concat(frames, ignore_index=True)
    print(f"\nMaster shape after merge: {master.shape}")

    # Drop duplicates
    before = len(master)
    master = master.drop_duplicates(subset=["CONTRACT_D", "Date"], keep="last")
    print(f"Duplicates dropped: {before - len(master):,}")

    # Drop rows with NaN in critical columns
    master = master.dropna(subset=["Date", "UNDRLNG_ST"])
    master = master[master["CLOSE_PRIC"].notna() | master["SETTLEMENT"].notna()]

    # Fill NaN OI with 0
    master["OI_NO_CON"] = master["OI_NO_CON"].fillna(0)

    print(f"Master shape after cleaning: {master.shape}")
    return master


# ─────────────────────────────────────────────
# PHASE 1.2: PARSE CONTRACT_D
# ─────────────────────────────────────────────

import re

CONTRACT_PATTERN = re.compile(
    r"OPTIDXBANKNIFTY(\d{2}-[A-Z]{3}-\d{4})(CE|PE)(\d+)"
)

def parse_contract(contract_str):
    m = CONTRACT_PATTERN.match(str(contract_str).strip())
    if not m:
        return None, None, None
    expiry_str, opt_type, strike = m.groups()
    try:
        expiry_date = pd.to_datetime(expiry_str, format="%d-%b-%Y")
        strike = int(strike)
    except Exception:
        return None, None, None
    return expiry_date, opt_type, strike


def parse_contracts(df):
    print("\nParsing CONTRACT_D...")
    parsed = df["CONTRACT_D"].map(parse_contract)
    df["expiry_date"] = parsed.map(lambda x: x[0])
    df["option_type"] = parsed.map(lambda x: x[1])
    df["strike"] = parsed.map(lambda x: x[2])

    failed = df["expiry_date"].isna().sum()
    print(f"  Failed to parse: {failed:,} rows — dropping them")
    df = df.dropna(subset=["expiry_date", "option_type", "strike"])
    df["strike"] = df["strike"].astype(int)
    return df


# ─────────────────────────────────────────────
# PHASE 1.3: DAYS TO EXPIRY
# ─────────────────────────────────────────────

def compute_dte(df):
    df["DTE"] = (df["expiry_date"] - df["Date"]).dt.days
    neg = (df["DTE"] < 0).sum()
    print(f"  Negative DTE rows (data errors): {neg:,} — dropping")
    df = df[df["DTE"] >= 0]
    expiry_day = (df["DTE"] == 0).sum()
    print(f"  Expiry-day rows (DTE=0): {expiry_day:,} — flagged but kept")
    df["is_expiry_day"] = (df["DTE"] == 0).astype(int)
    return df


# ─────────────────────────────────────────────
# PHASE 2: FILTERING
# ─────────────────────────────────────────────

def apply_filters(df):
    print("\nApplying liquidity filters...")
    n = len(df)

    # Use CLOSE_PRIC; fallback to SETTLEMENT if NaN
    df["close"] = df["CLOSE_PRIC"].where(df["CLOSE_PRIC"].notna(), df["SETTLEMENT"])
    df["spot"] = df["UNDRLNG_ST"]

    # Filter 1: remove zero close price
    df = df[df["close"] > 0]
    print(f"  After zero-close filter: {len(df):,}  (removed {n - len(df):,})")
    n = len(df)

    # Filter 2: OI >= 50
    df = df[df["OI_NO_CON"] >= OI_MIN]
    print(f"  After OI filter (>={OI_MIN}): {len(df):,}  (removed {n - len(df):,})")
    n = len(df)

    # Filter 3: moneyness 0.80–1.20
    df["moneyness"] = df["strike"] / df["spot"]
    df = df[(df["moneyness"] >= MONEYNESS_LOW) & (df["moneyness"] <= MONEYNESS_HIGH)]
    print(f"  After moneyness filter: {len(df):,}  (removed {n - len(df):,})")
    n = len(df)

    # Filter 4: DTE 1–90
    df = df[(df["DTE"] >= DTE_MIN) & (df["DTE"] <= DTE_MAX)]
    print(f"  After DTE filter ({DTE_MIN}–{DTE_MAX}): {len(df):,}  (removed {n - len(df):,})")
    n = len(df)

    # Flag sparse dates
    rows_per_day = df.groupby("Date").size()
    sparse_dates = rows_per_day[rows_per_day < MIN_ROWS_PER_DAY].index
    print(f"  Sparse dates (< {MIN_ROWS_PER_DAY} rows): {len(sparse_dates)}")
    df = df[~df["Date"].isin(sparse_dates)]
    print(f"  After dropping sparse dates: {len(df):,}")

    # Validate spot price consistency per day
    spot_std = df.groupby("Date")["spot"].std()
    bad_days = spot_std[spot_std > 100].index
    if len(bad_days):
        print(f"  WARNING: {len(bad_days)} days with inconsistent spot price (std > 100). Investigate!")
        for d in bad_days[:5]:
            print(f"    {d.date()}: std={spot_std[d]:.2f}")

    return df


# ─────────────────────────────────────────────
# PHASE 3: IMPLIED VOLATILITY
# ─────────────────────────────────────────────

def bs_price(S, K, T, r, sigma, option_type):
    """Black-Scholes option price."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "CE":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def compute_iv_row(row):
    """Compute IV for a single row via brentq root-finding."""
    S = row["spot"]
    K = row["strike"]
    T = row["DTE"] / 365.0
    r = RISK_FREE_RATE
    market_price = row["close"]
    opt_type = row["option_type"]

    if T <= 0 or S <= 0 or K <= 0 or market_price <= 0:
        return np.nan

    # Check intrinsic value
    intrinsic = max(0, S - K) if opt_type == "CE" else max(0, K - S)
    if market_price < intrinsic - 0.01:
        return np.nan  # price below intrinsic — data error

    def objective(sigma):
        return bs_price(S, K, T, r, sigma, opt_type) - market_price

    try:
        iv = brentq(objective, 1e-6, 10.0, xtol=1e-6, maxiter=100)
        if iv > IV_MAX:
            return np.nan
        return iv
    except Exception:
        return np.nan


def compute_iv_batch(df_slice):
    return [compute_iv_row(row) for _, row in df_slice.iterrows()]


def compute_all_iv(df):
    print("\nComputing implied volatility (this may take 5–15 minutes)...")
    rows = df.to_dict("records")

    def iv_for_record(rec):
        S = rec["spot"]
        K = rec["strike"]
        T = rec["DTE"] / 365.0
        r = RISK_FREE_RATE
        market_price = rec["close"]
        opt_type = rec["option_type"]

        if T <= 0 or S <= 0 or K <= 0 or market_price <= 0:
            return np.nan
        intrinsic = max(0, S - K) if opt_type == "CE" else max(0, K - S)
        if market_price < intrinsic - 0.01:
            return np.nan
        def objective(sigma):
            return bs_price(S, K, T, r, sigma, opt_type) - market_price
        try:
            iv = brentq(objective, 1e-6, 10.0, xtol=1e-6, maxiter=100)
            return np.nan if iv > IV_MAX else iv
        except Exception:
            return np.nan

    ivs = Parallel(n_jobs=N_JOBS, backend="loky", verbose=5)(
        delayed(iv_for_record)(rec) for rec in rows
    )
    df["IV"] = ivs

    total = len(df)
    valid = df["IV"].notna().sum()
    print(f"  IV computed: {valid:,}/{total:,} rows ({100*valid/total:.1f}%)")

    # Drop rows with NaN IV
    df = df.dropna(subset=["IV"])

    # Sanity check
    atm = df[df["moneyness"].between(0.98, 1.02)]
    print(f"\n  ATM IV stats (moneyness 0.98–1.02):")
    print(f"    Mean: {atm['IV'].mean()*100:.1f}%  |  Median: {atm['IV'].median()*100:.1f}%")
    print(f"    Min: {atm['IV'].min()*100:.1f}%  |  Max: {atm['IV'].max()*100:.1f}%")

    return df


# ─────────────────────────────────────────────
# PHASE 4: BUILD DAILY FEATURE TABLE
# ─────────────────────────────────────────────

def build_daily_features(df):
    print("\nBuilding daily feature table...")
    dates = sorted(df["Date"].unique())
    print(f"  Trading days in dataset: {len(dates)}")

    # ── Spot price per day (use median for robustness) ──
    daily_spot = df.groupby("Date")["spot"].median().rename("spot")

    # ── 4.1 Daily log return ──
    spot_series = daily_spot.sort_index()
    log_ret = np.log(spot_series / spot_series.shift(1)).rename("log_return")

    # ── 4.7 Historical Volatility ──
    hv10 = (log_ret.rolling(10).std() * np.sqrt(252)).rename("HV_10")
    hv20 = (log_ret.rolling(20).std() * np.sqrt(252)).rename("HV_20")
    hv30 = (log_ret.rolling(30).std() * np.sqrt(252)).rename("HV_30")

    # ── 4.2 ATM IV per day ──
    def get_atm_iv(grp):
        grp = grp.copy()
        grp["atm_dist"] = (grp["moneyness"] - 1.0).abs()
        # nearest expiry
        min_exp = grp["expiry_date"].min()
        front = grp[grp["expiry_date"] == min_exp]
        if front.empty:
            front = grp
        best_strike_idx = front.groupby("option_type")["atm_dist"].idxmin()
        atm_rows = front.loc[best_strike_idx.values]
        return atm_rows["IV"].mean()

    atm_iv = df.groupby("Date").apply(get_atm_iv).rename("ATM_IV")

    # ── 4.3 Skew = Put_IV(0.95m) - Call_IV(1.05m) ──
    def get_skew(grp):
        def closest_iv(sub, target_moneyness):
            candidates = sub[(sub["moneyness"] - target_moneyness).abs() < 0.02]
            if candidates.empty:
                return np.nan
            idx = (candidates["moneyness"] - target_moneyness).abs().idxmin()
            return candidates.loc[idx, "IV"]

        puts = grp[grp["option_type"] == "PE"]
        calls = grp[grp["option_type"] == "CE"]
        put_iv = closest_iv(puts, 0.95)
        call_iv = closest_iv(calls, 1.05)
        if pd.isna(put_iv) or pd.isna(call_iv):
            return np.nan
        return put_iv - call_iv

    skew = df.groupby("Date").apply(get_skew).rename("Skew")

    # ── 4.4 Term Structure Slope ──
    def get_ts_slope(grp):
        expiries = sorted(grp["expiry_date"].unique())
        if len(expiries) < 2:
            return np.nan
        near_exp, far_exp = expiries[0], expiries[1]

        def exp_atm_iv(sub):
            sub = sub.copy()
            sub["atm_dist"] = (sub["moneyness"] - 1.0).abs()
            best = sub.nsmallest(2, "atm_dist")
            return best["IV"].mean() if not best.empty else np.nan

        near_iv = exp_atm_iv(grp[grp["expiry_date"] == near_exp])
        far_iv = exp_atm_iv(grp[grp["expiry_date"] == far_exp])
        if pd.isna(near_iv) or pd.isna(far_iv):
            return np.nan
        return far_iv - near_iv

    ts_slope = df.groupby("Date").apply(get_ts_slope).rename("TS_Slope")

    # ── 4.5 OI Features ──
    total_oi = df.groupby("Date")["OI_NO_CON"].sum().rename("Total_OI")
    put_oi = df[df["option_type"] == "PE"].groupby("Date")["OI_NO_CON"].sum().rename("Put_OI")
    call_oi = df[df["option_type"] == "CE"].groupby("Date")["OI_NO_CON"].sum().rename("Call_OI")
    pcr_oi = (put_oi / call_oi.replace(0, np.nan)).rename("PCR_OI")
    oi_change = total_oi.diff().rename("OI_Change")

    # ── 4.6 Volume Features ──
    total_vol = df.groupby("Date")["TRADED_QUA"].sum().rename("Total_Volume")
    put_vol = df[df["option_type"] == "PE"].groupby("Date")["TRADED_QUA"].sum().rename("Put_Volume")
    call_vol = df[df["option_type"] == "CE"].groupby("Date")["TRADED_QUA"].sum().rename("Call_Volume")
    pcr_vol = (put_vol / call_vol.replace(0, np.nan)).rename("PCR_Volume")
    vol_change = total_vol.diff().rename("Volume_Change")

    # ── 4.8 IV-HV Spread ──
    iv_hv_spread = (atm_iv - hv20).rename("IV_HV_Spread")

    # ── 4.9 DTE / Expiry Features ──
    dte_nearest = df.groupby("Date")["DTE"].min().rename("DTE_nearest")
    is_expiry_week = (dte_nearest <= 5).astype(int).rename("Is_expiry_week")

    expiry_dates_all = sorted(df["expiry_date"].unique())
    def days_since_last_expiry(date):
        past = [e for e in expiry_dates_all if e < date]
        if not past:
            return np.nan
        return (date - max(past)).days

    days_since_exp = pd.Series(
        {d: days_since_last_expiry(d) for d in dates},
        name="Days_since_last_expiry"
    )

    # ── Merge all daily features ──
    daily = pd.concat([
        spot_series, log_ret,
        atm_iv, skew, ts_slope,
        total_oi, oi_change, pcr_oi,
        total_vol, vol_change, pcr_vol,
        hv10, hv20, hv30, iv_hv_spread,
        dte_nearest, is_expiry_week, days_since_exp
    ], axis=1).sort_index()

    daily.index.name = "Date"
    print(f"  Daily feature table shape: {daily.shape}")
    print(f"  Columns: {list(daily.columns)}")
    print(f"  NaN counts:\n{daily.isnull().sum().to_string()}")
    return daily


# ─────────────────────────────────────────────
# PHASE 5: GARCH (ROLLING WINDOW)
# ─────────────────────────────────────────────

def run_garch_rolling(daily):
    try:
        from arch import arch_model
    except ImportError:
        print("arch not installed — skipping GARCH. Run: pip install arch")
        return daily

    print(f"\nRunning rolling GARCH with warmup={GARCH_WARMUP} days...")
    returns = (daily["log_return"].dropna() * 100).sort_index()
    trading_days = returns.index.tolist()
    n = len(trading_days)

    if n < GARCH_WARMUP + 5:
        print(f"  Not enough data for rolling GARCH (need > {GARCH_WARMUP}, have {n})")
        return daily

    garch_forecasts = {}
    garch_residuals = {}

    for i in range(GARCH_WARMUP, n):
        train = returns.iloc[:i]
        try:
            am = arch_model(train, vol="Garch", p=1, q=1, dist="t", rescale=False)
            res = am.fit(disp="off", show_warning=False)
            forecast = res.forecast(horizon=1)
            fc_var = forecast.variance.iloc[-1, 0]
            fc_std = np.sqrt(max(fc_var, 0)) / 100  # back to decimal

            # standardized residual for last in-sample day
            cond_vol = np.sqrt(res.conditional_volatility.iloc[-1]) / 100
            actual_ret = returns.iloc[i - 1] / 100
            std_resid = actual_ret / cond_vol if cond_vol > 0 else np.nan

            garch_forecasts[trading_days[i]] = fc_std
            garch_residuals[trading_days[i - 1]] = std_resid
        except Exception as e:
            garch_forecasts[trading_days[i]] = np.nan
            garch_residuals[trading_days[i - 1]] = np.nan

        if (i - GARCH_WARMUP) % 20 == 0:
            pct = (i - GARCH_WARMUP) / (n - GARCH_WARMUP) * 100
            print(f"  GARCH progress: {pct:.0f}% ({i - GARCH_WARMUP}/{n - GARCH_WARMUP})")

    daily["GARCH_Forecast"] = pd.Series(garch_forecasts)
    daily["GARCH_Residual"] = pd.Series(garch_residuals)

    # GARCH error = actual realized vol (abs log return) - forecast
    # Use HV_5 proxy as "realized vol for that day" = |log_return|
    daily["Realized_Vol_proxy"] = daily["log_return"].abs()
    daily["GARCH_Error"] = daily["Realized_Vol_proxy"] - daily["GARCH_Forecast"]

    valid = daily["GARCH_Forecast"].notna().sum()
    print(f"  GARCH forecasts computed: {valid}/{len(daily)}")
    return daily


# ─────────────────────────────────────────────
# PHASE 6: LAG & ROLLING FEATURES
# ─────────────────────────────────────────────

def add_lag_and_rolling_features(daily):
    print("\nAdding lag and rolling features...")
    key_features = ["ATM_IV", "Skew", "OI_Change", "GARCH_Forecast", "GARCH_Residual"]

    for feat in key_features:
        if feat not in daily.columns:
            continue
        for lag in [1, 2, 5]:
            daily[f"{feat}_lag{lag}"] = daily[feat].shift(lag)

    for feat in ["ATM_IV", "Skew"]:
        if feat not in daily.columns:
            continue
        daily[f"{feat}_5d_mean"] = daily[feat].rolling(5).mean()
        daily[f"{feat}_5d_std"] = daily[feat].rolling(5).std()

    # Drop first 5 rows (lag warmup)
    daily = daily.iloc[5:].copy()
    print(f"  Shape after lag features: {daily.shape}")
    return daily


# ─────────────────────────────────────────────
# PHASE 7: TARGET VARIABLE
# ─────────────────────────────────────────────

def add_target(daily):
    print("\nCreating target variable...")
    if "GARCH_Error" not in daily.columns:
        print("  GARCH_Error not found — skipping target creation")
        return daily

    # Binary: 1 = GARCH underestimated (positive error), 0 = overestimated
    daily["Target_Binary"] = (daily["GARCH_Error"] > 0).astype(int)
    # Also keep continuous for regression option
    daily["Target_Regression"] = daily["GARCH_Error"]

    balance = daily["Target_Binary"].value_counts(normalize=True)
    print(f"  Class balance:\n{balance.to_string()}")
    if abs(balance.get(0, 0) - balance.get(1, 0)) > 0.20:
        print("  WARNING: Class imbalance > 20%. Use scale_pos_weight or SMOTE on train set.")

    return daily


# ─────────────────────────────────────────────
# FINAL FEATURE LIST (as per blueprint Phase 6.4)
# ─────────────────────────────────────────────

FINAL_FEATURES = [
    # GARCH outputs
    "GARCH_Forecast", "GARCH_Residual_lag1", "GARCH_Residual_lag2",
    # Options signals
    "ATM_IV", "ATM_IV_lag1", "ATM_IV_lag2", "ATM_IV_5d_mean",
    "Skew", "Skew_lag1", "TS_Slope", "IV_HV_Spread",
    # Market microstructure
    "OI_Change", "OI_Change_lag1", "PCR_OI", "Volume_Change", "PCR_Volume",
    # Calendar
    "DTE_nearest", "Is_expiry_week", "Days_since_last_expiry",
    # Historical volatility
    "HV_10", "HV_20", "HV_30",
]


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("BANKNIFTY Options Preprocessing Pipeline")
    print("=" * 60)

    # Phase 1
    master = load_all_files()
    master.to_parquet(f"{OUT_DIR}/raw/master_raw.parquet", index=False)
    print(f"\nSaved raw master to {OUT_DIR}/raw/master_raw.parquet")

    master = parse_contracts(master)
    master = compute_dte(master)

    # Phase 2
    master = apply_filters(master)
    master.to_parquet(f"{OUT_DIR}/processed/master_filtered.parquet", index=False)
    print(f"Saved filtered master to {OUT_DIR}/processed/master_filtered.parquet")

    # Phase 3
    master = compute_all_iv(master)
    master.to_parquet(f"{OUT_DIR}/processed/master_with_iv.parquet", index=False)
    print(f"Saved IV dataset to {OUT_DIR}/processed/master_with_iv.parquet")

    # Phase 4
    daily = build_daily_features(master)

    # Phase 5
    daily = run_garch_rolling(daily)

    # Save intermediate daily
    daily.to_parquet(f"{OUT_DIR}/processed/daily_pre_lags.parquet")
    print(f"\nSaved daily pre-lag table to {OUT_DIR}/processed/daily_pre_lags.parquet")

    # Phase 6
    daily = add_lag_and_rolling_features(daily)

    # Phase 7
    daily = add_target(daily)

    # Save final feature table
    # Keep only rows where GARCH forecast exists (after warmup)
    if "GARCH_Forecast" in daily.columns:
        final = daily.dropna(subset=["GARCH_Forecast"])
    else:
        print("\nWARNING: GARCH_Forecast not available — saving without GARCH columns")
        final = daily.copy()

    # Check which final features are available
    available_features = [f for f in FINAL_FEATURES if f in final.columns]
    missing_features = [f for f in FINAL_FEATURES if f not in final.columns]
    if missing_features:
        print(f"\nMissing features (will be excluded): {missing_features}")

    target_cols = ["Target_Binary", "Target_Regression"]
    meta_cols = ["spot", "log_return", "GARCH_Error", "Realized_Vol_proxy"]

    output_cols = available_features + [c for c in target_cols + meta_cols if c in final.columns]
    final = final[output_cols]

    final.to_parquet(f"{OUT_DIR}/features/final_features.parquet")
    final.to_csv(f"{OUT_DIR}/features/final_features.csv")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"Final dataset shape: {final.shape}")
    print(f"Date range: {final.index.min()} -> {final.index.max()}")
    print(f"Features: {available_features}")
    print(f"Saved to: {OUT_DIR}/features/final_features.parquet + .csv")
    print("=" * 60)

    return final


if __name__ == "__main__":
    final = main()
    print("\nSample output (last 5 rows):")
    print(final.tail().to_string())
