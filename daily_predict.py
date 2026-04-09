"""
daily_predict.py  —  BANKNIFTY Volatility Daily Signal Generator
=================================================================
Reads today's NSE option chain, combines it with historical GARCH + HV
features, and outputs a BUY/SELL vol signal with confidence.

PREREQUISITES (run once before each trading day):
  1. python preprocess.py          <- update data/features/final_features.parquet
  2. python xgboost_volatility_model.py   <- train models (needed once per retrain)
  3. Download today's NSE BANKNIFTY option chain (CSV or XLSX)
     nseindia.com -> Option Chain -> BANKNIFTY -> Download

USAGE:
  python daily_predict.py --chain path/to/chain.csv
  python daily_predict.py --chain path/to/chain.xlsx --spot 52500
  python daily_predict.py --chain chain.csv --spot 52500 --next-expiry 2026-04-30

DAILY WORKFLOW (4:00 PM after market close):
  4:00 PM  Market closes
  4:15 PM  Download option chain from NSE
  Run:     python preprocess.py
  Run:     python daily_predict.py --chain <path>
  Act:     Next morning if flag = ACT
"""

import argparse
import json
import os
import warnings
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────
FEATURES_PATH  = "data/features/final_features.parquet"
STATE_PATH     = "data/daily_state.json"   # persists yesterday's OI/volume/PCR
MODEL_CLF_PKL  = "models/xgb_classifier.pkl"
MODEL_REG_PKL  = "models/xgb_regressor.pkl"
MODEL_META     = "models/model_meta.json"
EXPIRY_CSV     = "data/expiry_dates.csv"   # optional: user-maintained expiry list

RISK_FREE_RATE = 0.065
GARCH_WARMUP   = 60


# ──────────────────────────────────────────────────────────────────────
# PHASE 0: PARSE NSE OPTION CHAIN
# ──────────────────────────────────────────────────────────────────────

def parse_option_chain(filepath: str) -> pd.DataFrame:
    """
    Parse the pre-formatted option chain produced by option_data_formating.py.

    Expected columns (21 data + 2 metadata):
      call_oi, call_chng_oi, call_volume, call_iv, call_ltp, call_net_chng,
      call_bid_qty, call_bid_price, call_ask_price, call_ask_qty,
      strike,
      put_bid_qty, put_bid_price, put_ask_price, put_ask_qty,
      put_net_chng, put_ltp, put_iv, put_volume, put_chng_oi, put_oi,
      [asset], [expiry_date]   ← optional metadata columns

    Remaps to internal uppercase names used by all downstream functions.
    """
    ext = os.path.splitext(filepath)[1].lower()
    df  = pd.read_csv(filepath) if ext == '.csv' else pd.read_excel(filepath)

    print(f"  Raw file shape: {df.shape}")

    # ── Remap to internal column names ──────────────────────────────
    rename_map = {
        'call_oi':        'C_OI',
        'call_chng_oi':   'C_CHNG_OI',
        'call_volume':    'C_VOLUME',
        'call_iv':        'C_IV',
        'call_ltp':       'C_LTP',
        'call_net_chng':  'C_CHNG',
        'call_bid_qty':   'C_BID_QTY',
        'call_bid_price': 'C_BID',
        'call_ask_price': 'C_ASK',
        'call_ask_qty':   'C_ASK_QTY',
        'strike':         'STRIKE',
        'put_bid_qty':    'P_BID_QTY',
        'put_bid_price':  'P_BID',
        'put_ask_price':  'P_ASK',
        'put_ask_qty':    'P_ASK_QTY',
        'put_net_chng':   'P_CHNG',
        'put_ltp':        'P_LTP',
        'put_iv':         'P_IV',
        'put_volume':     'P_VOLUME',
        'put_chng_oi':    'P_CHNG_OI',
        'put_oi':         'P_OI',
        'expiry_date':    'EXPIRY',   # metadata column from formatter
        'asset':          'ASSET',
    }
    df = df.rename(columns=rename_map)

    # ── Clean STRIKE (may still have commas if cleaning wasn't run) ──
    df['STRIKE'] = pd.to_numeric(
        df['STRIKE'].astype(str)
        .str.replace(',', '', regex=False)
        .str.replace('.00', '', regex=False),
        errors='coerce'
    )
    df = df.dropna(subset=['STRIKE'])
    df = df[df['STRIKE'] > 1000]
    df['STRIKE'] = df['STRIKE'].astype(int)

    # ── Ensure all numeric columns are numeric ───────────────────────
    numeric_cols = [c for c in df.columns if c not in ('EXPIRY', 'ASSET')]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.reset_index(drop=True)
    print(f"  Parsed {len(df)} strikes  "
          f"(range {df['STRIKE'].min()} – {df['STRIKE'].max()})")

    # ── Report if EXPIRY column is present ───────────────────────────
    if 'EXPIRY' in df.columns:
        print(f"  Expiry from file: {df['EXPIRY'].iloc[0]}")

    return df


def try_read_spot_from_file(filepath: str) -> float | None:
    """
    Estimate spot price from the formatted chain using put-call parity.
    ATM strike = strike where |call_iv - put_iv| is minimum.
    Falls back to None if insufficient IV data.
    """
    try:
        df = pd.read_csv(filepath)
        df['strike'] = pd.to_numeric(
            df['strike'].astype(str).str.replace(',', '').str.replace('.00', ''),
            errors='coerce'
        )
        df['call_iv'] = pd.to_numeric(df['call_iv'], errors='coerce')
        df['put_iv']  = pd.to_numeric(df['put_iv'],  errors='coerce')

        both = df.dropna(subset=['call_iv', 'put_iv'])
        if len(both) < 5:
            return None

        both = both.copy()
        both['iv_diff'] = (both['call_iv'] - both['put_iv']).abs()
        atm_row = both.loc[both['iv_diff'].idxmin()]
        spot = float(atm_row['strike'])
        print(f"  Spot auto-detected from put-call parity: {spot:,.0f}")
        return spot
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────
# PHASE 1: EXTRACT IV FEATURES FROM OPTION CHAIN
# ──────────────────────────────────────────────────────────────────────

def get_atm_strike(chain: pd.DataFrame, spot: float):
    """Return (atm_strike, atm_row_index) for the given spot price."""
    chain = chain.copy()
    chain["_dist"] = (chain["STRIKE"] - spot).abs()
    idx = chain["_dist"].idxmin()
    return int(chain.loc[idx, "STRIKE"]), idx


def get_atm_iv(chain: pd.DataFrame, spot: float) -> float:
    """
    ATM IV = average of call and put IV at the nearest-to-money strike.
    Falls back to ±1 strike average if the ATM row has NaN IV.
    """
    atm_strike, atm_idx = get_atm_strike(chain, spot)
    c_iv = chain.loc[atm_idx, "C_IV"]
    p_iv = chain.loc[atm_idx, "P_IV"]

    # If ATM has no trade, average the surrounding strikes
    if pd.isna(c_iv) or pd.isna(p_iv):
        nearby = chain[
            (chain["STRIKE"] >= atm_strike - 200) &
            (chain["STRIKE"] <= atm_strike + 200)
        ]
        c_iv = nearby["C_IV"].dropna().mean() if pd.isna(c_iv) else c_iv
        p_iv = nearby["P_IV"].dropna().mean() if pd.isna(p_iv) else p_iv

    if pd.isna(c_iv) and pd.isna(p_iv):
        raise ValueError(f"Cannot determine ATM IV near strike {atm_strike}")

    iv_values = [v for v in [c_iv, p_iv] if not pd.isna(v)]
    return float(np.mean(iv_values)) / 100.0   # NSE reports IV in %, convert to decimal


def get_skew(chain: pd.DataFrame, spot: float) -> float:
    """
    Skew = Put IV at 0.95*spot minus Call IV at 1.05*spot.
    Finds nearest available strike within a 2% band if exact strike missing.
    """
    def find_nearest_iv(col: str, target_strike: float, band_pct: float = 0.02) -> float:
        band = target_strike * band_pct
        candidates = chain[
            (chain["STRIKE"] >= target_strike - band) &
            (chain["STRIKE"] <= target_strike + band) &
            chain[col].notna()
        ]
        if candidates.empty:
            return np.nan
        idx = (candidates["STRIKE"] - target_strike).abs().idxmin()
        return float(candidates.loc[idx, col]) / 100.0

    otm_put_strike  = round(spot * 0.95 / 100) * 100
    otm_call_strike = round(spot * 1.05 / 100) * 100

    put_iv  = find_nearest_iv("P_IV",  otm_put_strike)
    call_iv = find_nearest_iv("C_IV",  otm_call_strike)

    if pd.isna(put_iv) or pd.isna(call_iv):
        print(f"  WARNING: Skew partially unavailable "
              f"(put_iv={put_iv}, call_iv={call_iv}) — using 0.0")
        return 0.0
    return float(put_iv - call_iv)


def get_ts_slope(chain: pd.DataFrame, spot: float) -> tuple[float, bool]:
    """
    Term structure slope = far expiry ATM IV - near expiry ATM IV.
    Requires the chain to have an EXPIRY column, which is rare.
    Falls back to 0.0 with a warning flag if only one expiry is available.
    Returns (ts_slope, used_fallback).
    """
    if "EXPIRY" not in chain.columns:
        return 0.0, True

    expiries = sorted(chain["EXPIRY"].dropna().unique())
    if len(expiries) < 2:
        return 0.0, True

    near_exp, far_exp = expiries[0], expiries[1]

    def exp_atm_iv(sub):
        sub = sub.copy()
        sub["_dist"] = (sub["STRIKE"] - spot).abs()
        best = sub.nsmallest(2, "_dist")
        vals = []
        for _, r in best.iterrows():
            for col in ["C_IV", "P_IV"]:
                if not pd.isna(r[col]):
                    vals.append(r[col] / 100.0)
        return float(np.mean(vals)) if vals else np.nan

    near_iv = exp_atm_iv(chain[chain["EXPIRY"] == near_exp])
    far_iv  = exp_atm_iv(chain[chain["EXPIRY"] == far_exp])

    if pd.isna(near_iv) or pd.isna(far_iv):
        return 0.0, True
    return float(far_iv - near_iv), False


def get_oi_volume(chain: pd.DataFrame) -> dict:
    """
    Compute total OI, volume and put/call ratios from the full chain.
    Returns dict with: total_call_oi, total_put_oi, pcr_oi,
                       total_call_vol, total_put_vol, pcr_volume, total_oi, total_volume
    """
    total_call_oi  = chain["C_OI"].sum(min_count=1)
    total_put_oi   = chain["P_OI"].sum(min_count=1)
    total_call_vol = chain["C_VOLUME"].sum(min_count=1)
    total_put_vol  = chain["P_VOLUME"].sum(min_count=1)

    pcr_oi     = total_put_oi  / (total_call_oi  + 1e-6)
    pcr_volume = total_put_vol / (total_call_vol + 1e-6)

    return {
        "total_call_oi":  float(total_call_oi),
        "total_put_oi":   float(total_put_oi),
        "total_oi":       float(total_call_oi + total_put_oi),
        "pcr_oi":         float(pcr_oi),
        "total_call_vol": float(total_call_vol),
        "total_put_vol":  float(total_put_vol),
        "total_volume":   float(total_call_vol + total_put_vol),
        "pcr_volume":     float(pcr_volume),
    }


# ──────────────────────────────────────────────────────────────────────
# PHASE 2: LOAD HISTORY & REFIT GARCH
# ──────────────────────────────────────────────────────────────────────

def load_history() -> pd.DataFrame:
    """Load the features parquet and validate it is reasonably up-to-date."""
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(
            f"{FEATURES_PATH} not found. Run preprocess.py first."
        )
    df = pd.read_parquet(FEATURES_PATH).sort_index()
    days_stale = (date.today() - df.index.max().date()).days
    if days_stale > 5:
        print(f"  WARNING: Features parquet is {days_stale} days stale "
              f"(last date: {df.index.max().date()}). "
              f"Run preprocess.py to update.")
    else:
        print(f"  History: {len(df)} rows, "
              f"last date {df.index.max().date()} ({days_stale}d ago)")
    return df


def refit_garch(log_returns: pd.Series) -> dict:
    """
    Refit GARCH(1,1) on the full return history and produce:
      - today's 1-step ahead vol forecast
      - yesterday's standardized residual (needed for GARCH_Residual_lag1)
    Returns dict with 'forecast' and 'last_residual'.
    """
    try:
        from arch import arch_model
    except ImportError:
        raise ImportError("arch not installed. Run: pip install arch")

    returns = (log_returns.dropna() * 100).sort_index()
    if len(returns) < GARCH_WARMUP:
        raise ValueError(
            f"Need at least {GARCH_WARMUP} return observations for GARCH. "
            f"Have {len(returns)}."
        )

    am  = arch_model(returns, vol="GARCH", p=1, o=1, q=1,
                     dist="t", rescale=False)
    res = am.fit(disp="off", show_warning=False)

    # 1-step ahead forecast (for today / the next trading day)
    fc_var = res.forecast(horizon=1).variance.iloc[-1, 0]
    fc_std = float(np.sqrt(max(fc_var, 0))) / 100.0   # back to decimal daily SD

    # Standardized residual for the LAST in-sample date (= yesterday)
    cond_vol   = float(np.sqrt(res.conditional_volatility.iloc[-1])) / 100.0
    last_ret   = float(returns.iloc[-1]) / 100.0
    last_resid = last_ret / cond_vol if cond_vol > 0 else np.nan

    return {"forecast": fc_std, "last_residual": last_resid}


def compute_hv(log_returns: pd.Series, windows=(10, 20, 30)) -> dict:
    """Compute annualized historical volatility for given rolling windows."""
    result = {}
    for w in windows:
        hv = float(log_returns.rolling(w).std().iloc[-1] * np.sqrt(252))
        result[f"HV_{w}"] = hv
    return result


def load_state() -> dict:
    """Load yesterday's persisted state (OI, volume, PCR) from JSON."""
    if not os.path.exists(STATE_PATH):
        return {}
    with open(STATE_PATH) as f:
        return json.load(f)


def save_state(state: dict) -> None:
    """Persist today's state for tomorrow's OI_Change / Volume_Change."""
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, default=str)
    print(f"  State saved to {STATE_PATH}")


# ──────────────────────────────────────────────────────────────────────
# PHASE 3: CALENDAR FEATURES
# ──────────────────────────────────────────────────────────────────────

def load_expiry_dates() -> list[date]:
    """
    Load known future expiry dates from expiry_dates.csv if it exists,
    otherwise generate dates algorithmically (last Thursday of each month).
    BANKNIFTY has monthly expiry on the last Thursday of each month.
    """
    if os.path.exists(EXPIRY_CSV):
        df = pd.read_csv(EXPIRY_CSV, parse_dates=["expiry_date"])
        dates = sorted(df["expiry_date"].dt.date.tolist())
        future = [d for d in dates if d >= date.today()]
        if future:
            print(f"  Loaded {len(future)} future expiry dates from {EXPIRY_CSV}")
            return future

    # Generate last Thursday of each month for the next 24 months
    def last_thursday(year: int, month: int) -> date:
        # Find last Thursday: start from last day, go back until Thursday
        if month == 12:
            last_day = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            last_day = date(year, month + 1, 1) - timedelta(days=1)
        days_back = (last_day.weekday() - 3) % 7   # 3 = Thursday
        return last_day - timedelta(days=days_back)

    expiries = []
    today = date.today()
    for offset in range(24):
        month = (today.month - 1 + offset) % 12 + 1
        year  = today.year + (today.month - 1 + offset) // 12
        expiries.append(last_thursday(year, month))

    future = sorted(set(d for d in expiries if d >= today))
    print(f"  Generated {len(future)} expiry dates (last Thursday of each month)")
    return future


def calendar_features(today: date, expiry_dates: list[date]) -> dict:
    """
    Compute DTE_nearest, Is_expiry_week, Days_since_last_expiry.
    """
    future_expiries = [e for e in expiry_dates if e >= today]
    past_expiries   = [e for e in expiry_dates if e < today]

    if not future_expiries:
        raise ValueError(
            "No future expiry dates available. Update expiry_dates.csv or "
            "pass --next-expiry YYYY-MM-DD."
        )

    nearest_expiry = min(future_expiries)
    dte_nearest    = (nearest_expiry - today).days
    is_expiry_week = 1 if dte_nearest <= 5 else 0

    if past_expiries:
        last_expiry          = max(past_expiries)
        days_since_last_exp  = (today - last_expiry).days
    else:
        days_since_last_exp  = 30   # fallback if no past history

    print(f"  Next expiry: {nearest_expiry}  DTE: {dte_nearest}  "
          f"Expiry week: {'YES' if is_expiry_week else 'NO'}")

    return {
        "DTE_nearest":              dte_nearest,
        "Is_expiry_week":           is_expiry_week,
        "Days_since_last_expiry":   days_since_last_exp,
        "next_expiry":              nearest_expiry,
    }


# ──────────────────────────────────────────────────────────────────────
# PHASE 4: ASSEMBLE FEATURE ROW
# ──────────────────────────────────────────────────────────────────────

def assemble_features(
    history:     pd.DataFrame,
    garch_out:   dict,
    hv:          dict,
    chain_iv:    dict,    # ATM_IV, Skew, TS_Slope
    chain_oi:    dict,    # OI, Volume, PCR
    yesterday:   dict,    # loaded from state file
    cal:         dict,    # calendar features
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Build a single-row DataFrame matching the model's FEATURE_COLS.
    All values are scalars; order matches feature_cols exactly.
    """
    h = history.sort_index()

    # ── GARCH features ──────────────────────────────────────────────
    garch_forecast    = garch_out["forecast"]
    # GARCH_Residual_lag1[today] = GARCH_Residual[yesterday] = from GARCH refit
    garch_res_lag1    = garch_out["last_residual"]
    # GARCH_Residual_lag2[today] = GARCH_Residual_lag1[last row in history]
    garch_res_lag2    = float(h["GARCH_Residual_lag1"].iloc[-1])

    # ── IV lag features ─────────────────────────────────────────────
    atm_iv_today      = chain_iv["ATM_IV"]
    atm_iv_lag1       = float(h["ATM_IV"].iloc[-1])
    atm_iv_lag2       = float(h["ATM_IV"].iloc[-2])
    # 5d mean uses last 4 history rows + today
    atm_iv_5d_mean    = float(
        np.mean([atm_iv_today] + h["ATM_IV"].iloc[-4:].tolist())
    )
    skew_lag1         = float(h["Skew"].iloc[-1])
    ts_slope          = chain_iv["TS_Slope"]
    iv_hv_spread      = atm_iv_today - hv["HV_20"]

    # ── OI / Volume features ────────────────────────────────────────
    total_oi_today    = chain_oi["total_oi"]
    total_vol_today   = chain_oi["total_volume"]
    pcr_oi_today      = chain_oi["pcr_oi"]
    pcr_oi_yesterday  = yesterday.get("pcr_oi", pcr_oi_today)   # fallback: no change

    total_oi_yest     = yesterday.get("total_oi",  total_oi_today)
    total_vol_yest    = yesterday.get("total_volume", total_vol_today)

    oi_change         = total_oi_today  - total_oi_yest
    volume_change     = total_vol_today - total_vol_yest
    oi_change_lag1    = float(h["OI_Change"].iloc[-1])
    pcr_oi_change1d   = pcr_oi_today - pcr_oi_yesterday

    # ── Historical volatility ────────────────────────────────────────
    hv_20     = hv["HV_20"]
    hv_10     = hv["HV_10"]
    hv_30     = hv["HV_30"]
    hv_ratio  = hv_10 / hv_30 if hv_30 > 0 else 1.0

    # ── Regime features (Fix C, D, G) ───────────────────────────────
    # ATM_IV_regime: today's IV vs 30-day rolling median of history
    atm_iv_history    = h["ATM_IV"].iloc[-29:].tolist() + [atm_iv_today]
    rolling_med_30    = float(np.median(atm_iv_history))
    atm_iv_regime     = 1 if atm_iv_today > rolling_med_30 else 0

    # GARCH_Bias_rolling: 20-day rolling mean of PAST GARCH errors (shifted 1)
    # history's GARCH_Error column = Realized_Vol_proxy - GARCH_Forecast (past days)
    garch_errors_past = h["GARCH_Error"].iloc[-20:]
    garch_bias_rolling = float(garch_errors_past.mean())

    # GARCH_Bias_short: 5-day rolling mean of PAST GARCH errors
    garch_bias_short  = float(h["GARCH_Error"].iloc[-5:].mean())

    # HV_GARCH_ratio: daily HV / GARCH forecast (> 1 = GARCH underestimating)
    hv_20_daily       = hv_20 / np.sqrt(252)
    hv_garch_ratio    = hv_20_daily / garch_forecast if garch_forecast > 0 else 1.0

    # ATM_IV_trend: today's IV vs 5-day mean
    atm_iv_trend      = atm_iv_today - atm_iv_5d_mean

    # Vol_of_Vol: 10-day rolling std of ATM_IV (including today)
    atm_iv_10d        = h["ATM_IV"].iloc[-9:].tolist() + [atm_iv_today]
    vol_of_vol        = float(np.std(atm_iv_10d, ddof=1)) if len(atm_iv_10d) >= 2 else 0.0

    # Binary threshold features
    hv_garch_above_1  = 1 if hv_garch_ratio > 1.0 else 0
    garch_bias_pos    = 1 if garch_bias_short > 0 else 0

    # ── Assemble dict ────────────────────────────────────────────────
    row = {
        "GARCH_Forecast":           garch_forecast,
        "GARCH_Residual_lag1":      garch_res_lag1,
        "GARCH_Residual_lag2":      garch_res_lag2,
        "ATM_IV":                   atm_iv_today,
        "ATM_IV_lag1":              atm_iv_lag1,
        "ATM_IV_lag2":              atm_iv_lag2,
        "ATM_IV_5d_mean":           atm_iv_5d_mean,
        "Skew":                     chain_iv["Skew"],
        "Skew_lag1":                skew_lag1,
        "TS_Slope":                 ts_slope,
        "IV_HV_Spread":             iv_hv_spread,
        "OI_Change":                oi_change,
        "OI_Change_lag1":           oi_change_lag1,
        "PCR_OI":                   pcr_oi_today,
        "Volume_Change":            volume_change,
        "PCR_Volume":               chain_oi["pcr_volume"],
        "DTE_nearest":              cal["DTE_nearest"],
        "Is_expiry_week":           cal["Is_expiry_week"],
        "Days_since_last_expiry":   cal["Days_since_last_expiry"],
        "HV_20":                    hv_20,
        "HV_ratio":                 hv_ratio,
        "ATM_IV_regime":            atm_iv_regime,
        "GARCH_Bias_rolling":       garch_bias_rolling,
        "HV_GARCH_ratio":           hv_garch_ratio,
        "GARCH_Bias_short":         garch_bias_short,
        "ATM_IV_trend":             atm_iv_trend,
        "Vol_of_Vol":               vol_of_vol,
        "PCR_OI_change1d":          pcr_oi_change1d,
        "HV_GARCH_above_1":         hv_garch_above_1,
        "GARCH_Bias_positive":      garch_bias_pos,
    }

    # Select only the features the model was trained on, in the exact order
    missing = [f for f in feature_cols if f not in row]
    if missing:
        raise ValueError(
            f"Cannot compute these required features: {missing}. "
            f"Ensure preprocess.py has been run and the parquet is up-to-date."
        )

    return pd.DataFrame([{k: row[k] for k in feature_cols}])


# ──────────────────────────────────────────────────────────────────────
# PHASE 5: PREDICT AND REPORT
# ──────────────────────────────────────────────────────────────────────

def load_models():
    """Load classifier and regressor from disk."""
    import joblib
    if not os.path.exists(MODEL_CLF_PKL):
        raise FileNotFoundError(
            f"{MODEL_CLF_PKL} not found. Run xgboost_volatility_model.py first."
        )
    clf = joblib.load(MODEL_CLF_PKL)
    reg = joblib.load(MODEL_REG_PKL)
    return clf, reg


def load_threshold(default: float = 0.30) -> tuple[float, list[str]]:
    """
    Load the decision threshold and feature list saved during training.
    Falls back to default if the file doesn't exist.
    """
    if os.path.exists(MODEL_META):
        with open(MODEL_META) as f:
            meta = json.load(f)
        threshold = float(meta.get("threshold", default))
        feature_cols = meta.get("feature_cols", [])
        print(f"  Loaded threshold={threshold:.2f} and {len(feature_cols)} features "
              f"from {MODEL_META}")
        return threshold, feature_cols
    print(f"  WARNING: {MODEL_META} not found — using default threshold={default:.2f}. "
          f"Run xgboost_volatility_model.py to generate it.")
    return default, []


def print_report(
    today_date:      date,
    spot:            float,
    garch_forecast:  float,
    xgb_correction:  float,
    final_vol:       float,
    prob:            float,
    pred_class:      int,
    threshold:       float,
    chain_iv:        dict,
    chain_oi:        dict,
    cal:             dict,
    hv:              dict,
    hv_garch_ratio:  float,
    garch_bias_short: float,
    ts_fallback:     bool,
) -> None:
    """Print the formatted daily signal report."""
    direction  = "BUY VOL  [LONG]" if pred_class == 1 else "SELL VOL [SHORT]"
    confidence = prob if pred_class == 1 else 1 - prob
    act        = confidence > 0.62

    flag_str   = "[ACT ]" if act else "[SKIP]"
    flag_note  = "confidence > 62%, edge present" if act else \
                 "confidence near 50%, no clear edge — skip"

    ts_note = " (fallback=0, single expiry)" if ts_fallback else ""

    print()
    print("=" * 50)
    print(f"   BANKNIFTY VOL SIGNAL — {today_date}")
    print("=" * 50)
    print(f"  Spot:                {spot:,.2f}")
    print()
    print(f"  GARCH Forecast:      {garch_forecast * np.sqrt(252) * 100:.2f}%  annualized")
    print(f"  XGB Correction:      {xgb_correction * np.sqrt(252) * 100:+.2f}%")
    print(f"  Final Forecast:      {final_vol * np.sqrt(252) * 100:.2f}%  annualized")
    print()
    print(f"  Signal:              {direction}")
    print(f"  Probability:         {prob:.1%}  (threshold {threshold:.0%})")
    print(f"  Confidence:          {confidence:.1%}")
    print(f"  Trade Flag:          {flag_str}  — {flag_note}")
    print()
    print("  Supporting Signals:")
    print(f"    ATM IV today:      {chain_iv['ATM_IV'] * 100:.2f}%")
    print(f"    HV_20:             {hv['HV_20'] * 100:.2f}%  (annualized)")
    print(f"    HV/GARCH ratio:    {hv_garch_ratio:.3f}  "
          f"({'> 1: GARCH may underestimate' if hv_garch_ratio > 1 else '< 1: GARCH may overestimate'})")
    print(f"    GARCH 5d bias:     {garch_bias_short * 100:+.3f}%  "
          f"({'pos: recent underestimation' if garch_bias_short > 0 else 'neg: recent overestimation'})")
    print(f"    IV Skew:           {chain_iv['Skew']:+.4f}  "
          f"({'puts expensive' if chain_iv['Skew'] > 0 else 'calls expensive'})")
    print(f"    TS Slope:          {chain_iv['TS_Slope']:+.4f}{ts_note}")
    print(f"    PCR OI:            {chain_oi['pcr_oi']:.3f}")
    print(f"    PCR Volume:        {chain_oi['pcr_volume']:.3f}")
    print(f"    DTE nearest:       {cal['DTE_nearest']} days")
    print(f"    Expiry week:       {'YES' if cal['Is_expiry_week'] else 'NO'}")
    print(f"    Next expiry:       {cal['next_expiry']}")
    print("=" * 50)
    print()


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="BANKNIFTY daily volatility signal generator"
    )
    parser.add_argument(
        "--chain", required=True,
        help="Path to NSE BANKNIFTY option chain file (CSV or XLSX)"
    )
    parser.add_argument(
        "--spot", type=float, default=None,
        help="BANKNIFTY spot price. Auto-detected from file header if omitted."
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Today's date in YYYY-MM-DD (default: today). Override for backtesting."
    )
    parser.add_argument(
        "--next-expiry", type=str, default=None,
        help="Override next expiry date in YYYY-MM-DD. If omitted, computed automatically."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    today = (datetime.strptime(args.date, "%Y-%m-%d").date()
             if args.date else date.today())

    print("=" * 50)
    print(f"  BANKNIFTY Daily Signal — {today}")
    print("=" * 50)

    # ── Load model meta (threshold + feature list) ───────────────────
    threshold, feature_cols = load_threshold()
    if not feature_cols:
        # Fallback: use default feature list from xgboost_volatility_model.py
        feature_cols = [
            "GARCH_Forecast", "GARCH_Residual_lag1", "GARCH_Residual_lag2",
            "ATM_IV", "ATM_IV_lag1", "ATM_IV_lag2", "ATM_IV_5d_mean",
            "Skew", "Skew_lag1", "TS_Slope", "IV_HV_Spread",
            "OI_Change", "OI_Change_lag1", "PCR_OI", "Volume_Change", "PCR_Volume",
            "DTE_nearest", "Is_expiry_week", "Days_since_last_expiry",
            "HV_20", "HV_ratio", "ATM_IV_regime", "GARCH_Bias_rolling",
            "HV_GARCH_ratio", "GARCH_Bias_short", "ATM_IV_trend",
            "Vol_of_Vol", "PCR_OI_change1d", "HV_GARCH_above_1", "GARCH_Bias_positive",
        ]
        print(f"  Using built-in feature list ({len(feature_cols)} features)")

    # ── Phase 0: Parse option chain ──────────────────────────────────
    print("\n[Phase 0] Parsing option chain...")
    chain = parse_option_chain(args.chain)

    # ── Extract expiry date from chain metadata or filename ──────────
    expiry_from_chain = None

    # Method 1: From the EXPIRY column added by option_data_formating.py
    if 'EXPIRY' in chain.columns:
        raw_exp = chain['EXPIRY'].iloc[0]
        try:
            expiry_from_chain = datetime.strptime(str(raw_exp), "%d-%b-%Y").date()
            print(f"  Expiry from chain column: {expiry_from_chain}")
        except ValueError:
            pass

    # Method 2: Parse from filename directly (format: option_chain_BANKNIFTY-28-Apr-2026.csv)
    if expiry_from_chain is None:
        try:
            fname = os.path.basename(args.chain)                  # option_chain_BANKNIFTY-28-Apr-2026.csv
            name  = fname.replace('option_chain_', '').replace('.csv', '')
            # name = 'BANKNIFTY-28-Apr-2026'
            date_part = '-'.join(name.split('-')[1:])             # '28-Apr-2026'
            try:
                expiry_from_chain = datetime.strptime(date_part, "%d-%b-%Y").date()
            except ValueError:
                expiry_from_chain = datetime.strptime(date_part, "%d-%B-%Y").date()
            print(f"  Expiry from filename: {expiry_from_chain}")
        except Exception:
            print("  WARNING: Could not parse expiry from filename")

    # Inject into args so calendar features use the correct DTE
    if expiry_from_chain and not args.next_expiry:
        args.next_expiry = str(expiry_from_chain)

    # ── Determine spot price ─────────────────────────────────────────
    spot = args.spot
    if spot is None:
        spot = try_read_spot_from_file(args.chain)
    if spot is None:
        # Best estimate: midpoint of range of strikes with highest OI
        top_oi_strikes = chain.nlargest(5, "C_OI")["STRIKE"].tolist()
        spot = float(np.median(top_oi_strikes))
        print(f"  WARNING: Spot not provided. Estimated from top OI strikes: {spot:,.0f}")
        print(f"  Pass --spot <value> for accuracy.")
    else:
        print(f"  Spot price: {spot:,.2f}")

    # ── Phase 1: Extract IV features ─────────────────────────────────
    print("\n[Phase 1] Extracting option chain features...")
    atm_iv   = get_atm_iv(chain, spot)
    skew     = get_skew(chain, spot)
    ts_slope, ts_fallback = get_ts_slope(chain, spot)
    oi_vol   = get_oi_volume(chain)

    print(f"  ATM IV:    {atm_iv*100:.2f}%")
    print(f"  Skew:      {skew:+.4f}")
    print(f"  TS Slope:  {ts_slope:+.4f}" + (" [fallback]" if ts_fallback else ""))
    print(f"  PCR OI:    {oi_vol['pcr_oi']:.3f}")
    print(f"  PCR Vol:   {oi_vol['pcr_volume']:.3f}")
    print(f"  Total OI:  {oi_vol['total_oi']:,.0f}")

    chain_iv_dict = {"ATM_IV": atm_iv, "Skew": skew, "TS_Slope": ts_slope}

    # ── Phase 2: Load history & refit GARCH ──────────────────────────
    print("\n[Phase 2] Loading history and refitting GARCH...")
    history  = load_history()
    garch_out = refit_garch(history["log_return"])
    hv        = compute_hv(history["log_return"])

    print(f"  GARCH forecast (daily SD): {garch_out['forecast']*100:.3f}%  "
          f"({garch_out['forecast']*np.sqrt(252)*100:.2f}% annualized)")
    print(f"  Last residual:  {garch_out['last_residual']:.3f}")
    print(f"  HV_10: {hv['HV_10']*100:.2f}%  "
          f"HV_20: {hv['HV_20']*100:.2f}%  "
          f"HV_30: {hv['HV_30']*100:.2f}%")

    # ── Load yesterday's state ────────────────────────────────────────
    yesterday = load_state()
    if yesterday:
        prev_date = yesterday.get("date", "unknown")
        print(f"  Yesterday state: {prev_date}  "
              f"OI={yesterday.get('total_oi',0):,.0f}  "
              f"Vol={yesterday.get('total_volume',0):,.0f}")
    else:
        print("  No previous state found — OI_Change / Volume_Change will be 0")

    # ── Phase 3: Calendar features ────────────────────────────────────
    print("\n[Phase 3] Computing calendar features...")
    expiry_dates = load_expiry_dates()
    if args.next_expiry:
        # User-specified next expiry takes priority
        user_expiry = datetime.strptime(args.next_expiry, "%Y-%m-%d").date()
        if user_expiry not in expiry_dates:
            expiry_dates = sorted(expiry_dates + [user_expiry])
        print(f"  Using user-specified next expiry: {user_expiry}")
    cal = calendar_features(today, expiry_dates)

    # ── Phase 4: Assemble feature row ─────────────────────────────────
    print("\n[Phase 4] Assembling feature row...")
    X_today = assemble_features(
        history=history,
        garch_out=garch_out,
        hv=hv,
        chain_iv=chain_iv_dict,
        chain_oi=oi_vol,
        yesterday=yesterday,
        cal=cal,
        feature_cols=feature_cols,
    )
    print(f"  Feature row shape: {X_today.shape}")
    nan_feats = X_today.columns[X_today.iloc[0].isna()].tolist()
    if nan_feats:
        print(f"  WARNING: NaN in features: {nan_feats} — predictions may be unreliable")

    # ── Phase 5: Predict ─────────────────────────────────────────────
    print("\n[Phase 5] Running model...")
    clf, reg = load_models()

    prob        = float(clf.predict_proba(X_today)[0][1])
    pred_class  = 1 if prob >= threshold else 0
    xgb_corr    = float(reg.predict(X_today)[0])
    final_vol   = garch_out["forecast"] + xgb_corr

    hv_garch_ratio  = float(X_today["HV_GARCH_ratio"].iloc[0])
    garch_bias_short = float(X_today["GARCH_Bias_short"].iloc[0])

    # ── Report ────────────────────────────────────────────────────────
    print_report(
        today_date=today,
        spot=spot,
        garch_forecast=garch_out["forecast"],
        xgb_correction=xgb_corr,
        final_vol=final_vol,
        prob=prob,
        pred_class=pred_class,
        threshold=threshold,
        chain_iv=chain_iv_dict,
        chain_oi=oi_vol,
        cal=cal,
        hv=hv,
        hv_garch_ratio=hv_garch_ratio,
        garch_bias_short=garch_bias_short,
        ts_fallback=ts_fallback,
    )

    # ── Save today's state for tomorrow ──────────────────────────────
    save_state({
        "date":         str(today),
        "total_oi":     oi_vol["total_oi"],
        "total_volume": oi_vol["total_volume"],
        "pcr_oi":       oi_vol["pcr_oi"],
        "spot":         spot,
    })

    # ── Save prediction log ───────────────────────────────────────────
    log_path = "outputs/daily_predictions.csv"
    log_row  = {
        "date":             str(today),
        "spot":             spot,
        "prob":             round(prob, 4),
        "pred_class":       pred_class,
        "threshold":        threshold,
        "garch_forecast":   round(garch_out["forecast"] * np.sqrt(252) * 100, 3),
        "xgb_correction":   round(xgb_corr * np.sqrt(252) * 100, 3),
        "final_vol":        round(final_vol * np.sqrt(252) * 100, 3),
        "atm_iv":           round(atm_iv * 100, 3),
        "skew":             round(skew, 4),
        "pcr_oi":           round(oi_vol["pcr_oi"], 4),
        "dte_nearest":      cal["DTE_nearest"],
        "hv_garch_ratio":   round(hv_garch_ratio, 4),
        "garch_bias_short": round(garch_bias_short * 100, 4),
        "act":              int(abs(prob - 0.5) > 0.12),   # confidence > 62%
    }

    os.makedirs("outputs", exist_ok=True)
    log_df = pd.DataFrame([log_row])
    if os.path.exists(log_path):
        existing = pd.read_csv(log_path)
        # Avoid duplicate entries for the same date
        existing = existing[existing["date"] != str(today)]
        log_df = pd.concat([existing, log_df], ignore_index=True)
    log_df.to_csv(log_path, index=False)
    print(f"  Prediction logged to {log_path}")


if __name__ == "__main__":
    main()
