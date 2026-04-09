"""
data_loader.py — All file loading, caching, and merging logic.
Every component imports from here; no component opens files directly.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Project root = two levels up from this file (vol_dashboard/ -> project root)
ROOT = Path(__file__).resolve().parent.parent

FEATURE_COLS = [
    "GARCH_Forecast", "GARCH_Residual_lag1", "GARCH_Residual_lag2",
    "ATM_IV", "ATM_IV_lag1", "ATM_IV_lag2", "ATM_IV_5d_mean",
    "Skew", "Skew_lag1", "TS_Slope", "IV_HV_Spread",
    "OI_Change", "OI_Change_lag1", "PCR_OI", "Volume_Change", "PCR_Volume",
    "DTE_nearest", "Is_expiry_week", "Days_since_last_expiry",
    "HV_10", "HV_20", "HV_30",
]

# Columns that are in decimal form and need *100 conversion to percent
VOL_COLS = [
    "GARCH_Forecast", "Final_Forecast", "Realized_Vol_proxy",
    "XGB_Correction", "HV_10", "HV_20", "HV_30",
    "ATM_IV", "ATM_IV_lag1", "ATM_IV_lag2", "ATM_IV_5d_mean",
    "Skew", "Skew_lag1", "TS_Slope", "IV_HV_Spread",
]


@st.cache_data
def load_features() -> pd.DataFrame:
    path = ROOT / "data" / "features" / "final_features.parquet"
    try:
        df = pd.read_parquet(path)
        # Date is stored as the index named "Date"
        df = df.reset_index()
        # Normalise to lowercase "date"
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "date"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        before = len(df)
        df = df.dropna(subset=["date"])
        dropped = before - len(df)
        if dropped:
            print(f"[data_loader] Dropped {dropped} rows with unparseable dates")
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except FileNotFoundError:
        st.error(f"Missing: {path}\nRun `python preprocess.py` first.")
        return pd.DataFrame()


@st.cache_data
def load_test_forecasts() -> pd.DataFrame:
    path = ROOT / "outputs" / "test_set_forecasts.csv"
    try:
        df = pd.read_csv(path)
        # Drop spurious unnamed index column if present
        if df.columns[0] == "Unnamed: 0":
            df = df.drop(columns=["Unnamed: 0"])
        # Handle both "date" and "Date" column names
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "date"})
        if "date" not in df.columns:
            df = df.reset_index()
            df.rename(columns={"index": "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except FileNotFoundError:
        st.error(f"Missing: {path}\nRun `python xgboost_volatility_model.py` first.")
        return pd.DataFrame()


@st.cache_data
def build_master() -> pd.DataFrame:
    """
    Merge final_features with test_set_forecasts on date.
    Training-period rows will have NaN for XGB columns — handled in components.
    All vol columns are normalised to percent (multiplied by 100 if in decimal form).
    """
    features = load_features()
    if features.empty:
        return pd.DataFrame()

    forecasts = load_test_forecasts()

    forecast_cols = [
        "date", "Final_Forecast", "XGB_Correction",
        "XGB_Predicted_Class", "XGB_Pred_Probability",
        "Correct_Direction", "Target_Binary", "Realized_Vol_proxy",
    ]
    if not forecasts.empty:
        forecast_cols = [c for c in forecast_cols if c in forecasts.columns]
        master = features.merge(
            forecasts[forecast_cols],
            on="date",
            how="left",
            suffixes=("", "_test"),
        )
        # Prefer test CSV's Realized_Vol_proxy over parquet's when both exist
        if "Realized_Vol_proxy_test" in master.columns:
            master["Realized_Vol_proxy"] = master["Realized_Vol_proxy_test"].fillna(
                master["Realized_Vol_proxy"]
            )
            master = master.drop(columns=["Realized_Vol_proxy_test"])
    else:
        master = features.copy()

    # Fill Realized_Vol_proxy gaps from HV_10
    if "Realized_Vol_proxy" not in master.columns:
        master["Realized_Vol_proxy"] = master["HV_10"]
    else:
        master["Realized_Vol_proxy"] = master["Realized_Vol_proxy"].fillna(master["HV_10"])

    # Unit normalisation: if GARCH_Forecast is in decimal, convert all vol cols to percent
    if master["GARCH_Forecast"].median() < 0.1:
        for col in VOL_COLS:
            if col in master.columns:
                master[col] = master[col] * 100

    # Delta features for context strip
    master["PCR_OI_delta"] = master["PCR_OI"].diff()

    return master


def get_available_dates(df: pd.DataFrame) -> list:
    return sorted(df["date"].dt.date.unique().tolist(), reverse=True)


def derive_signal(row: dict) -> tuple:
    """
    Derive (signal_str, confidence_float) from a row dict.
    Returns 'NO XGB DATA' if XGB columns are missing or NaN.
    """
    CONFIDENCE_THRESHOLD = 0.60

    prob = row.get("XGB_Pred_Probability", np.nan)
    pred_class = row.get("XGB_Predicted_Class", np.nan)

    if pd.isna(prob) or pd.isna(pred_class):
        return "NO XGB DATA", 0.0

    if pred_class == 1:
        conf = float(prob)
        signal = "BUY VOL" if conf >= CONFIDENCE_THRESHOLD else "HOLD"
    else:
        conf = float(1 - prob)
        signal = "SELL VOL" if conf >= CONFIDENCE_THRESHOLD else "HOLD"

    return signal, conf
