"""
context_strip.py — Panel 5: ATM IV / Skew / HV20 / PCR / DTE context row.
"""

import streamlit as st
import pandas as pd


def render_context_strip(day: pd.Series):
    st.markdown("#### Market Context")

    def fmt_pct(v, decimals=2):
        if pd.isna(v):
            return "—"
        return f"{float(v):.{decimals}f}%"

    def fmt_num(v, decimals=2):
        if pd.isna(v):
            return "—"
        return f"{float(v):.{decimals}f}"

    def fmt_delta(v, decimals=2):
        if pd.isna(v):
            return None
        return f"{float(v):+.{decimals}f}"

    cols = st.columns(5)

    cols[0].metric(
        "ATM IV",
        fmt_pct(day.get("ATM_IV")),
        help="At-the-money implied volatility today (%)",
    )
    cols[1].metric(
        "Skew",
        fmt_pct(day.get("Skew"), 3),
        help="Put IV (0.95m) minus Call IV (1.05m) — positive = bearish skew",
    )
    cols[2].metric(
        "HV 20",
        fmt_pct(day.get("HV_20")),
        help="20-day historical (realized) volatility (%)",
    )
    cols[3].metric(
        "PCR (OI)",
        fmt_num(day.get("PCR_OI")),
        delta=fmt_delta(day.get("PCR_OI_delta")),
        delta_color="off",
        help="Put-call ratio by open interest (delta vs prior day)",
    )
    dte = day.get("DTE_nearest")
    cols[4].metric(
        "DTE Near",
        f"{int(dte)}d" if pd.notna(dte) else "—",
        help="Days to nearest expiry",
    )
