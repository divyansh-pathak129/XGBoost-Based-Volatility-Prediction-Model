"""
forecast_cards.py — Panel 2: GARCH / Correction / Final metric cards.
"""

import streamlit as st
import pandas as pd


def render_forecast_cards(day: pd.Series):
    garch    = day.get("GARCH_Forecast",  float("nan"))
    corr     = day.get("XGB_Correction",  float("nan"))
    final_fc = day.get("Final_Forecast",  float("nan"))

    # Values are already in percent
    def fmt(v):
        if pd.isna(v):
            return "—"
        return f"{v:.3f}%"

    def fmt_delta(v):
        if pd.isna(v):
            return None
        return f"{v:+.3f}%"

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "GARCH Forecast",
        fmt(garch),
        help="Raw GARCH(1,1) one-day-ahead vol forecast (daily %)",
    )
    col2.metric(
        "XGB Correction",
        fmt(corr),
        delta=fmt_delta(corr),
        delta_color="normal",
        help="Correction the XGBoost regressor applied on top of GARCH",
    )

    # Final forecast falls back to GARCH-only with a note
    if not pd.isna(final_fc):
        col3.metric(
            "Final Forecast",
            fmt(final_fc),
            help="Final = GARCH + XGB Correction",
        )
    else:
        col3.metric(
            "Final Forecast",
            fmt(garch) + " (GARCH only)" if not pd.isna(garch) else "—",
            help="No XGB data for this date — showing raw GARCH forecast",
        )
