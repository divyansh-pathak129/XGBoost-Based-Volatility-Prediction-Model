"""
signal_log.py — Panel 4: Recent signal accuracy table (last 15 test-period days).
"""

import streamlit as st
import pandas as pd


def render_signal_log(master: pd.DataFrame, selected_date):
    st.subheader("Recent Signal Log")

    from data_loader import derive_signal

    # Only rows with XGB data, up to and including selected date
    df = master[
        master["XGB_Pred_Probability"].notna() &
        (master["date"].dt.date <= selected_date)
    ].tail(15).copy()

    if df.empty:
        st.info("No XGB signal history available for this date range.")
        return

    rows = []
    for _, row in df.iterrows():
        signal, conf = derive_signal(row.to_dict())
        correct = row.get("Correct_Direction", float("nan"))
        correct_str = "✅" if correct == 1 else ("❌" if correct == 0 else "—")
        garch_res = row.get("GARCH_Residual_lag1", float("nan"))
        rows.append({
            "Date":      row["date"].strftime("%d %b %y"),
            "Signal":    signal,
            "Conf":      f"{conf:.0%}",
            "Correct":   correct_str,
            "GARCH Res": f"{garch_res:+.3f}" if pd.notna(garch_res) else "—",
        })

    display_df = pd.DataFrame(rows).iloc[::-1]  # most recent first

    def color_signal(val):
        if val == "BUY VOL":  return "color: #00e676"
        if val == "SELL VOL": return "color: #ff5252"
        if val == "HOLD":     return "color: #ffaa00"
        return ""

    styled = display_df.style.map(color_signal, subset=["Signal"])
    st.dataframe(styled, use_container_width=True, hide_index=True)
