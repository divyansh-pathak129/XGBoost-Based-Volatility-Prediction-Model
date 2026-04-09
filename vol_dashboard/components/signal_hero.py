"""
signal_hero.py — Panel 1: BUY VOL / SELL VOL / HOLD hero block.
"""

import streamlit as st
import pandas as pd


SIGNAL_COLORS = {
    "BUY VOL":     ("#00e676", "#0d3b1e"),
    "SELL VOL":    ("#ff5252", "#3b0d0d"),
    "HOLD":        ("#ffaa00", "#3b2e00"),
    "NO XGB DATA": ("#888888", "#1a1a1a"),
}


def render_signal_hero(day: pd.Series):
    from data_loader import derive_signal

    signal, confidence = derive_signal(day.to_dict())
    text_color, bg_color = SIGNAL_COLORS.get(signal, ("#ffffff", "#111111"))

    # Final forecast: prefer XGB-corrected, fall back to GARCH only
    final_fc = day.get("Final_Forecast", float("nan"))
    garch_fc = day.get("GARCH_Forecast", float("nan"))
    display_fc = final_fc if not pd.isna(final_fc) else garch_fc

    # Values are already in percent after build_master() normalisation
    ann_fc = display_fc * (252 ** 0.5) if not pd.isna(display_fc) else None

    col1, col2, col3 = st.columns([2, 2, 1.5])

    with col1:
        st.markdown(f"""
        <div style="background:{bg_color}; border-radius:12px; padding:24px; text-align:center;">
            <div style="font-size:0.85rem; color:#aaa; letter-spacing:0.1em; margin-bottom:6px;">SIGNAL</div>
            <div style="font-size:2.8rem; font-weight:900; color:{text_color}; line-height:1;">{signal}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        ann_str   = f"{ann_fc:.2f}% ann."  if ann_fc is not None else "—"
        daily_str = f"{display_fc:.3f}% daily" if not pd.isna(display_fc) else "—"
        label     = "FINAL FORECAST" if not pd.isna(final_fc) else "GARCH FORECAST"
        st.markdown(f"""
        <div style="background:#1a1a2e; border-radius:12px; padding:24px; text-align:center;">
            <div style="font-size:0.85rem; color:#aaa; letter-spacing:0.1em; margin-bottom:6px;">{label}</div>
            <div style="font-size:2.2rem; font-weight:700; color:#4a9eff; line-height:1;">{ann_str}</div>
            <div style="font-size:0.9rem; color:#888; margin-top:6px;">{daily_str}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        bar_pct = int(confidence * 100)
        st.markdown(f"""
        <div style="background:#1a1a1a; border-radius:12px; padding:24px; text-align:center;">
            <div style="font-size:0.85rem; color:#aaa; letter-spacing:0.1em; margin-bottom:6px;">CONFIDENCE</div>
            <div style="font-size:2rem; font-weight:700; color:{text_color}; line-height:1;">{bar_pct}%</div>
            <div style="background:#333; border-radius:4px; height:8px; margin-top:10px;">
                <div style="background:{text_color}; width:{bar_pct}%; height:8px; border-radius:4px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
