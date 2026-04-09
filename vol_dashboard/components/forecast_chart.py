"""
forecast_chart.py — Panel 3: Forecast vs Realized Volatility line chart.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd


def render_forecast_chart(master: pd.DataFrame, selected_date):
    st.subheader("Forecast vs Realized Volatility")

    df = master.dropna(subset=["GARCH_Forecast"]).copy()

    fig = go.Figure()

    # Realized vol — full history (HV_10 fills gaps)
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["Realized_Vol_proxy"],
        name="Realized Vol",
        line=dict(color="#f0c040", width=1.5),
        opacity=0.85,
    ))

    # GARCH baseline — full history
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["GARCH_Forecast"],
        name="GARCH Forecast",
        line=dict(color="#888888", width=1.5, dash="dash"),
        opacity=0.75,
    ))

    # XGB-corrected forecast — only where available (test period)
    test_df = df.dropna(subset=["Final_Forecast"])
    if not test_df.empty:
        fig.add_trace(go.Scatter(
            x=test_df["date"], y=test_df["Final_Forecast"],
            name="GARCH + XGB",
            line=dict(color="#4a9eff", width=2),
        ))

    # Vertical line for selected date — use shape+annotation to avoid
    # Plotly's internal sum() call on string x-values (TypeError on annotated vlines)
    ts = pd.Timestamp(selected_date)
    fig.add_shape(
        type="line",
        x0=ts, x1=ts,
        y0=0, y1=1,
        yref="paper",
        line=dict(dash="dot", color="#ffffff", width=1),
        opacity=0.4,
    )
    fig.add_annotation(
        x=ts,
        y=1.0,
        yref="paper",
        text="Selected",
        showarrow=False,
        font=dict(color="#aaaaaa", size=11),
        bgcolor="rgba(0,0,0,0.5)",
        xanchor="left",
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Daily Vol (%)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#cccccc",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified",
    )

    fig.update_xaxes(
        rangeselector=dict(buttons=[
            dict(count=1,  label="1M", step="month", stepmode="backward"),
            dict(count=3,  label="3M", step="month", stepmode="backward"),
            dict(count=6,  label="6M", step="month", stepmode="backward"),
            dict(label="All", step="all"),
        ]),
        gridcolor="#2a2a3e",
    )
    fig.update_yaxes(gridcolor="#2a2a3e")

    st.plotly_chart(fig, use_container_width=True)
