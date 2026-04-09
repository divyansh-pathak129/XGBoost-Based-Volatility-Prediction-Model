"""
app.py — BANKNIFTY Volatility Forecasting Dashboard.
Run from the project root:  streamlit run vol_dashboard/app.py
"""

import datetime
import streamlit as st
from data_loader import build_master, get_available_dates, load_test_forecasts
from components.signal_hero    import render_signal_hero
from components.forecast_cards import render_forecast_cards
from components.forecast_chart import render_forecast_chart
from components.signal_log     import render_signal_log
from components.context_strip  import render_context_strip

st.set_page_config(
    page_title="BANKNIFTY Vol Forecast",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.6rem !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { font-size: 0.75rem !important; letter-spacing: 0.06em; color: #888; }
[data-testid="stSidebar"]     { background-color: #0c0c14; }
.block-container              { padding-top: 1.2rem; }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
master = build_master()

if master.empty:
    st.error(
        "Could not load pipeline data.\n\n"
        "Make sure these files exist:\n"
        "- `data/features/final_features.parquet`  (run `python preprocess.py`)\n"
        "- `outputs/test_set_forecasts.csv`  (run `python xgboost_volatility_model.py`)"
    )
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📉 BANKNIFTY\nVol Forecast")
    st.divider()

    available_dates = get_available_dates(master)
    selected_date = st.date_input(
        "Trading Date",
        value=available_dates[0],
        min_value=available_dates[-1],
        max_value=available_dates[0],
    )

    # Warn if date has no XGB data (training period)
    test_df = load_test_forecasts()
    if not test_df.empty:
        test_dates = set(test_df["date"].dt.date.tolist())
        if selected_date not in test_dates:
            st.warning("Training-period date — XGB forecast not available.")

    st.caption(f"Data: {available_dates[-1]} → {available_dates[0]}")

    # Stale-data warning — only fire if latest data is more than 7 calendar days old
    # (tolerates weekends, public holidays, and intra-week gaps from missing monthly files)
    latest = available_dates[0]
    if latest < (datetime.date.today() - datetime.timedelta(days=7)):
        st.warning(
            f"⚠️ Latest data: {latest}. "
            "Add new monthly files to `BANKNIFTY/` and re-run `preprocess.py`."
        )

    st.divider()
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# ── Filter to selected date ───────────────────────────────────────────────────
day_row = master[master["date"].dt.date == selected_date]

if day_row.empty:
    st.warning(f"No data for {selected_date}. Select another date.")
    st.stop()

day = day_row.iloc[0]

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    f"<h2 style='margin-bottom:0;'>BANKNIFTY Volatility Forecast "
    f"<span style='color:#888; font-size:1rem;'>{selected_date.strftime('%d %b %Y')}</span></h2>",
    unsafe_allow_html=True,
)
st.divider()

# ── Panel 1: Signal Hero ──────────────────────────────────────────────────────
render_signal_hero(day)
st.divider()

# ── Panel 2: Forecast Breakdown Cards ────────────────────────────────────────
render_forecast_cards(day)
st.divider()

# ── Panels 3 + 4: Chart | Signal Log (side by side) ──────────────────────────
col_left, col_right = st.columns([6, 4])
with col_left:
    render_forecast_chart(master, selected_date)
with col_right:
    render_signal_log(master, selected_date)

st.divider()

# ── Panel 5: Market Context Strip ────────────────────────────────────────────
render_context_strip(day)
