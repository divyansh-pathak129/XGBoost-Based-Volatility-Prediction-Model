# BANKNIFTY Volatility Forecasting Dashboard — Implementation Guide
### A complete step-by-step plan for Claude Code

---

## Overview

Build a Streamlit dashboard for personal daily use. It reads from existing pipeline output files
(parquet/CSV/JSON) — it never re-runs the pipeline or the model. The dashboard shows 5 panels:
today's signal hero, GARCH breakdown cards, forecast vs realized chart, recent signal log, and
a market context strip. Default view is always the latest available day; a date selector allows
browsing history.

**Stack:** Python · Streamlit · Pandas · Plotly · XGBoost (read-only, for live prediction path)

---

## Data Files Used — Confirm All Paths Before Starting

| File | Used For | Required? |
|---|---|---|
| `data/features/final_features.parquet` | Full history: all panels, context strip, chart | ✅ Hard required |
| `outputs/test_set_forecasts.csv` | Historical accuracy columns (Correct_Direction, XGB_Predicted_Class) | ✅ Hard required |
| `data/daily_state.json` | Latest OI/volume/PCR delta state (for "today" context) | Optional |
| `models/xgb_classifier.ubj` | Live prediction if daily_predict.py hasn't been run yet | Optional |
| `models/xgb_regressor.ubj` | Live prediction (same) | Optional |

### Critical column inventory

**`final_features.parquet`** must contain:
```
date, GARCH_Forecast, ATM_IV, ATM_IV_lag1, ATM_IV_5d_mean,
Skew, TS_Slope, IV_HV_Spread, OI_Change, PCR_OI,
Volume_Change, PCR_Volume, DTE_nearest, Is_expiry_week,
Days_since_last_expiry, HV_10, HV_20, HV_30,
GARCH_Residual_lag1, GARCH_Residual_lag2
```
Also needs one of: `Realized_Vol_proxy` OR `HV_10` as a realized vol stand-in.

**`test_set_forecasts.csv`** must contain:
```
date (index or column), GARCH_Forecast, Final_Forecast,
Realized_Vol_proxy, XGB_Predicted_Class, XGB_Pred_Probability,
XGB_Correction, Correct_Direction, Target_Binary
```

> Run `python parquet_viewer.py data/features/final_features.parquet` to confirm column names
> before coding any component.

---

## Project Structure

```
vol_dashboard/
├── app.py                      ← main Streamlit entry point, layout assembly
├── data_loader.py              ← all file loading + caching + merging logic
├── components/
│   ├── signal_hero.py          ← Panel 1: BUY VOL / SELL VOL / HOLD hero
│   ├── forecast_cards.py       ← Panel 2: GARCH / Correction / Final metric cards
│   ├── forecast_chart.py       ← Panel 3: forecast vs realized line chart
│   ├── signal_log.py           ← Panel 4: recent signal accuracy table
│   └── context_strip.py        ← Panel 5: ATM IV / Skew / HV20 / PCR row
├── utils.py                    ← shared helpers: vol formatting, signal color map
└── requirements.txt
```

---

## Step 1 — Project Scaffold & Dependencies

### What to do

1. Create the folder structure above.
2. Create `requirements.txt`:
```
streamlit>=1.32.0
pandas>=2.0.0
plotly>=5.18.0
pyarrow>=14.0.0
numpy>=1.26.0
xgboost>=2.0.0
scikit-learn>=1.3.0
```
3. Create a bare `app.py` that just runs `st.title("BANKNIFTY Vol Dashboard")` and confirm
   it boots with `streamlit run vol_dashboard/app.py` from the project root.
4. Verify both data files exist at their expected paths before writing any other code.

### Potential Problems & Solutions

**Problem:** Dashboard and pipeline share a Python environment, version conflicts arise.
**Solution:** Always use a dedicated venv: `python -m venv vol_dash_env`. Never install into
the pipeline's environment.

**Problem:** `pyarrow` missing, parquet won't load.
**Solution:** It must be in requirements.txt explicitly. `fastparquet` is NOT a substitute —
it handles some parquet encodings differently. Use `pyarrow` only.

**Problem:** `streamlit run` from the wrong directory breaks all relative paths.
**Solution:** Anchor every path in `data_loader.py` to `Path(__file__).resolve().parent.parent`
(two levels up from `vol_dashboard/`, pointing at the project root). Test by running from
multiple directories.

---

## Step 2 — Data Loader (`data_loader.py`)

This is the most important file. Every component imports from here. No component should
directly open any file.

### What to do

```python
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # project root

FEATURE_COLS = [
    "GARCH_Forecast", "GARCH_Residual_lag1", "GARCH_Residual_lag2",
    "ATM_IV", "ATM_IV_lag1", "ATM_IV_lag2", "ATM_IV_5d_mean",
    "Skew", "Skew_lag1", "TS_Slope", "IV_HV_Spread",
    "OI_Change", "OI_Change_lag1", "PCR_OI", "Volume_Change", "PCR_Volume",
    "DTE_nearest", "Is_expiry_week", "Days_since_last_expiry",
    "HV_10", "HV_20", "HV_30",
]

@st.cache_data
def load_features() -> pd.DataFrame:
    path = ROOT / "data" / "features" / "final_features.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

@st.cache_data
def load_test_forecasts() -> pd.DataFrame:
    path = ROOT / "outputs" / "test_set_forecasts.csv"
    df = pd.read_csv(path)
    # Handle case where date is the index
    if "date" not in df.columns:
        df = df.reset_index()
        df.rename(columns={"index": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

@st.cache_data
def build_master() -> pd.DataFrame:
    """
    Merge final_features with test_set_forecasts on date.
    For dates NOT in test_set_forecasts (i.e. training period),
    XGB columns will be NaN — handle gracefully in each component.
    """
    features = load_features()
    forecasts = load_test_forecasts()

    forecast_cols = [
        "date", "Final_Forecast", "XGB_Correction",
        "XGB_Predicted_Class", "XGB_Pred_Probability",
        "Correct_Direction", "Target_Binary", "Realized_Vol_proxy"
    ]
    # Only keep columns that actually exist in the test file
    forecast_cols = [c for c in forecast_cols if c in forecasts.columns]

    master = features.merge(
        forecasts[forecast_cols],
        on="date",
        how="left"  # keep all feature rows, NaN for test-only columns
    )
    return master

def get_available_dates(df: pd.DataFrame) -> list:
    return sorted(df["date"].dt.date.unique().tolist(), reverse=True)

def derive_signal(row) -> tuple[str, float]:
    """
    Derive signal and confidence from a single row of master df.
    Returns (signal_str, confidence_float).
    Falls back to GARCH-only if XGB columns are missing.
    """
    CONFIDENCE_THRESHOLD = 0.60  # matches daily_predict.py default

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
```

### Key design decisions
- `@st.cache_data` on every loader — parquet is large, cache is mandatory.
- `build_master()` is the single source of truth. All components receive a slice of this.
- `derive_signal()` lives in the loader so signal logic is in one place and consistent.
- Left-merge means training-period rows exist with NaN XGB columns — components must handle this.

### Potential Problems & Solutions

**Problem:** `date` column in `final_features.parquet` is stored as integer (`20250401`) or
string (`"2025-04-01"`).
**Solution:** In `load_features()`:
```python
df["date"] = pd.to_datetime(df["date"], infer_datetime_format=True, errors="coerce")
df = df.dropna(subset=["date"])
```
Log dropped row count to stdout.

**Problem:** `test_set_forecasts.csv` uses the date as the DataFrame index (written by pandas
`to_csv()` without `index=False`).
**Solution:** The loader already handles this with `reset_index()` + rename. Additionally
check: `if df.columns[0] == "Unnamed: 0": df = df.drop(columns=["Unnamed: 0"])`.

**Problem:** `Realized_Vol_proxy` is missing from `final_features.parquet` for the full
history (it may only exist in `test_set_forecasts.csv`).
**Solution:** After the merge in `build_master()`, for rows where `Realized_Vol_proxy` is NaN,
fill it from `HV_10` as a proxy:
```python
if "Realized_Vol_proxy" not in master.columns:
    master["Realized_Vol_proxy"] = master["HV_10"]
else:
    master["Realized_Vol_proxy"] = master["Realized_Vol_proxy"].fillna(master["HV_10"])
```
This gives the forecast chart a full history line even outside the test period.

**Problem:** `Final_Forecast` is only in `test_set_forecasts.csv`, leaving it NaN for all
training-period rows in the master table.
**Solution:** For the forecast chart, plot `Final_Forecast` only where it exists, and plot
`GARCH_Forecast` for the full period. The chart will naturally show GARCH alone for the
training period and both lines for the test period.

**Problem:** File not found (pipeline hasn't been run yet).
**Solution:** Wrap loaders in try/except, return empty DataFrame with correct schema, and
show `st.error()` in the app rather than crashing.

---

## Step 3 — Sidebar & Date Selector (`app.py`)

### What to do

The sidebar is minimal — just a date picker defaulting to the latest day.

```python
with st.sidebar:
    st.markdown("## 📊 BANKNIFTY\nVol Forecast")
    st.divider()

    master = build_master()
    available_dates = get_available_dates(master)

    selected_date = st.date_input(
        "Date",
        value=available_dates[0],        # default = latest
        min_value=available_dates[-1],
        max_value=available_dates[0],
    )

    st.caption(f"History: {available_dates[-1]} → {available_dates[0]}")

    # Warn if selected date is outside test period (no XGB data)
    test_df = load_test_forecasts()
    test_dates = set(test_df["date"].dt.date.tolist())
    if selected_date not in test_dates:
        st.warning("Selected date is in training period — no XGB forecast available.")

# Filter master to selected date (single row)
day_row = master[master["date"].dt.date == selected_date]

if day_row.empty:
    st.warning(f"No data for {selected_date}. Select another date.")
    st.stop()

day = day_row.iloc[0]  # single row as Series — pass to components
```

### Potential Problems & Solutions

**Problem:** `st.date_input` accepts any date in range, including weekends/holidays with no
data in the parquet.
**Solution:** The `day_row.empty` guard + `st.stop()` after filtering handles this. Optionally
restrict the date picker: `st.date_input(..., value=available_dates[0])` but there's no
built-in "only show available dates" in Streamlit — the guard is sufficient.

**Problem:** Latest date in `final_features.parquet` is stale (pipeline not re-run today).
**Solution:** In the sidebar, add a last-updated line:
```python
latest = available_dates[0]
st.caption(f"Last data: {latest}")
if latest < (datetime.date.today() - datetime.timedelta(days=1)):
    st.warning("⚠️ Data may be stale. Re-run preprocess.py.")
```

---

## Step 4 — Panel 1: Signal Hero (`components/signal_hero.py`)

### What to do

The largest, most prominent element. Renders differently based on signal value.

```python
import streamlit as st

SIGNAL_COLORS = {
    "BUY VOL":      ("#00e676", "#0d3b1e"),   # (text, background)
    "SELL VOL":     ("#ff5252", "#3b0d0d"),
    "HOLD":         ("#ffaa00", "#3b2e00"),
    "NO XGB DATA":  ("#888888", "#1a1a1a"),
}

def render_signal_hero(day: pd.Series):
    from data_loader import derive_signal
    signal, confidence = derive_signal(day.to_dict())

    text_color, bg_color = SIGNAL_COLORS.get(signal, ("#ffffff", "#111111"))

    # Final forecast: prefer Final_Forecast, fall back to GARCH_Forecast
    final_fc = day.get("Final_Forecast", float("nan"))
    garch_fc  = day.get("GARCH_Forecast", float("nan"))
    display_fc = final_fc if not pd.isna(final_fc) else garch_fc

    ann_fc = display_fc * (252 ** 0.5) * 100 if not pd.isna(display_fc) else None

    col1, col2, col3 = st.columns([2, 2, 1.5])

    with col1:
        st.markdown(f"""
        <div style="background:{bg_color}; border-radius:12px; padding:24px; text-align:center;">
            <div style="font-size:0.85rem; color:#aaa; letter-spacing:0.1em;">SIGNAL</div>
            <div style="font-size:2.8rem; font-weight:900; color:{text_color};">{signal}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        ann_str = f"{ann_fc:.2f}% ann." if ann_fc else "—"
        daily_str = f"{display_fc*100:.3f}% daily" if not pd.isna(display_fc) else "—"
        st.markdown(f"""
        <div style="background:#1a1a2e; border-radius:12px; padding:24px; text-align:center;">
            <div style="font-size:0.85rem; color:#aaa; letter-spacing:0.1em;">FINAL FORECAST</div>
            <div style="font-size:2.2rem; font-weight:700; color:#4a9eff;">{ann_str}</div>
            <div style="font-size:0.9rem; color:#888;">{daily_str}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        bar_pct = int(confidence * 100)
        st.markdown(f"""
        <div style="background:#1a1a1a; border-radius:12px; padding:24px; text-align:center;">
            <div style="font-size:0.85rem; color:#aaa; letter-spacing:0.1em;">CONFIDENCE</div>
            <div style="font-size:2rem; font-weight:700; color:{text_color};">{bar_pct}%</div>
            <div style="background:#333; border-radius:4px; height:8px; margin-top:8px;">
                <div style="background:{text_color}; width:{bar_pct}%; height:8px; border-radius:4px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
```

### Potential Problems & Solutions

**Problem:** `Final_Forecast` and `XGB_Pred_Probability` are NaN for training-period dates.
**Solution:** The `derive_signal()` function already handles this and returns `"NO XGB DATA"`.
The hero box renders in grey with clear label. The forecast card falls back to GARCH_Forecast.

**Problem:** `unsafe_allow_html=True` custom cards don't respect Streamlit's light/dark theme.
**Solution:** Hardcode dark backgrounds (`#1a1a1a`, `#1a1a2e`) — this dashboard is for personal
use, always use dark theme. Add to `app.py`:
```python
# Force dark theme via config (create .streamlit/config.toml)
# [theme]
# base = "dark"
```
Create `.streamlit/config.toml` at project root with that content.

**Problem:** Annualised vol formula wrong for different vol units.
**Solution:** Check `final_features.parquet` to confirm if `GARCH_Forecast` is stored as
decimal (0.00892) or percentage (0.892). The README shows `0.892%` daily so the value is
already in percentage form — multiply by `sqrt(252)` directly, not by `sqrt(252) * 100`.
Add an assertion in `data_loader.py`:
```python
assert df["GARCH_Forecast"].median() < 5, \
    "GARCH_Forecast looks like it may be in decimal form, not percent — check units"
```

---

## Step 5 — Panel 2: Forecast Breakdown Cards (`components/forecast_cards.py`)

### What to do

Three `st.metric` cards showing the decomposition: GARCH → Correction → Final.

```python
import streamlit as st
import pandas as pd

def render_forecast_cards(day: pd.Series):
    garch    = day.get("GARCH_Forecast", float("nan"))
    corr     = day.get("XGB_Correction", float("nan"))
    final_fc = day.get("Final_Forecast", float("nan"))

    def fmt(v, pct=True):
        if pd.isna(v): return "—"
        return f"{v:.3f}%" if pct else f"{v:+.3f}%"

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "GARCH Forecast",
        fmt(garch),
        help="Raw GARCH(1,1) one-day-ahead vol forecast"
    )
    col2.metric(
        "XGB Correction",
        fmt(corr),
        delta=fmt(corr) if not pd.isna(corr) else None,
        delta_color="normal",
        help="Correction the XGBoost regressor applied on top of GARCH"
    )
    col3.metric(
        "Final Forecast",
        fmt(final_fc) if not pd.isna(final_fc) else fmt(garch),
        help="Final = GARCH + XGB Correction. Falls back to GARCH if no XGB data."
    )
```

### Potential Problems & Solutions

**Problem:** `XGB_Correction` can be negative (SELL VOL case) — `st.metric` delta shows red
arrow for negative which is actually correct behavior here, but may confuse direction.
**Solution:** The delta arrow pointing down for a negative correction is accurate — GARCH is
overestimating, so we subtract. No change needed, the visual is correct.

**Problem:** All three values NaN on a training-period date.
**Solution:** `fmt()` returns `"—"` for NaN. For the Final Forecast card, fall back to showing
`GARCH_Forecast` with a label suffix `"(GARCH only)"` when `Final_Forecast` is NaN.

---

## Step 6 — Panel 3: Forecast vs Realized Chart (`components/forecast_chart.py`)

### What to do

An interactive Plotly line chart over the full history. This is the most important
"is the model working?" visual.

```python
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

def render_forecast_chart(master: pd.DataFrame, selected_date):
    st.subheader("Forecast vs Realized Volatility")

    # Restrict to rows that have at least GARCH data
    df = master.dropna(subset=["GARCH_Forecast"]).copy()

    fig = go.Figure()

    # Realized vol — full history (HV_10 fills gaps where Realized_Vol_proxy is NaN)
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["Realized_Vol_proxy"],
        name="Realized Vol",
        line=dict(color="#f0c040", width=1.5),
        opacity=0.85
    ))

    # GARCH — full history
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["GARCH_Forecast"],
        name="GARCH Forecast",
        line=dict(color="#888888", width=1.5, dash="dash"),
        opacity=0.75
    ))

    # Final (XGB-corrected) — only where available (test period)
    test_df = df.dropna(subset=["Final_Forecast"])
    if not test_df.empty:
        fig.add_trace(go.Scatter(
            x=test_df["date"], y=test_df["Final_Forecast"],
            name="GARCH + XGB",
            line=dict(color="#4a9eff", width=2),
        ))

    # Vertical line for selected date
    fig.add_vline(
        x=str(selected_date),
        line_dash="dot",
        line_color="#ffffff",
        opacity=0.4,
        annotation_text="Selected",
        annotation_position="top"
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

    st.plotly_chart(fig, use_container_width=True)
```

### Potential Problems & Solutions

**Problem:** `Realized_Vol_proxy` is NaN for most of the history (only exists in test CSV).
**Solution:** Handled in `build_master()` in Step 2 — it is filled from `HV_10` for all rows
where it is NaN. This gives a continuous realized vol line for the full chart.

**Problem:** GARCH and Realized vol are on different scales (one is decimal, one is percent).
**Solution:** After loading in `build_master()`, check and normalize:
```python
# If GARCH_Forecast median is below 0.1, it's in decimal form — convert to %
if master["GARCH_Forecast"].median() < 0.1:
    for col in ["GARCH_Forecast", "Final_Forecast", "Realized_Vol_proxy",
                "XGB_Correction", "HV_10", "HV_20", "HV_30"]:
        if col in master.columns:
            master[col] = master[col] * 100
```
Do this normalization once in `build_master()`, never in individual components.

**Problem:** Chart looks cluttered with 250+ trading days of data.
**Solution:** Add a range selector to the chart:
```python
fig.update_xaxes(
    rangeselector=dict(buttons=[
        dict(count=1, label="1M", step="month"),
        dict(count=3, label="3M", step="month"),
        dict(count=6, label="6M", step="month"),
        dict(label="All", step="all"),
    ])
)
```

**Problem:** `add_vline` with a date value crashes if `selected_date` is `datetime.date` not str.
**Solution:** Always convert: `x=pd.Timestamp(selected_date).isoformat()`

---

## Step 7 — Panel 4: Recent Signal Log (`components/signal_log.py`)

### What to do

A compact table of the last 15 days from the test period. Rows where XGB data is available
only — skip training-period rows.

```python
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
        rows.append({
            "Date":       row["date"].strftime("%d %b"),
            "Signal":     signal,
            "Conf":       f"{conf:.0%}",
            "Correct":    correct_str,
            "GARCH Err":  f"{row['GARCH_Residual_lag1']:+.3f}" if pd.notna(row.get("GARCH_Residual_lag1")) else "—",
        })

    display_df = pd.DataFrame(rows).iloc[::-1]  # most recent first

    def color_signal(val):
        if val == "BUY VOL":  return "color: #00e676"
        if val == "SELL VOL": return "color: #ff5252"
        if val == "HOLD":     return "color: #ffaa00"
        return ""

    styled = display_df.style.applymap(color_signal, subset=["Signal"])
    st.dataframe(styled, use_container_width=True, hide_index=True)
```

### Potential Problems & Solutions

**Problem:** `Correct_Direction` is only meaningful on dates where we know realized vol
(i.e. the day has passed). For today's row it will be NaN.
**Solution:** The `"—"` fallback handles this. The table will show `"—"` for today's row
naturally.

**Problem:** `derive_signal()` called in a loop over 15 rows is slightly redundant.
**Solution:** At 15 rows it's negligible. Don't over-optimise. Clarity > micro-performance.

**Problem:** `.applymap` is deprecated in newer pandas (replaced by `.map`).
**Solution:** Use `styled = display_df.style.map(color_signal, subset=["Signal"])`.
Both work in pandas 2.x but `.map` is forward-compatible.

---

## Step 8 — Panel 5: Market Context Strip (`components/context_strip.py`)

### What to do

A single row of 5 compact `st.metric` cards showing today's market inputs.

```python
import streamlit as st
import pandas as pd

def render_context_strip(day: pd.Series):
    st.markdown("#### Today's Market Context")

    def fmt_pct(v, decimals=2):
        if pd.isna(v): return "—"
        return f"{float(v):.{decimals}f}%"

    def fmt_num(v, decimals=2):
        if pd.isna(v): return "—"
        return f"{float(v):.{decimals}f}"

    cols = st.columns(5)

    cols[0].metric("ATM IV",    fmt_pct(day.get("ATM_IV")),
                   help="At-the-money implied volatility today")
    cols[1].metric("Skew",      fmt_pct(day.get("Skew"), 3),
                   help="Put IV (0.95m) minus Call IV (1.05m)")
    cols[2].metric("HV 20",     fmt_pct(day.get("HV_20")),
                   help="20-day historical (realized) volatility")
    cols[3].metric("PCR (OI)",  fmt_num(day.get("PCR_OI")),
                   help="Put-call ratio by open interest")
    cols[4].metric("DTE Near",  f"{int(day['DTE_nearest'])}d" if pd.notna(day.get("DTE_nearest")) else "—",
                   help="Days to nearest expiry")
```

### Potential Problems & Solutions

**Problem:** `ATM_IV` may be stored as decimal (0.142) or percent (14.2) in the parquet.
**Solution:** The unit normalization in `build_master()` (Step 2) handles vol columns.
For `ATM_IV` specifically, add it to the normalization list there. Verify once with
`parquet_viewer.py` before coding this component.

**Problem:** `PCR_OI` can be very large (> 5) on expiry days — looks alarming out of context.
**Solution:** Add a delta vs previous day:
```python
# In data_loader.py, compute prev-day PCR after build_master()
master["PCR_OI_delta"] = master["PCR_OI"].diff()
```
Then pass `delta=fmt_num(day.get("PCR_OI_delta"))` to the metric card.

---

## Step 9 — Assemble `app.py`

### What to do

Wire everything together in the correct render order.

```python
import streamlit as st
import datetime
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
    initial_sidebar_state="expanded"
)

# ── CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetricValue"]  { font-size: 1.6rem !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"]  { font-size: 0.75rem !important; letter-spacing: 0.06em; color: #888; }
[data-testid="stSidebar"]      { background-color: #0c0c14; }
.block-container               { padding-top: 1.2rem; }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────
master = build_master()

if master.empty:
    st.error("Could not load data. Run preprocess.py and xgboost_volatility_model.py first.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────
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

    test_df = load_test_forecasts()
    test_dates = set(test_df["date"].dt.date.tolist())
    if selected_date not in test_dates:
        st.warning("Training-period date — XGB forecast not available.")

    st.caption(f"Data: {available_dates[-1]} → {available_dates[0]}")

    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# ── Filter to selected date ───────────────────────────────
day_row = master[master["date"].dt.date == selected_date]

if day_row.empty:
    st.warning(f"No data for {selected_date}.")
    st.stop()

day = day_row.iloc[0]

# ── Panel 1: Signal Hero ──────────────────────────────────
render_signal_hero(day)
st.divider()

# ── Panel 2: Forecast Cards ───────────────────────────────
render_forecast_cards(day)
st.divider()

# ── Panels 3 + 4: Chart | Signal Log ─────────────────────
col_left, col_right = st.columns([6, 4])
with col_left:
    render_forecast_chart(master, selected_date)
with col_right:
    render_signal_log(master, selected_date)

st.divider()

# ── Panel 5: Context Strip ────────────────────────────────
render_context_strip(day)
```

### Potential Problems & Solutions

**Problem:** `build_master()` is called at the top level before sidebar renders — if the
files are missing, `st.error` fires immediately with no sidebar visible.
**Solution:** This is fine and intentional — you can't use the dashboard without data.
The error message should say exactly which files are missing.

**Problem:** `st.cache_data.clear()` on the Refresh button clears ALL caches, causing a
full reload of both parquet files on the next interaction.
**Solution:** This is the correct behavior — it is what a manual refresh should do. The
reload takes 2–3 seconds at most.

---

## Step 10 — Create `.streamlit/config.toml`

### What to do

Create this file at the project root (same level as `vol_dashboard/`):

```toml
[theme]
base = "dark"
backgroundColor = "#0e0e18"
secondaryBackgroundColor = "#1a1a2e"
textColor = "#e0e0e0"
primaryColor = "#4a9eff"
```

This ensures the dark aesthetic is consistent and doesn't depend on the user's system theme.

### Potential Problems & Solutions

**Problem:** `.streamlit/config.toml` path is not found when running from a subdirectory.
**Solution:** Always run `streamlit run vol_dashboard/app.py` from the project root, where
`.streamlit/` lives. Never `cd` into `vol_dashboard/` and run from there.

---

## Step 11 — Testing Checklist

Run through this before calling the dashboard done:

| Test | What to verify |
|---|---|
| Latest date is default | Open app, date picker shows most recent trading day |
| BUY VOL renders green, SELL VOL red, HOLD amber | Switch between dates with known signals |
| Training-period date shows "NO XGB DATA" hero | Pick any date before TRAIN_END (2025-12-31) |
| Forecast cards show "—" for XGB cols on training date | Same test as above |
| Chart shows GARCH line for full history | Should run from earliest parquet date |
| Chart shows blue GARCH+XGB line only from test period | Visual check on the chart |
| Realized vol line is continuous (no big gaps) | HV_10 filling NaN is working |
| Signal log shows ✅/❌ for past days, "—" for today | Check last row of table |
| Context strip shows all 5 values, no NaN crashes | Pick several dates including edge dates |
| Refresh button clears cache and reloads cleanly | Click it, verify no error |
| Warning shows for stale data | Manually set system date forward one day and check |

---

## Common Global Problems

**Problem:** Vol units inconsistency — some columns in percent, some in decimal.
**Solution:** The unit normalization block in `build_master()` (Step 2) is the single fix
point. Check parquet with `parquet_viewer.py` before coding, confirm once, normalize once.

**Problem:** Dashboard runs fine locally but crashes when path to project root changes.
**Solution:** `ROOT = Path(__file__).resolve().parent.parent` is the only safe way.
Never use `os.getcwd()` or hardcoded strings.

**Problem:** Streamlit re-renders everything on every widget interaction.
**Solution:** All heavy I/O is behind `@st.cache_data`. The only code that runs on every
interaction is the filtering (`day_row = master[...]`) and the component render functions,
which are fast pandas operations on already-loaded data.

---

## Final File Checklist

```
vol_dashboard/
├── app.py                      ✓ entry point, layout, sidebar, CSS
├── data_loader.py              ✓ all loaders, build_master(), derive_signal(), unit normalization
├── components/
│   ├── signal_hero.py          ✓ BUY VOL / SELL VOL / HOLD with confidence bar
│   ├── forecast_cards.py       ✓ GARCH / Correction / Final metric row
│   ├── forecast_chart.py       ✓ full-history line chart, range selector, vline
│   ├── signal_log.py           ✓ last 15 days table with ✅/❌
│   └── context_strip.py        ✓ ATM IV / Skew / HV20 / PCR / DTE row
├── utils.py                    ✓ vol formatting helpers (optional, if shared across components)
└── requirements.txt            ✓ streamlit, pandas, plotly, pyarrow, numpy, xgboost, scikit-learn

.streamlit/
└── config.toml                 ✓ dark theme config (lives at project root, not inside vol_dashboard/)
```

Run with:
```bash
# Always from project root
streamlit run vol_dashboard/app.py
```
