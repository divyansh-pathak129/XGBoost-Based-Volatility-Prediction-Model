The file structure looks quite messy — this is a **multi-header NSE option chain HTML/Excel export** which is notoriously tricky to parse. Let me design the full plan accounting for this exact format.

---

## Full Plan: Daily Prediction System

---

### **Phase 0 — Parse Today's Option Chain File**

This is the hardest step. NSE's option chain has **merged headers** (CALLS | STRIKE | PUTS) across 3 rows, which pandas reads as a MultiIndex mess — exactly what you're seeing.

**What the raw file looks like:**
```
Row 1: CALLS          |  STRIKE  | PUTS
Row 2: OI | IV | LTP  |          | OI | IV | LTP ...
Row 3: actual data...
```

**Solution — custom parser:**
```python
import pandas as pd

def parse_option_chain(filepath):
    # Skip the merged header rows, read raw
    df = pd.read_excel(filepath, header=None, skiprows=3)
    
    # Manually assign column names based on NSE's fixed structure
    call_cols = ['C_OI','C_CHNG_OI','C_VOLUME','C_IV',
                 'C_LTP','C_CHNG','C_BID_QTY','C_BID',
                 'C_ASK','C_ASK_QTY']
    put_cols  = ['P_BID_QTY','P_BID','P_ASK','P_ASK_QTY',
                 'P_CHNG','P_LTP','P_IV','P_VOLUME',
                 'P_CHNG_OI','P_OI']
    
    df.columns = call_cols + ['STRIKE'] + put_cols
    
    # Clean commas and convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(',',''), 
            errors='coerce'
        )
    
    return df.dropna(subset=['STRIKE'])
```

**Problems that can arise:**
- NSE occasionally adds/removes a column → fix by printing `df.shape` and checking column count is 21
- `-` values for no-trade strikes → handled by `errors='coerce'` which converts to NaN
- CSV vs Excel format → check extension and use `read_csv` or `read_excel` accordingly

---

### **Phase 1 — Extract Core Features from Option Chain**

Once parsed, compute the features your model needs from the chain.

**Step 1.1 — Find ATM Strike**
```python
def get_atm_strike(df, spot_price):
    df['dist'] = abs(df['STRIKE'] - spot_price)
    atm_idx = df['dist'].idxmin()
    return df.loc[atm_idx, 'STRIKE'], atm_idx
```

**Step 1.2 — ATM IV**
```python
atm_strike, atm_idx = get_atm_strike(df, spot_price)
atm_iv = (df.loc[atm_idx, 'C_IV'] + df.loc[atm_idx, 'P_IV']) / 2
```

**Step 1.3 — Skew (Put IV at 0.95 spot minus Call IV at 1.05 spot)**
```python
otm_put_strike  = round(spot_price * 0.95 / 100) * 100
otm_call_strike = round(spot_price * 1.05 / 100) * 100

put_iv  = df[df['STRIKE']==otm_put_strike]['P_IV'].values[0]
call_iv = df[df['STRIKE']==otm_call_strike]['C_IV'].values[0]
skew = put_iv - call_iv
```

**Step 1.4 — PCR, OI Change, Volume**
```python
total_call_oi = df['C_OI'].sum()
total_put_oi  = df['P_OI'].sum()
pcr_oi        = total_put_oi / total_call_oi

total_call_vol = df['C_VOLUME'].sum()
total_put_vol  = df['P_VOLUME'].sum()
pcr_volume     = total_put_vol / total_call_vol
```

**Problems that can arise:**
- OTM strike not present in chain (deep OTM not listed) → find nearest available strike within 2% band instead
- ATM IV is NaN if no trade happened at that strike → fallback to average of ±1 strike IVs
- PCR division by zero if calls have no OI → add `+ 1e-6` to denominator

---

### **Phase 2 — Compute Term Structure Slope**

Your model needs `TS_Slope` = far expiry IV minus near expiry IV. This requires option chain data for **2 different expiries**.

**Option A (ideal):** Download two expiry chains from NSE — current week + next month expiry.

**Option B (fallback):** NSE's combined chain sometimes has both expiries. Filter by expiry date:
```python
# If your chain has an EXPIRY column
near = df[df['EXPIRY'] == nearest_expiry]
far  = df[df['EXPIRY'] == next_expiry]

ts_slope = far_atm_iv - near_atm_iv
```

**Problems:**
- Only one expiry available → set `TS_Slope = 0` as neutral fallback, flag it in output
- Different strikes available per expiry → always anchor to ATM of that expiry's spot

---

### **Phase 3 — Load Historical Data and Compute GARCH + HV Features**

These features come from your **existing pipeline**, not the option chain.

```python
# Load your existing processed features (yesterday's row)
features_history = pd.read_parquet("data/features/final_features.parquet")

# Get lagged values from yesterday
atm_iv_lag1   = features_history['ATM_IV'].iloc[-1]
atm_iv_lag2   = features_history['ATM_IV'].iloc[-2]
atm_iv_5d     = features_history['ATM_IV'].iloc[-5:].mean()
skew_lag1     = features_history['Skew'].iloc[-1]
oi_change_lag1= features_history['OI_Change'].iloc[-1]
garch_res_lag1= features_history['GARCH_Residual_lag1'].iloc[-1]
garch_res_lag2= features_history['GARCH_Residual_lag2'].iloc[-1]

# Run GARCH on latest spot returns to get today's forecast
# (your preprocess.py already does this — just re-run it)
```

**Problems:**
- `final_features.parquet` is stale (you haven't run preprocess today) → always run `preprocess.py` first before prediction
- GARCH warmup needs 60 days minimum → if your history is short, GARCH forecast will be unreliable; print a warning if history < 90 rows

---

### **Phase 4 — Compute Calendar Features**

```python
from datetime import date

today = date.today()

# Days to nearest expiry (BANKNIFTY expires every Thursday)
def days_to_nearest_expiry(today, expiry_dates):
    future = [e for e in expiry_dates if e >= today]
    return (min(future) - today).days

dte_nearest         = days_to_nearest_expiry(today, your_expiry_list)
is_expiry_week      = 1 if dte_nearest <= 5 else 0
days_since_expiry   = (today - last_expiry_date).days
```

**Problems:**
- Hardcoding expiry dates will go stale → maintain a rolling list; NSE publishes expiry calendars yearly; add a simple CSV `expiry_dates.csv` to your repo and update quarterly

---

### **Phase 5 — Assemble the Feature Row and Predict**

```python
import numpy as np
import xgboost as xgb

FEATURE_COLS = [
    "GARCH_Forecast", "GARCH_Residual_lag1", "GARCH_Residual_lag2",
    "ATM_IV", "ATM_IV_lag1", "ATM_IV_lag2", "ATM_IV_5d_mean",
    "Skew", "Skew_lag1", "TS_Slope", "IV_HV_Spread",
    "OI_Change", "OI_Change_lag1", "PCR_OI", "Volume_Change", "PCR_Volume",
    "DTE_nearest", "Is_expiry_week", "Days_since_last_expiry",
    "HV_10", "HV_20", "HV_30",
]

today_row = pd.DataFrame([{
    "GARCH_Forecast":       garch_forecast,
    "GARCH_Residual_lag1":  garch_res_lag1,
    "GARCH_Residual_lag2":  garch_res_lag2,
    "ATM_IV":               atm_iv,
    "ATM_IV_lag1":          atm_iv_lag1,
    "ATM_IV_lag2":          atm_iv_lag2,
    "ATM_IV_5d_mean":       atm_iv_5d,
    "Skew":                 skew,
    "Skew_lag1":            skew_lag1,
    "TS_Slope":             ts_slope,
    "IV_HV_Spread":         atm_iv - hv_20,
    "OI_Change":            oi_change_today,
    "OI_Change_lag1":       oi_change_lag1,
    "PCR_OI":               pcr_oi,
    "Volume_Change":        vol_change_today,
    "PCR_Volume":           pcr_volume,
    "DTE_nearest":          dte_nearest,
    "Is_expiry_week":       is_expiry_week,
    "Days_since_last_expiry": days_since_expiry,
    "HV_10":                hv_10,
    "HV_20":                hv_20,
    "HV_30":                hv_30,
}])[FEATURE_COLS]

# Load models and predict
clf = xgb.XGBClassifier(); clf.load_model("models/xgb_classifier.ubj")
reg = xgb.XGBRegressor();  reg.load_model("models/xgb_regressor.ubj")

prob        = clf.predict_proba(today_row)[0][1]
pred_class  = 1 if prob > 0.5 else 0
correction  = reg.predict(today_row)[0]
final_vol   = garch_forecast + correction
```

---

### **Phase 6 — Generate the Daily Signal Report**

```python
direction = "BUY VOL  🔺" if pred_class == 1 else "SELL VOL 🔻"
confidence = prob if pred_class == 1 else 1 - prob
trade_flag = "✅ ACT"  if confidence > 0.62 else "⏸ SKIP"

print(f"""
========================================
   BANKNIFTY VOL SIGNAL — {today}
========================================
GARCH Forecast:      {garch_forecast*np.sqrt(252)*100:.2f}% annualized
XGB Correction:      {correction*np.sqrt(252)*100:+.2f}%
Final Forecast:      {final_vol*np.sqrt(252)*100:.2f}% annualized

Signal:              {direction}
Confidence:          {confidence:.1%}
Trade Flag:          {trade_flag}

Supporting Signals:
  ATM IV today:      {atm_iv*100:.2f}%
  Skew:              {skew:+.4f}  (positive = puts expensive)
  PCR OI:            {pcr_oi:.2f}
  DTE nearest:       {dte_nearest} days
  Expiry week:       {"YES" if is_expiry_week else "NO"}
========================================
""")
```

---

### **Daily Workflow (End-to-End)**

```
4:00 PM — Market closes
  ↓
4:15 PM — Download NSE BANKNIFTY option chain 
           (nseindia.com → Option Chain → Download CSV)
  ↓
Run: python preprocess.py          ← updates GARCH + HV features
Run: python daily_predict.py       ← your new wrapper script
  ↓
Read the signal report
  ↓
Next morning — place trade at open if flag = ACT
```

---

### **Summary of All Potential Problems**

| Step | Problem | Fix |
|---|---|---|
| Phase 0 | Multi-header parse fails | Use `header=None, skiprows=3`, assign columns manually |
| Phase 1 | ATM strike has NaN IV | Average ±1 strike IVs |
| Phase 1 | OTM strike missing | Find nearest within 2% band |
| Phase 2 | Only one expiry available | Set TS_Slope=0, flag in output |
| Phase 3 | Stale parquet file | Always run preprocess.py first |
| Phase 4 | Expiry dates go stale | Maintain expiry_dates.csv, update quarterly |
| Phase 5 | Feature mismatch / wrong column order | Always select via `FEATURE_COLS` list |
| All | Model confidence near 0.5 | Skip trade — no edge, don't force it |

---

