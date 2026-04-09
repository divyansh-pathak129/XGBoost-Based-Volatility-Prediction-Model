---

### **What We Know From the Files**

**New formatted CSV (`option_chain_BANKNIFTY-28-Apr-2026.csv`):**
- Already has 21 columns, properly named — same structure `daily_predict.py` expects internally
- Strike column has comma-formatted strings like `"54,600.00"` — needs cleaning
- Has all columns: `call_oi`, `call_iv`, `call_volume`, `put_oi`, `put_iv`, `put_volume` etc.
- **Filename encodes asset (`BANKNIFTY`) and expiry date (`28-Apr-2026`)** — this is valuable
- BANKNIFTY spot appears to be around **55,000–57,000** based on where IVs are symmetric

**`option_data_formating.py`:**
- Takes raw NSE file, renames to 21 columns, outputs to `option_chain_processed/` folder
- Filename format: `option_chain_{ASSET}-{DATE}.csv`
- Does **not** clean NaN values or fix the strike column yet

**`daily_predict.py`:**
- `parse_option_chain()` expects raw NSE format → **needs to be replaced** with a parser for your new format
- `get_ts_slope()` needs an `EXPIRY` column → we can inject it from the filename
- Spot auto-detection will fail → we need to derive it from the chain itself
- Everything else (GARCH, HV, features, models) is intact and correct

---

### **Step 1 — Extend `option_data_formating.py`: Add Cleaning Stage**

**Add this block at the end of the script, after `df = df.sort_values(by='strike')` and before saving:**

```python
# ── CLEANING STAGE ──────────────────────────────────────────────────

# Step 1A: Clean strike — remove commas, convert to numeric int
df['strike'] = (
    df['strike'].astype(str)
    .str.replace(',', '', regex=False)
    .str.replace('.00', '', regex=False)
)
df['strike'] = pd.to_numeric(df['strike'], errors='coerce')
df = df.dropna(subset=['strike'])
df['strike'] = df['strike'].astype(int)

# Step 1B: Interpolate IV columns — vol smile is smooth, interpolation is valid
# Sort by strike before interpolating so the curve is monotonic
df = df.sort_values('strike').reset_index(drop=True)
df['call_iv'] = df['call_iv'].interpolate(method='linear', limit_direction='both')
df['put_iv']  = df['put_iv'].interpolate(method='linear', limit_direction='both')

# Step 1C: Interpolate LTP columns
df['call_ltp'] = df['call_ltp'].interpolate(method='linear', limit_direction='both')
df['put_ltp']  = df['put_ltp'].interpolate(method='linear', limit_direction='both')

# Step 1D: Fill OI and volume NaN with 0 — missing = no open position/trade
oi_vol_cols = ['call_oi', 'call_chng_oi', 'call_volume',
               'put_oi',  'put_chng_oi',  'put_volume']
df[oi_vol_cols] = df[oi_vol_cols].fillna(0)

# Step 1E: Drop strikes where both call_iv AND put_iv are still NaN after interpolation
# These are extreme deep OTM strikes with truly no data on either side
df = df.dropna(subset=['call_iv', 'put_iv'], how='all')

# Step 1F: Add metadata columns extracted from filename
# Filename format: option_chain_{ASSET}-{DATE}.csv → asset and expiry already in parts[]
df['asset']       = asset_name         # e.g. 'BANKNIFTY'
df['expiry_date'] = date               # e.g. '28-Apr-2026'

# Step 1G: Print quality report
print(f"\n--- Cleaning Report ---")
print(f"Total strikes after cleaning : {len(df)}")
print(f"Strike range                 : {df['strike'].min()} to {df['strike'].max()}")
print(f"Strikes with call_iv         : {df['call_iv'].notna().sum()}")
print(f"Strikes with put_iv          : {df['put_iv'].notna().sum()}")
print(f"Asset                        : {df['asset'].iloc[0]}")
print(f"Expiry date in file          : {df['expiry_date'].iloc[0]}")
```

**Potential problems and fixes:**

| Problem | Fix |
|---|---|
| Strike like `"43,000.00"` — both comma and `.00` | Strip comma first, then `.00`, then convert to int |
| Extreme edge strikes (deep OTM) still NaN after interpolation | `limit_direction='both'` extrapolates from nearest known point — covers this |
| `date` variable in the script is a string like `"28-Apr-2026"` — conflicts with Python's `date` type if imported | Don't import `datetime.date` in this script — keep `date` as a plain string variable |

---

### **Step 2 — Replace `parse_option_chain()` in `daily_predict.py`**

The current `parse_option_chain()` expects raw NSE format (3 merged header rows). Your new formatted file already has clean column names in row 1. **Replace the entire function** with this:

```python
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
```

**Also update `try_read_spot_from_file()`** — it currently scans header rows for a number in BANKNIFTY range, but your file has no such header. Replace it with spot estimation from the chain:

```python
def try_read_spot_from_file(filepath: str) -> float | None:
    """
    Estimate spot price from the formatted chain using put-call parity.
    ATM strike = strike where |call_iv - put_iv| is minimum.
    Falls back to None if insufficient IV data.
    """
    try:
        df = pd.read_csv(filepath)
        df['strike'] = pd.to_numeric(
            df['strike'].astype(str).str.replace(',','').str.replace('.00',''),
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
```

**Potential problems and fixes:**

| Problem | Fix |
|---|---|
| File was saved without running cleaning step — strike still has `"43,000.00"` | The `parse_option_chain()` above re-cleans strike defensively, so it works either way |
| `EXPIRY` column not in file (cleaning step not run yet) | `get_ts_slope()` already handles missing EXPIRY with fallback to 0.0 — no crash |
| Spot auto-detection gives wrong strike if IV data is sparse near ATM | Falls back to `top OI strikes median` already in `main()` — safe |

---

### **Step 3 — Use Expiry Date From Filename in `main()`**

The filename `option_chain_BANKNIFTY-28-Apr-2026.csv` contains the expiry date. After parsing the chain, extract it and pass it to `calendar_features()` so `DTE_nearest` is accurate.

**Add this block in `main()` right after `chain = parse_option_chain(args.chain)`:**

```python
# ── Extract expiry date from chain metadata or filename ──────────────
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
        expiry_from_chain = datetime.strptime(date_part, "%d-%b-%Y").date()
        print(f"  Expiry from filename: {expiry_from_chain}")
    except Exception:
        print("  WARNING: Could not parse expiry from filename")

# Inject into expiry_dates list so calendar features use the correct DTE
if expiry_from_chain and not args.next_expiry:
    args.next_expiry = str(expiry_from_chain)
```

**Potential problems and fixes:**

| Problem | Fix |
|---|---|
| Filename date format changes (e.g. `28-April-2026` vs `28-Apr-2026`) | Try both `%d-%b-%Y` and `%d-%B-%Y` format strings in a try/except chain |
| Asset name is not BANKNIFTY (future use for NIFTY etc.) | `asset_name` is already extracted from filename in `option_data_formating.py` — can pass via `ASSET` column |

-
**Daily workflow — run these 3 commands in order:**

```bash
# 1. Format and clean the raw option chain
python option_data_formating.py option_chain-fo-28-APR2026-BANKNIFTY.csv

# 2. Update GARCH and HV features from latest spot data
python preprocess.py

# 3. Run prediction
python daily_predict.py --chain data/option_chain_processed/option_chain_BANKNIFTY-28-Apr-2026.csv
```

No `--spot` needed — auto-detected from put-call parity. No `--next-expiry` needed — parsed from filename.

**First time only:**
```bash
pip install pandas numpy scipy arch xgboost scikit-learn shap joblib pyarrow openpyxl
python xgboost_volatility_model.py    # trains and saves models
```

---

### **Full Change Summary for Claude Code**

| File | Action | Exact Location | What to Add/Change |
|---|---|---|---|
| `option_data_formating.py` | **Add** cleaning block | After `df = df.sort_values(by='strike')`, before `df.to_csv(...)` | Steps 1A through 1G above |
| `daily_predict.py` | **Replace** `parse_option_chain()` | Lines 55–125 | New function from Step 2 |
| `daily_predict.py` | **Replace** `try_read_spot_from_file()` | Lines 128–148 | New parity-based spot estimator from Step 2 |
| `daily_predict.py` | **Add** expiry extraction block | In `main()`, right after `chain = parse_option_chain(args.chain)` | 15-line block from Step 3 |