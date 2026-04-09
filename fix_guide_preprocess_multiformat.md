

---

**PLAN: Fix preprocess.py for multi-format dataset compatibility**

This plan is for fixing `preprocess.py` in the BANKNIFTY volatility forecasting pipeline. The pipeline currently works with Dataset1 format files (e.g. `BANK_NIFTY_AUG25.xlsx`) but crashes when Dataset2 format files (e.g. `BANK_NIFTY_April2023.xlsx`, `BANK_NIFTY_Dec2023.xlsx`) are added to the `BANK_NIFTY_Data1/` folder.

**Note on folder name:** The README says data goes in `BANK_NIFTY_Data1/` but the config at the top of `preprocess.py` has `DATA_DIR = "BANKNIFTY"`. These must match. Confirm which folder the files are actually in and make sure `DATA_DIR` points to the right place. This is not a Dataset1 vs Dataset2 problem — it's a pre-existing inconsistency. Fix it first before running anything.

---

**CHANGE 1 — Fix the crash in `fix_date_column` (CRITICAL)**

This is the only thing that causes the actual crash. The current function assumes the `Date` column is always numeric if it's not already datetime64. Dataset1 files have `Date` as plain strings like `"01-08-2025"`. When pandas reads these, `.max()` on a string Series returns a string, and `string <= 9999999` throws:

```
TypeError: '<=' not supported between instances of 'str' and 'int'
```

Replace the entire `fix_date_column` function with this:

```python
def fix_date_column(df, fname):
    """Handle all date formats found across NSE monthly files:
    - datetime64         : already parsed by pandas (ideal case)
    - string DD-MM-YYYY  : e.g. "01-08-2025"  — Dataset1 newer files
    - string DD/MM/YYYY  : e.g. "01/08/2025"  — alternate string variant
    - int/float DDMMYY   : e.g. 10226         — NSE FEB26 quirk
    - float Excel serial : e.g. 45139.0       — Dataset2 older files
    """
    col = df["Date"]

    # Already datetime — nothing to do
    if pd.api.types.is_datetime64_any_dtype(col):
        return df

    non_null = col.dropna()
    if len(non_null) == 0:
        print(f"  [{fname}] WARNING: Date column is entirely null!")
        return df

    sample_val = non_null.iloc[0]

    # Branch 1: string dates — check type BEFORE any numeric comparison
    if isinstance(sample_val, str):
        for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d-%b-%Y"):
            parsed = pd.to_datetime(col, format=fmt, errors="coerce")
            if parsed.notna().mean() > 0.8:
                df["Date"] = parsed
                print(f"  [{fname}] Parsed string dates (format='{fmt}'). "
                      f"Sample: {df['Date'].dropna().iloc[0].date()}")
                return df
        # Fallback: let pandas infer
        parsed = pd.to_datetime(col, errors="coerce")
        df["Date"] = parsed
        sample = df["Date"].dropna()
        print(f"  [{fname}] Parsed string dates via auto-infer. "
              f"Sample: {sample.iloc[0].date() if len(sample) else 'N/A'}")
        return df

    # Branch 2: numeric dates — now safe to call .max()
    try:
        numeric_max = float(non_null.max())
    except (TypeError, ValueError):
        df["Date"] = pd.to_datetime(col, errors="coerce")
        print(f"  [{fname}] Date fallback parse (unknown type).")
        return df

    # Sub-branch 2a: DDMMYY 6-digit integer (max <= 9,999,999)
    if numeric_max <= 9_999_999:
        try:
            date_strs = (
                col.astype(float).astype("Int64").astype(str).str.zfill(6)
            )
            test = pd.to_datetime(date_strs, format="%d%m%y", errors="coerce")
            if test.notna().mean() > 0.8:
                df["Date"] = test
                print(f"  [{fname}] Converted DDMMYY integer dates. "
                      f"Sample: {df['Date'].dropna().iloc[0].date()}")
                return df
        except Exception:
            pass

    # Sub-branch 2b: Excel serial float (values > 40,000 = year 2009+)
    if numeric_max > 40_000:
        try:
            parsed = pd.to_datetime(
                col, unit="D", origin="1899-12-30", errors="coerce"
            )
            if parsed.notna().mean() > 0.8:
                df["Date"] = parsed
                print(f"  [{fname}] Converted Excel serial dates. "
                      f"Sample: {df['Date'].dropna().iloc[0].date()}")
                return df
        except Exception as e:
            print(f"  [{fname}] Excel serial conversion failed: {e}")

    # Final fallback
    df["Date"] = pd.to_datetime(col, errors="coerce")
    sample = df["Date"].dropna()
    print(f"  [{fname}] Date generic fallback. "
          f"Sample: {sample.iloc[0].date() if len(sample) else 'N/A'}")
    return df
```

**Potential problem:** `infer_datetime_format=True` is deprecated in pandas 2.x. The replacement above uses plain `pd.to_datetime(col, errors="coerce")` which auto-infers in pandas 2.x without the warning. If you're on pandas 1.x, this still works fine.

---

**CHANGE 2 — Add per-file date validation inside `load_all_files`**

This catches silent failures where a file's dates parsed to NaT in bulk — no crash, but the rows get silently dropped later with no clear signal about which file caused it.

Find this block inside `load_all_files`:

```python
    # Drop rows with NaN in critical columns
    master = master.dropna(subset=["Date", "UNDRLNG_ST"])
    master = master[master["CLOSE_PRIC"].notna() | master["SETTLEMENT"].notna()]
```

Add the following block immediately after those two lines:

```python
    # Per-file date range check — catches silent parse failures
    date_summary = (
        master.groupby("source_file")["Date"]
        .agg(["min", "max", "count"])
        .rename(columns={"min": "earliest", "max": "latest", "count": "valid_rows"})
    )
    print("\nDate range per source file (verify these look correct):")
    print(date_summary.to_string())
    bad_files = date_summary[date_summary["valid_rows"] < 100]
    if len(bad_files):
        print(f"\nWARNING: These files produced suspiciously few valid dates "
              f"(date parsing likely failed): {bad_files.index.tolist()}")
```

**Potential problem:** A file shows `valid_rows = 0` in this table. That means date parsing completely failed for it. Look at what `fix_date_column` printed for that file to see which branch it tried, then open the file in Excel to check the actual raw Date column values. Add a new branch to `fix_date_column` for that format.

---

**CHANGE 3 — Add a print statement for volume column detection in `build_daily_features`**

The volume column is already hardcoded as `TRADED_QUA` on line 408:

```python
total_vol = df.groupby("Date")["TRADED_QUA"].sum().rename("Total_Volume")
```

Both dataset formats use `TRADED_QUA` so there is no bug here. But if a future file uses a different name it will throw a `KeyError`. Make it resilient with one small change — find those four lines that reference `TRADED_QUA` in `build_daily_features` (lines 408–411) and replace them:

```python
    # Detect volume column name robustly
    _VOL_CANDIDATES = ["TRADED_QUA", "TRADED_QTY", "VOLUME", "TRDNG_VALUE"]
    _vol_col = next((c for c in _VOL_CANDIDATES if c in df.columns), None)
    if _vol_col is None:
        print(f"  WARNING: No volume column found. Available: {list(df.columns)}")
        print(f"  PCR_Volume and Volume_Change will be NaN.")
        df["_vol_tmp"] = np.nan
        _vol_col = "_vol_tmp"
    else:
        print(f"  Volume column detected: '{_vol_col}'")

    total_vol = df.groupby("Date")[_vol_col].sum().rename("Total_Volume")
    put_vol = df[df["option_type"] == "PE"].groupby("Date")[_vol_col].sum().rename("Put_Volume")
    call_vol = df[df["option_type"] == "CE"].groupby("Date")[_vol_col].sum().rename("Call_Volume")
    pcr_vol = (put_vol / call_vol.replace(0, np.nan)).rename("PCR_Volume")
    vol_change = total_vol.diff().rename("Volume_Change")
```

**Potential problem:** If `_vol_col` ends up being `"_vol_tmp"` (all NaN), the features `PCR_Volume` and `Volume_Change` will be NaN across the board. The model will train but those two features will be useless. This is acceptable degraded behavior — it is far better than a crash. The warning message will tell you what to fix.

---

**CHANGE 4 — Decide on date range scope and implement it**

Adding Dataset2 files (2023) to the folder means the pipeline will now produce a dataset spanning 2023–2026 instead of just the 2025–2026 window. This directly affects the GARCH warmup, the train/test split dates in `xgboost_volatility_model.py`, and the README's description.

You have two options. Pick one:

**Option A — Keep only the intended study period (recommended if you want clean comparability)**

Add this block in `main()`, after `master = compute_dte(master)` and before `master = apply_filters(master)`:

```python
    # Scope to study period — change these dates to match your intended window
    STUDY_START = pd.Timestamp("2025-04-01")
    STUDY_END   = pd.Timestamp("2026-03-31")
    before = len(master)
    master = master[(master["Date"] >= STUDY_START) & (master["Date"] <= STUDY_END)]
    print(f"\nDate-range filter ({STUDY_START.date()} to {STUDY_END.date()}): "
          f"kept {len(master):,} rows, dropped {before - len(master):,}")
    if len(master) == 0:
        raise ValueError("Date-range filter removed ALL rows. "
                         "Check that your files contain data in the specified window.")
```

Use this option if the 2023 data was added only for HV warmup purposes and you don't want 2023 contracts influencing GARCH or XGBoost training.

**Option B — Use all data (2023–2026) for richer training (recommended if more data helps)**

Do not add any date-range filter. Instead, update `TRAIN_END` and `VAL_END` in `xgboost_volatility_model.py` to reflect the new date range. The current defaults are:

```python
TRAIN_END = "2025-12-31"
VAL_END   = "2026-01-31"
```

With 2023 data added, change them to something like:

```python
TRAIN_END = "2025-09-30"   # adjust based on your actual date range
VAL_END   = "2026-01-31"
```

The key invariant to preserve: train set must end well before the test period begins. Review the actual date range from the date summary table (Change 2) and set these accordingly.

**Potential problem with Option B:** The GARCH rolling window warmup is `GARCH_WARMUP = 60` trading days. With the dataset now starting in 2023, the first 60 days are 2023 warmup rows, which get dropped at the end of `main()` via `final = daily.dropna(subset=["GARCH_Forecast"])`. This is correct behavior. No code change needed, but be aware that the effective start of your GARCH outputs shifts.

---

**HOW TO VERIFY IT WORKED**

After making all changes, run:

```bash
python preprocess.py
```

Check these things in the console output:

1. Every file should print one of: `Parsed string dates`, `Converted DDMMYY integer dates`, `Converted Excel serial dates`, or `Date generic fallback`. No `TypeError`.

2. The date range table should show sensible calendar dates for every file (not `1899`, `1970`, or `NaT`).

3. The line `Volume column detected: 'TRADED_QUA'` should appear.

4. Final line should read something like `Final dataset shape: (N, 26)` where N is greater than before (more rows from 2023 data) if you chose Option B, or roughly the same if Option A.

Then quick-check the output:

```python
import pandas as pd
df = pd.read_parquet("data/features/final_features.parquet")
print(df.shape)
print(df.index.min(), df.index.max())
print(df[["ATM_IV", "GARCH_Forecast", "PCR_Volume"]].isna().mean())
```

`PCR_Volume` NaN rate should be well under 50%. If it's 100%, the volume column was not found — check the warning message and add the correct column name to `_VOL_CANDIDATES`.

`GARCH_Forecast` NaN rate will be nonzero (the warmup rows are expected to be NaN before they get dropped). After `dropna(subset=["GARCH_Forecast"])` in `main()`, the final file should have 0 NaN in `GARCH_Forecast`.

---

**SUMMARY OF ALL CHANGES**

| Change | Location | Lines affected | Why |
|--------|----------|---------------|-----|
| 1 | `fix_date_column` | Full function replacement (~40 lines) | Fix TypeError crash on string dates |
| 2 | `load_all_files` | Insert ~10 lines after `dropna` | Catch silent date parse failures per file |
| 3 | `build_daily_features` | Replace ~4 lines, add ~10 lines | Robust volume column detection |
| 4 | `main()` | Insert ~8 lines (Option A) OR edit `xgboost_volatility_model.py` constants (Option B) | Correct date scope for the combined dataset |

Nothing else changes. All IV computation, GARCH rolling, lag feature engineering, and target creation logic stays exactly as-is.