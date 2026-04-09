import pandas as pd
import os
import sys

# === CONFIG PATHS ===
RAW_FOLDER = r'C:\Users\Aditya\Downloads\code\Stock-Volatility-Preprocessing-Pipeline\data\option_chain_raw'
PROCESSED_FOLDER = r'C:\Users\Aditya\Downloads\code\Stock-Volatility-Preprocessing-Pipeline\data\option_chain_processed'

# === GET FILE NAME FROM TERMINAL ===
if len(sys.argv) < 2:
    print("Usage: python script.py <filename>")
    sys.exit(1)

input_filename = sys.argv[1]

# Full input path
file_path = os.path.join(RAW_FOLDER, input_filename)

# === EXTRACT ASSET & DATE ===
name_without_ext = input_filename.replace('.csv', '')
parts = name_without_ext.split('-')

asset_name = parts[3]
date = '-'.join(parts[4:])

# Output filename
output_filename = f'option_chain_{asset_name}-{date}.csv'
output_path = os.path.join(PROCESSED_FOLDER, output_filename)

# === LOAD DATA ===
df = pd.read_csv(file_path, header=1)

# Drop empty columns
df = df.dropna(axis=1, how='all')

# Rename columns
df.columns = [
    "call_oi", "call_chng_oi", "call_volume", "call_iv", "call_ltp",
    "call_net_chng", "call_bid_qty", "call_bid_price",
    "call_ask_price", "call_ask_qty",
    "strike",
    "put_bid_qty", "put_bid_price", "put_ask_price", "put_ask_qty",
    "put_net_chng", "put_ltp", "put_iv", "put_volume",
    "put_chng_oi", "put_oi"
]

# Replace '-' with NaN
df = df.replace('-', pd.NA)

# Clean numeric columns
numeric_cols = df.columns.drop('strike')

df[numeric_cols] = (
    df[numeric_cols]
    .astype(str)
    .apply(lambda col: col.str.replace(',', '', regex=False))
)

df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Drop rows without strike
df = df.dropna(subset=['strike'])

# Sort (optional)
df = df.sort_values(by='strike')

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

# === SAVE OUTPUT ===
df.to_csv(output_path, index=False)

print(f"\nProcessed file saved at:\n{output_path}")

#Run command: python option_data_formating.py {RAW_DATA_FILE_NAME}
