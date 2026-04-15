# BANKNIFTY Volatility Forecasting Pipeline

A two-stage volatility forecasting system for BANKNIFTY options. The pipeline ingests raw NSE options data, computes implied volatility, fits a rolling GARCH model, and then trains an XGBoost model to correct GARCH's systematic blind spots using options market signals.

**Author:** Divyansh Pathak — [divyansh.pathak129@gmail.com](mailto:divyansh.pathak129@gmail.com)

---

## How It Works

```
Stage 1 — GARCH(1,1) / GJR-GARCH
  Spot returns  →  baseline volatility forecast

Stage 2 — XGBoost correction
  GARCH forecast + options signals (IV, skew, OI, PCR, term structure, HV)
  →  predict whether GARCH under/overestimates tomorrow's vol
  →  Final forecast = GARCH forecast ± XGBoost correction
```

GARCH is blind to real-time options market signals — OI spikes, skew surges, expiry effects. XGBoost learns from the residuals GARCH leaves behind.

---

## Repository Structure

```
stockspipeline/
├── preprocess.py                    # Full data pipeline (Phases 1–7)
├── xgboost_volatility_model.py      # XGBoost model (Phases 0–8)
├── option_data_formating.py         # Format raw NSE option chain CSVs
├── daily_predict.py                 # Daily volatility signal generator
├── parquet_viewer.py                # Inspect .parquet files
├── BANKNIFTY/                       # Raw NSE monthly Excel files (input)
├── data/                            # Intermediate and final datasets
├── vol_dashboard/                   # Streamlit dashboard (read-only)
├── models/                          # Trained XGBoost models (.ubj, .pkl)
└── outputs/                         # Plots and forecast CSV
```

---

## Requirements

**Python 3.10+**

```bash
pip install pandas numpy scipy arch xgboost scikit-learn shap \
            matplotlib seaborn joblib pyarrow openpyxl
```

---

## Usage

See **[QUICKSTART.md](QUICKSTART.md)** to get running in minutes.

See **[DOCUMENTATION.md](DOCUMENTATION.md)** for the full reference — architecture, configuration, feature engineering, model history, and troubleshooting.

---

## Sample Outputs

### EDA — ATM IV, GARCH vs Realized Vol, Skew, GARCH Error
![EDA Signals](assets/eda_signals.png)

### Feature Correlation Matrix
![Correlation Heatmap](assets/correlation_heatmap.png)

### Baseline XGBoost Feature Importances
![Feature Importances](assets/feature_importances_baseline.png)

### SHAP — Global Feature Importance
![SHAP Importance](assets/shap_importance.png)

### SHAP — Direction and Magnitude
![SHAP Beeswarm](assets/shap_beeswarm.png)

### ROC Curve — Test Set
![ROC Curve](assets/roc_curve.png)

### Forecast Comparison — GARCH vs GARCH+XGBoost vs Realized Vol
![Forecast Comparison](assets/forecast_comparison.png)

---

## License

MIT License — free to use, modify, and distribute with attribution.

---

## Contributing

Pull requests are welcome. If you're adding support for other indices (NIFTY 50, FINNIFTY) or alternate volatility models (EGARCH), please open an issue first.
