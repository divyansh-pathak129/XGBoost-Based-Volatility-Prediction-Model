
---

## Data Evaluation Plan: Visualization and Analysis

### Goal

Before training any model on the full 12-month dataset, we need to understand the data deeply. This phase answers three questions: Is the data healthy? Are the features well-behaved? Is the target learnable? Every visualization below produces a specific conclusion that drives a specific action.

---

### Step 1: Dataset Health Check

**What to visualize:** Print a summary table showing row count, date range, number of trading days, features available, and NaN count per column. Then plot a heatmap of missing values across all columns and all dates — rows on Y axis, columns on X axis, white cells = missing, colored cells = present.

**How to visualize:** Use seaborn heatmap with `yticklabels=False` so individual dates don't clutter. Color scheme: white for missing, dark blue for present.

**Conclusion to make:** Any column with more than 15% missing values needs a decision — either drop it or impute it. Any date range with large horizontal white bands means those trading days have no data at all and must be investigated. Expected outcome: HV columns should have NaN only in the first 30 rows (rolling warmup). GARCH columns should have NaN only in the first 60 rows. If you see NaN scattered randomly in the middle of the dataset, that means specific trading days had bad options data upstream.

**Action from conclusion:** Drop columns above 15% NaN threshold. For scattered NaN in the middle, forward fill using the previous day's value (never mean fill for time series).

---

### Step 2: Target Variable Analysis

**What to visualize:** Three charts side by side.

Chart 1 — Bar chart of class counts: Two bars showing how many days fall in Class 0 (GARCH overestimated) vs Class 1 (GARCH underestimated). Label each bar with count and percentage.

Chart 2 — Target over time: Plot Target_Binary as a scatter plot against Date, with Class 0 dots in blue and Class 1 dots in red. This shows whether one class dominates specific time periods.

Chart 3 — GARCH_Error distribution: Plot a histogram of the raw GARCH_Error column (the continuous version before binarizing) with a vertical red line at zero. Show the mean and median as annotations.

**How to visualize:** Use matplotlib with 3 subplots in one row, figure size 18x5. For the time scatter, use `alpha=0.7` so overlapping points are visible.

**Conclusion to make:** If the bar chart shows more than 60/40 imbalance, you have a class imbalance problem that requires `scale_pos_weight` correction in XGBoost. If the time scatter shows one class dominating a specific 2-3 month block, this means GARCH behaved systematically differently in that period (likely a high-volatility event like a market crash or political shock) — this is a regime shift and your model needs to account for it. If the GARCH_Error histogram is heavily skewed left or right, the threshold of zero for binarizing may not be the best split point — consider using the median instead of zero as the split.

**Action from conclusion:** If imbalance exists, set `scale_pos_weight = count_class_0 / count_class_1`. If a time cluster exists in one class, note the date range and flag it as a regime period. If histogram is skewed, recompute Target_Binary using median as threshold instead of zero.

---

### Step 3: Feature Distributions

**What to visualize:** A grid of histograms — one per feature — showing the distribution of each feature's values across all trading days. Total features are approximately 22, so arrange in a 5x5 grid. On each histogram, draw a vertical red line at the mean and annotate the skewness value in the top corner of each plot.

**How to visualize:** Use matplotlib subplots with `figsize=(20, 20)`. For each feature use `bins=30`. Compute skewness using `scipy.stats.skew()` and print it on the plot. Use a different color for features that have skewness above 2 or below -2 (color them orange instead of blue) to visually flag problems.

**Conclusion to make:** Features with absolute skewness above 2 are heavily skewed and will hurt XGBoost performance — they need log transformation. Features that look bimodal (two humps in the histogram) suggest the feature behaves differently in two regimes, which is actually useful information. Features with almost no variation (very narrow spike) are low-variance and should be considered for removal. Based on financial knowledge, expect these features to be skewed: `Total_OI`, `OI_Change`, `Volume_Change`, `Total_Volume` — these are count-based features that can spike dramatically on high-activity days.

**Action from conclusion:** For every feature with absolute skewness above 2, apply log transformation: `log1p(abs(value)) * sign(value)`. This preserves sign for negative values like OI_Change while compressing extreme values. Recheck skewness after transformation — it should come below 1.5. Drop any feature where more than 90% of values are the same (near-zero variance).

---

### Step 4: Outlier Detection

**What to visualize:** Two types of charts.

Chart type 1 — Box plots: A row of box plots, one per feature, all in one figure. Rotate X axis labels 45 degrees for readability. Outliers will appear as dots outside the whiskers.

Chart type 2 — Z-score time series: For your five most important features (ATM_IV, Skew, OI_Change, GARCH_Forecast, PCR_Volume), plot their Z-score over time as a line chart. Draw horizontal red dashed lines at +3 and -3. Any point crossing these lines is a statistical outlier.

**How to visualize:** For box plots use seaborn `boxplot` after standardizing all features to the same scale using `StandardScaler` — otherwise features with large absolute values (like Total_OI in millions) will crush the chart. For Z-score time series, use a 5-row subplot figure with shared X axis.

**Conclusion to make:** Box plots tell you which features have the most extreme outliers. Z-score time series tells you WHEN they happened. If multiple features spike simultaneously on the same dates, those are market shock days (budget announcements, RBI policy, global events). These are not data errors — they are real events. Removing them would make the model blind to exactly the kind of situations where vol forecasting matters most. However, if a single feature has an isolated extreme outlier on a specific day and no other features show anything unusual, that is likely a data error from NSE.

**Action from conclusion:** Do not remove outlier rows that correspond to genuine market events — flag them with a new binary column called `Is_market_shock` (1 if any feature has Z-score above 3 on that day). This becomes an additional feature for XGBoost. For isolated single-feature data errors (only one feature spiking with no market explanation), cap that feature value at the 99th percentile for that specific row.

---

### Step 5: Feature Correlations

**What to visualize:** Two charts.

Chart 1 — Feature-to-feature correlation heatmap: Full correlation matrix of all features, displayed as a heatmap with annotations showing the correlation value in each cell. Use a diverging colormap (red for negative, blue for positive, white for zero). Mask the upper triangle to avoid redundancy.

Chart 2 — Feature-to-target correlation bar chart: Horizontal bar chart showing the Pearson correlation of each feature with Target_Binary, sorted from most positive to most negative. Color bars green if correlation is above 0.1 or below -0.1, and gray if between -0.1 and 0.1. Annotate each bar with the correlation value.

**How to visualize:** For the heatmap use `seaborn.heatmap` with `annot=True, fmt='.2f', cmap='RdBu_r', vmin=-1, vmax=1`. Figure size 16x14. For the bar chart use `plt.barh` with horizontal orientation, figure size 10x8.

**Conclusion to make:** Feature-to-feature heatmap — if two features have correlation above 0.85 or below -0.85, they are nearly redundant. Keeping both gives XGBoost no additional information and wastes model capacity. For example, HV_10, HV_20, and HV_30 are all rolling volatility windows and will likely correlate highly with each other. Feature-to-target bar chart — features with absolute correlation below 0.05 with the target are weak predictors on their own. They might still contribute in combination with other features inside XGBoost, but if you have many such features they add noise on a small dataset.

**Action from conclusion:** For pairs of features with correlation above 0.85, keep only the one with higher individual correlation to the target and drop the other. For features with near-zero target correlation (below 0.05 absolute), note them but do not immediately drop — first check their SHAP values after training.

---

### Step 6: Time Series Stationarity and Stability

**What to visualize:** Three charts.

Chart 1 — Rolling mean and standard deviation of ATM_IV: Plot ATM_IV as a line over time. Overlay a 20-day rolling mean as a thicker line. Overlay a shaded band showing plus/minus one rolling standard deviation. This shows whether IV is mean-reverting or trending.

Chart 2 — Rolling mean of key features over time: For ATM_IV, Skew, HV_20, and GARCH_Forecast, compute their 30-day rolling mean and plot all four on the same chart with different colors. This reveals regime shifts — periods where the statistical behavior of features changed permanently.

Chart 3 — Autocorrelation function (ACF) plot: For ATM_IV and log_return, plot the ACF for up to 20 lags. Use the standard ACF bar chart where bars above the blue confidence band indicate significant autocorrelation.

**How to visualize:** For Chart 1 and 2 use matplotlib with `figsize=(16, 5)`. For Chart 3 use `statsmodels.graphics.tsaplots.plot_acf` with `lags=20` in a 1x2 subplot.

**Conclusion to make:** Chart 1 — if ATM_IV shows a clear upward or downward trend over the year, it is non-stationary and you should use IV changes rather than IV levels as a feature. If it mean-reverts constantly, levels are fine. Chart 2 — if you see a clear step change in rolling means around a specific date (for example, ATM_IV suddenly jumps from 14% average to 22% average and stays there), that is a regime break. Your model trained before that break will behave differently from your model trained after it. Chart 3 — if ATM_IV shows significant autocorrelation at lag 1 and lag 2 (bars above the confidence band), it confirms your lag features are well-chosen. If log_return shows NO significant autocorrelation (expected for efficient markets), that confirms GARCH is appropriate.

**Action from conclusion:** If ATM_IV is non-stationary, add `ATM_IV_change = ATM_IV - ATM_IV_lag1` as a new feature alongside the level. If a regime break is detected, note the date and consider training separate models for pre-break and post-break periods, or adding a `Post_regime_break` binary flag as a feature.

---

### Step 7: GARCH Model Quality Check

**What to visualize:** Four charts in a 2x2 grid.

Chart 1 — GARCH Forecast vs Realized Vol over time: Plot both `GARCH_Forecast` and `Realized_Vol_proxy` as lines on the same chart. They should track each other loosely — GARCH should lead or match realized vol directionally.

Chart 2 — GARCH Error distribution: Histogram of GARCH_Error with a normal distribution curve overlaid. Annotate with mean, standard deviation, and kurtosis. Also plot a Q-Q plot to check normality.

Chart 3 — Standardized GARCH Residuals over time: Plot `GARCH_Residual` as a line over time. Should look like white noise (random, no pattern). If you see clusters of large residuals together, GARCH is still missing some volatility clustering.

Chart 4 — ACF of squared GARCH Residuals: Plot the autocorrelation of `GARCH_Residual ** 2` for 15 lags. If GARCH is working well, this ACF should show no significant bars. Significant bars mean there is still structure in the residuals that GARCH missed.

**How to visualize:** 2x2 subplot grid, `figsize=(16, 12)`. For the Q-Q plot use `scipy.stats.probplot`. For ACF use `statsmodels.graphics.tsaplots.plot_acf`.

**Conclusion to make:** Chart 1 — if GARCH_Forecast is consistently much higher or lower than Realized_Vol_proxy for long stretches, GARCH has a systematic bias in that period. Chart 2 — high kurtosis (above 3) means your GARCH errors have fat tails, confirming the t-distribution assumption in your GARCH model was the right choice. Chart 3 — if residuals look random (no visual pattern), GARCH is capturing volatility clustering well. If you see runs of positive or negative residuals, GARCH is missing something. Chart 4 — significant ACF in squared residuals means you should try GARCH(1,2) or GARCH(2,1) instead of GARCH(1,1) — the extra lag might capture the remaining structure.

**Action from conclusion:** If systematic bias exists in Chart 1, add a `GARCH_Bias_rolling` feature defined as the 20-day rolling mean of GARCH_Error. This tells XGBoost what GARCH's recent bias has been, allowing it to correct for it. If Chart 4 shows significant ACF, change GARCH order to (1,2) in the pipeline config and rerun.

---

### Step 8: Feature-Target Relationship (Non-linear)

**What to visualize:** For your top 6 features by absolute correlation with target, create scatter plots of feature value vs GARCH_Error (continuous). Color each point by Target_Binary class (blue for Class 0, red for Class 1). Add a LOWESS smoothing line (locally weighted regression line) to show the non-linear trend.

**How to visualize:** 2x3 subplot grid, `figsize=(18, 10)`. Use `statsmodels.nonparametric.smoothers_lowess.lowess` to compute the smoothing line. Plot it as a thick black line over the scatter.

**Conclusion to make:** If the LOWESS line is roughly flat (no slope), the feature has a weak linear relationship with the target — but XGBoost can still use it if there's a non-linear threshold effect visible in the scatter. If you see a clear curve or bend in the LOWESS line, XGBoost will capture this effectively. If the scatter shows two distinct clusters of red and blue points separated by a feature value (for example, all Class 1 points cluster when OI_Change is below -50000), that threshold is a very strong split candidate and will appear high in XGBoost's feature importance.

**Action from conclusion:** Note any feature where you see a clear threshold effect. These features will naturally become early splits in XGBoost decision trees and carry high importance. No code action needed — this visualization validates that XGBoost is the right model type for this data structure.

---

### Step 9: Train-Validation-Test Split Visualization

**What to visualize:** A single chart showing ATM_IV over the full date range as a continuous line. Shade the training period in light blue, validation period in light orange, and test period in light green. Mark the split dates with vertical dashed lines and label them.

**How to visualize:** Use matplotlib with `axvspan` for the shaded regions. Figure size 16x4.

**Conclusion to make:** This visualization confirms two things. First, that the splits are chronologically correct with no overlap. Second, that each split period contains representative market behavior. If the test set falls in a period where ATM_IV is unusually high or low compared to the training period, the model will face an out-of-distribution problem — the test market regime is one the model never trained on. If this is the case, you need to either expand the training set or acknowledge that test performance may not reflect real-world performance when deployed in a normal-vol environment.

**Action from conclusion:** If test period IV levels are dramatically different from training, note this as a limitation in your model documentation. Consider using a validation period that also includes a high-vol stretch so the model has seen stress conditions during tuning.

---

### Step 10: Final Pre-Modeling Checklist

After completing all visualizations above, produce a single printed summary that answers these questions before proceeding to model training:

How many rows in final dataset — is it above 150? What is the exact class balance ratio? Which features need log transformation due to skewness above 2? Which features are highly correlated with each other and should be deduplicated? Are there any regime breaks detected — if yes, on what date? Is GARCH_Error systematic (mean far from zero) or unbiased (mean near zero)? Does ATM_IV show significant lag-1 autocorrelation confirming lag features are valid? Are there market shock days that need the Is_market_shock flag?

Print this checklist as a formatted table at the end of the visualization script. Save all plots to the outputs folder with descriptive filenames. Save the checklist summary as `outputs/pre_modeling_checklist.txt` so it can be referenced during model training.