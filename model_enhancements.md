Overall Verdict
Progress is real but the model is not deployable yet. Validation improved significantly from the 8-month run (AUC 0.49 → 0.61) which confirms more data helped. But the test set completely collapsed with AUC 0.328 — worse than random. The model is confidently predicting the wrong class on March 2026 data. This is not a tuning problem. It is a structural problem with three interconnected causes.

The Core Diagnosis: What Actually Went Wrong
Problem 1 — The model learned the wrong regime
Your training data runs July to December 2025 — a period where GARCH overestimated vol 80% of the time (Class 0 dominated). March 2026 was different. In your test set, 10 out of 19 days were Class 1 (GARCH underestimated) — nearly 53%. That is a completely different statistical environment than what the model trained on. When a model sees a regime it has never been trained on, AUC drops below 0.5, meaning it inverts — it confidently predicts the wrong answer. This is exactly what happened.
The confusion matrix proves it: the model predicted Class 0 for every single test row. It got all 9 genuine Class 0 correct and missed 9 of 10 Class 1. It essentially said "it's always a GARCH overestimate" because that's what 80% of its training data showed.
Problem 2 — scale_pos_weight = 4.13 overcorrected then backfired
During training, setting this to 4.13 helped the model pay attention to the rare Class 1 cases. But it created a fragile decision boundary that only holds when the data distribution matches training. When March 2026 had genuinely more Class 1 days, the model's internal probability calibration was wrong — it had been trained to be skeptical of Class 1 signals, so even on genuinely high-volatility days it stayed below the decision threshold.
Problem 3 — HV_10, HV_20, HV_30 are splitting SHAP weight across redundant features
All three are rolling standard deviations of the same return series. They correlate above 0.9 with each other. XGBoost treats them as three separate features and distributes importance across them, which makes each appear weaker than it is and wastes the model's capacity. The combined effect is that HV as a signal is underweighted relative to noisier features like OI_Change.

The Six Fixes and Why Each One Works
Fix A — Change scale_pos_weight to √(imbalance ratio) = 2.03
The square root rule is a well-researched heuristic for class imbalance. It gives enough weight to the minority class without destroying probability calibration. After changing this, separately tune the decision threshold on the validation set — sweep from 0.3 to 0.7 in 0.05 steps and pick the threshold that maximizes macro F1. Never use the default 0.5 with imbalanced data.
Fix B — Replace HV_10, HV_20, HV_30 with two features: HV_20 and HV_ratio
Keep HV_20 as your primary rolling vol (it had the highest individual correlation with the target). Add HV_ratio = HV_10 / HV_30 as a new feature. When HV_ratio is above 1.0, short-term vol is elevated relative to long-term — this is a compression signal that XGBoost can use without the multicollinearity. Drop HV_10 and HV_30 individually.
Fix C — Add a regime flag
Compute ATM_IV_regime = 1 if today's ATM_IV is above its own 30-day rolling median, else 0. This binary flag tells XGBoost which vol environment it is currently in. When this flag changes, the model knows to adjust its predictions. This directly addresses the distribution shift between your training and test periods.
Fix D — Add GARCH_Bias_rolling as a feature
Compute a 20-day rolling mean of GARCH_Error. Call it GARCH_Bias_rolling. When this number has been positive for 15 days straight, GARCH has been systematically underestimating — the model should learn to predict Class 1 more aggressively during such stretches. This feature gives the model memory of GARCH's recent behavior without leaking future data.
Fix E — Change to 3-class target with a dead zone
The binary split at zero is too noisy. A GARCH_Error of +0.0001 and +0.018 are both labeled Class 1, but one is meaningless and the other is a strong signal. Create three classes: clear overestimate (error below -0.003), ambiguous (between -0.003 and +0.003), clear underestimate (above +0.003). Either train a 3-class XGBoost directly, or drop the ambiguous rows entirely and train a cleaner binary model on just the unambiguous cases. This will reduce your dataset by roughly 20–30 rows but dramatically improve signal quality.
Fix F — Use walk-forward cross-validation with 5 expanding windows
Replace the single train/val/test split with 5 expanding windows. Window 1: train on months 1–5, test on month 6. Window 2: train on months 1–6, test on month 7. And so on. Average the AUC across all 5 windows. This gives you a performance estimate that is not sensitive to how one specific month behaved. The early stopping at iteration 10 happened because your single validation set was too small (41 rows) to give stable gradient signals — walk-forward CV fixes this.

Expected Results After Fixes
With all six fixes applied on your 12-month dataset, realistic targets become:
MetricCurrentAfter fixesVal AUC0.6110.62–0.68Test AUC0.3280.54–0.62Val/test AUC gap0.283below 0.10MAE improvement-21.1%+3–8%Direction accuracy31.6%55–65%
The biggest single improvement will come from Fix C (regime flag) and Fix D (GARCH_Bias_rolling) because they directly address the distribution shift that caused the test collapse. Implement those two first before anything else.
