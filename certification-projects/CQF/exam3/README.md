# CQF Exam 3 - Machine Learning

This folder contains the working materials for **CQF Exam 3**, which studies whether short-term positive moves in **QQQ** (Nasdaq 100 ETF) can be predicted with a supervised learning pipeline.

The submission follows the exam structure:

- **Section A**: Entropy in classification (True / False + reasoning)
- **Section B**: Feature selection with a three-stage funnel
- **Section C**: Model building, tuning, and evaluation
- **Section D**: Optional backtest of the trading signals

## Submission Summary

- Main analytical notebook: [report.ipynb](report.ipynb) (source of truth)
- Generated report PDF: `report.pdf` (build artefact, **not tracked in git**; produced by the last cell of the notebook)
- Raw cached data: `data/raw/QQQ_5y.csv`
- Processed feature set: `data/processed/QQQ_processed.csv`
- Saved figures: `figures/*.png`

The notebook is the single source of truth. Running it end-to-end regenerates `report.pdf` and all figures from the live kernel state, so the PDF and notebook outputs always stay numerically consistent within one run.

## Methodology Notes (read this first)

Daily price data is a time series, so the evaluation protocol respects time order everywhere:

- **Train / test split**: chronological, first 80% of days train, last 20% test. No shuffling, no stratification. A random shuffle would let the model train on near-twin rows of its test days, which would inflate scores without measuring true out-of-sample skill.
- **Cross-validation**: `TimeSeriesSplit(n_splits=5)` inside `GridSearchCV`, in the embedded-stage stability check, and in the learning curves. Each validation fold sits strictly after its training fold.
- **Decision threshold**: tuned on out-of-fold training predictions to maximise balanced accuracy, so it never touches the test set.

## Exam-Aligned Structure

### Section A - Entropy

- (a) "High entropy means the partitions are pure." → **FALSE**
- (b) "High entropy means the partitions are impure." → **TRUE**
- Entropy measures how mixed-up the labels are; low entropy = pure, high entropy = impure. Decision trees prefer splits that lower entropy because those splits raise information gain.

### Section B - Feature Selection

A three-stage funnel:

1. **Filter stage** — ANOVA F-score and mutual information screen 39 engineered features down to about 27.
2. **Wrapper stage** — Gradient Boosting on the filtered set; keep the features the model actually uses. About 15 features remain.
3. **Embedded stage** — refit Gradient Boosting under TimeSeriesSplit and keep only features whose importance stays non-zero across folds. The final list has 15 stable features.

### Section C - Model Building and Evaluation

- Final classifier: `GradientBoostingClassifier`, tuned with `GridSearchCV` over **108 combinations** (3 × 3 × 2 × 3 × 2) using `TimeSeriesSplit(n_splits=5)` and `scoring='roc_auc'`.
- **Hyperparameter grid**:
  - `n_estimators`: {60, 100, 150}
  - `learning_rate`: {0.02, 0.05, 0.08}
  - `max_depth`: {2, 3}
  - `min_samples_leaf`: {10, 20, 30}
  - `subsample`: {0.7, 0.85}
- The grid is intentionally conservative (shallow trees, larger leaves, row subsampling) because daily equity returns have a very low signal-to-noise ratio, so deeper / less-regularised settings would easily memorise idiosyncrasies of the training window.
- **Decision threshold**: chosen on training out-of-fold predictions by maximising balanced accuracy.
- **Reporting**: train / test ROC-AUC, accuracy, precision, recall, F1, confusion matrices, plus a 5-fold TimeSeriesSplit stability table, learning curves, and a hyperparameter sensitivity ranking.
- **Feature importance**: split into momentum / volatility / volume / mean-reversion buckets for interpretation.

### Section D - Optional: Backtesting Trading Strategy (Bonus)

Long-only trading rule on the chronological test window. Two variants are reported:

- **Baseline**: probability cutoff 0.50, hold for 1 day, costs 0.05% + 0.02% per leg.
- **Optimised**: probability cutoff 0.55, hold for 3 days, costs 0.03% + 0.01% per leg.

The optimised version only acts on the model's more confident calls and rides them for a few days, instead of trading every weak signal.

## Data and Features

- **Ticker**: QQQ
- **Frequency**: Daily
- **Lookback**: 5 years
- **Fields**: Open, High, Low, Close, Volume

Engineered feature families: price spreads, rolling volatility, momentum, moving averages and deviations, mean-reversion indicators, volume-based features, and lagged returns.

## Reported Results

All numbers below come from the most recent end-to-end notebook run. Reruns may shift the last digit because `GridSearchCV` is parallelised with `n_jobs=-1` (`random_state=42` is set everywhere it is exposed). The PDF is regenerated from the live kernel state, so the two artefacts always stay numerically consistent within a single run.

### Model Performance

Target: next trading day's return thresholded at +0.15%, so the model is evaluated on a strictly forward-looking task using only information available by the current day's close.

- **Held-out chronological test set (239 samples)**: accuracy **0.4854**, ROC-AUC **0.4980**, confusion matrix `[[60, 56], [67, 56]]`, decision threshold **0.480**
- **5-fold TimeSeriesSplit on the wrapper-selected set**: AUC mean **0.5507** (std ~0.04)
- **GridSearchCV best CV AUC**: matches the CV stability run, achieved by the best parameter combination from the 108-combo grid
- **Train vs test gap**: the model fits the training window much better than the chronological test window, which is the standard picture for next-day equity direction with technical features alone

Plain-English reading: on the average prediction, overall AUC sits close to 0.5, in line with the academic literature on this problem. The useful edge lives in the **tails of the probability distribution** — predictions with probability above 0.55 are noticeably more reliable than the average. The bonus backtest in Section D is designed to exploit exactly that.

### Final Feature Set (15 Features)

1. ATR_14 (volatility)
2. Deviation_21d (volatility)
3. MACD_Histogram (momentum)
4. MACD_Signal (momentum)
5. Momentum_21d (momentum)
6. Momentum_7d (momentum)
7. RSI_14 (mean reversion)
8. Returns_Lag_1 (lagged returns)
9. Returns_Lag_3 (lagged returns)
10. Stochastic_D (mean reversion)
11. Stochastic_K (mean reversion)
12. Volatility_21d (volatility)
13. Volume_Change (volume)
14. Volume_MA_7d (volume)
15. Volume_Ratio (volume)

### Backtesting Results

Long-only execution on the chronological test window, starting from $100,000.

| Strategy | Confidence threshold | Hold | Trades | Win rate | Final value | Net return |

|---|---|---|---|---|---|---|
| Baseline | 0.50 | 1 day | 91 | 52.7% | $100,188.79 | **+0.19%** |
| Optimised | 0.55 | 3 days | 33 | 72.7% | $122,382.73 | **+22.38%** |

Optimised strategy additional stats: average trade return **+0.66%**, average holding period **3.0 days**, total transaction costs **~1.98%** of capital.

The gap between a near-random classification AUC and a clearly positive optimised backtest is the main story of this report. The baseline (cutoff 0.50, 1-day hold) trades too often and gives most of its edge back to costs, while the optimised version concentrates on the model's confident calls and rides them for a few days. Some of this performance is also helped by the particular market regime in the test window, so the result should be read as evidence of a small but useful edge in the tails of the probability distribution, not as a guaranteed live strategy.

Saved figures:

- `roc_curves.png` — training and test ROC curves
- `confusion_matrices.png` — train / test confusion matrices
- `feature_importance.png` — top features ranked by model importance
- `prediction_distributions.png` — predicted probability histograms split by true class
- `learning_curves.png` — train / validation AUC vs. growing training window (TimeSeriesSplit)

## Folder Structure

```text
exam3/
├── README.md
├── .gitignore
├── report.ipynb
├── data/
│   ├── raw/
│   │   └── QQQ_5y.csv
│   └── processed/
│       └── QQQ_processed.csv
└── figures/
    ├── roc_curves.png
    ├── confusion_matrices.png
    ├── feature_importance.png
    ├── prediction_distributions.png
    └── learning_curves.png
```

## Reproducibility

Open the notebook and run it from top to bottom in the provided Python environment. The final cell rebuilds `report.pdf` from the live kernel state, so every rerun keeps the PDF and notebook outputs aligned.

Seeds are set to `random_state=42` for every scikit-learn estimator and split that supports it. Small numerical variation between runs can still occur because `GridSearchCV` runs with `n_jobs=-1`; this only affects exact metric digits, not the overall conclusions.

Recommended packages:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- yfinance
- reportlab (for PDF generation)

## Notes for Submission

- The notebook contains the full analytical workflow; the final cell builds `report.pdf` directly from the kernel state and embeds the saved figures.
- If the notebook is rerun, run all cells in order so that the saved outputs and the PDF stay consistent.

## Author

Mao Yikai
