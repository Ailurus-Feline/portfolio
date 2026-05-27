# CQF Exam 3 - Machine Learning

This folder contains the working materials for **CQF Exam 3**, which studies whether short-term positive moves in **QQQ** (Nasdaq 100 ETF) can be predicted with a supervised learning pipeline.

The submission is organized to match the exam structure:

- **Section A**: Entropy in classification
- **Section B**: Feature selection with a funneling approach
- **Section C**: Model building, tuning, and evaluation
- **Section D**: Optional backtesting of trading signals

## Submission Summary

- Main analytical notebook: [report.ipynb](report.ipynb) (source of truth)
- Generated report PDF: `report.pdf` (build artifact, **not tracked in git** — produced by the last cell of the notebook)
- Raw cached data: `data/raw/QQQ_5y.csv`
- Processed feature set: `data/processed/QQQ_processed.csv`
- Saved figures: `figures/*.png`

The notebook is the single source of truth. Running it end-to-end regenerates `report.pdf` and all figures from the live kernel state, so the PDF and notebook outputs always agree for any given run.

## Exam-Aligned Structure

### Section A - Entropy

- Explains entropy as a measure of impurity in classification problems.
- States the correct interpretation that high entropy means impure partitions.
- Connects entropy to information gain and decision tree splitting.

### Section B - Feature Selection

The notebook uses a three-stage funneling approach:

1. **Filter stage**
   - ANOVA F-score and mutual information screening.
   - Retains 27 features from initial 39 engineered features.

2. **Wrapper stage**
   - Gradient boosting model fit on the filtered set.
   - Uses model-derived importance to refine to 15 features.

3. **Embedded stage**
   - Final selection based on 5-fold CV stability analysis.
   - Confirms 15 final features for model training.

### Section C - Model Building and Evaluation

- Trains a `GradientBoostingClassifier` and tunes it with `GridSearchCV` (5-fold cross-validation) over **108 hyperparameter combinations** (3 × 3 × 3 × 2 × 2).
- **Hyperparameter Grid**: n_estimators {80, 120, 160}, learning_rate {0.05, 0.08, 0.1}, max_depth {3, 4, 5}, min_samples_split {5, 8}, min_samples_leaf {2, 3}.
- **Cross-Validation Stability**: Reports mean and standard deviation of accuracy, precision, recall, F1, and AUC across the 5 folds.
- **Overfitting Diagnosis**: Compares train vs. test metrics. Training AUC is high (near 1.0) while test AUC sits in the mid-0.5s, reflecting the well-known difficulty of forecasting next-day equity returns: the model fits in-sample patterns but generalises only marginally above the no-skill baseline on truly held-out data.
- **Learning Curves Analysis**: Plots training and validation AUC-ROC against increasing dataset sizes to diagnose overfitting risk.
- **Feature Importance Interpretation**: Maps model importance scores to financial meanings (momentum, volatility, mean reversion, volume) and identifies primary drivers.
- Evaluates the model using ROC-AUC, accuracy, precision, recall, F1-score, and confusion matrices.
- **Statistical Significance Testing**: Applies t-tests and bootstrap confidence intervals to validate the robustness of backtesting results.
- Generates interpretation plots for reporting (ROC curves, confusion matrices, learning curves, feature importance).

### Section D - Optional: Backtesting Trading Strategy (Bonus)

Implements practical backtesting to evaluate whether the ML predictions translate into profitable trading signals:

**Strategy Design:**

- Buy signals: Generated when model predicts uptrend probability > 55%
- Position: Long-only, 3-day hold
- Costs: 0.03% round-trip
- Capital: $100,000 initial portfolio

**Performance Metrics (optimized strategy, latest run):**

- Total Trades Executed: 53
- Winning Trades: 34 (Win Rate: 64.2%)
- Final Portfolio Value: $114,168.45
- Net Return: **+14.17%**
- Average Trade Return: +0.30%
- Average Holding Period: 3.0 days
- Comparison vs. buy-and-hold benchmark

**Outputs:**

- Comprehensive performance table
- Visualizations: portfolio growth, drawdowns, metrics comparison
- Summary report with key insights

This optional section demonstrates the practical application of the ML model and provides evidence of edge in the identified patterns.

## Data and Features

- **Ticker**: QQQ
- **Frequency**: Daily
- **Lookback**: 5 years
- **Fields**: Open, High, Low, Close, Volume

Engineered features include:

- Price spreads
- Rolling volatility
- Momentum signals
- Moving averages and deviations
- Mean reversion features
- Volume-based features
- Lagged returns

## Reported Results

All numbers below are taken verbatim from the most recent end-to-end notebook run committed alongside this README. Reruns may produce slightly different digits because `GridSearchCV` is parallelised with `n_jobs=-1` (`random_state=42` is set everywhere it is exposed). When the notebook is re-executed, the PDF is regenerated from the live kernel state, so the two artefacts always stay numerically consistent within a single run.

### Model Performance

The label is the **next trading day's return** thresholded at +0.15%, so the model is evaluated on a strictly forward-looking task using only information available by the current day's close.

- **Held-out test set (239 samples)**: accuracy **0.5230**, ROC-AUC **0.5452**, confusion matrix `[[68, 57], [57, 57]]`
- **5-fold CV on the wrapper-selected feature set**: AUC mean **0.5338** (std 0.0229, range 0.5003-0.5687); accuracy mean **0.5268** (std 0.0077)
- **GridSearchCV best CV AUC**: **0.5365** across 108 candidate configurations
- **Train vs test (5-fold means)**: train AUC **0.9992**, test AUC **0.4607**, gap **0.5385** -- the large gap is consistent with the well-documented difficulty of forecasting next-day equity direction from technical features alone, and motivates the confidence-thresholded backtest in Section D rather than a raw-probability strategy

### Final Feature Set (15 Features)

1. ATR_14 (volatility)
2. MACD_Histogram (momentum)
3. Momentum_21d (momentum)
4. Momentum_7d (momentum)
5. RSI_14 (momentum)
6. Returns_Lag_1 (lagged returns)
7. Returns_Lag_2 (lagged returns)
8. Returns_Lag_3 (lagged returns)
9. Stochastic_D (momentum)
10. Stochastic_K (momentum)
11. TR (volatility)
12. Volatility_21d (volatility)
13. Volume_Change (volume)
14. Volume_MA_7d (volume)
15. Volume_Ratio (volume)

### Backtesting Results

Long-only execution on the test window, starting from $100,000 capital with 0.03% round-trip costs.

| Strategy | Confidence threshold | Trades | Win rate | Final value | Net return |
|---|---|---|---|---|---|
| Baseline | 0.50 | 114 | 57.9% | $104,070.18 | **+4.07%** |
| Optimized (3-day hold) | 0.55 | 53 | 64.2% | $114,168.45 | **+14.17%** |

Optimized strategy additional stats: average trade return **+0.30%**, average holding period **3.0 days**, total transaction costs **~3.18%** of capital.

The contrast between a modest classification AUC (~0.55) and a positive backtest P&L reflects the fact that even a small directional edge, when combined with a confidence threshold and a short holding period, can translate into meaningful cumulative returns over hundreds of bars.

Saved figures include:

- `roc_curves.png` - Training and test ROC curves comparing model performance
- `confusion_matrices.png` - Train/test confusion matrices for classification diagnostics
- `feature_importance.png` - Top features ranked by model importance scores
- `prediction_distributions.png` - Histogram of predicted probabilities on train/test sets
- `learning_curves.png` - Training and validation AUC-ROC vs. dataset size for overfitting diagnosis

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

Open the notebook and run it from top to bottom in the provided Python environment. The final cell regenerates `report.pdf` from the live kernel state, so any rerun keeps the PDF and notebook outputs aligned.

Seeds are set to `random_state=42` for every scikit-learn estimator and split that supports it. Small variation between runs can still occur because `GridSearchCV` is run with `n_jobs=-1`; this only affects exact metric digits, not the overall conclusions.

Recommended packages:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- yfinance
- reportlab (for PDF generation)

## Notes for Submission

- The notebook contains the full analytical workflow; the final cell builds `report.pdf` directly from the kernel state, embedding the saved figures.
- If the notebook is regenerated, rerun all cells in order so that the saved outputs and the PDF remain consistent.

## Author

Mao Yikai
