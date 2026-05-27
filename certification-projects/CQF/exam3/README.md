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

**Performance Metrics (optimized strategy, illustrative):**

- Total Trades Executed: ~50
- Win Rate: ~60-65%
- Final Portfolio Value: ~$110k-$120k
- Net Return: roughly +10% to +18% over the test window
- Average Holding Period: ~3 days
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

### Model Performance

The label is the **next trading day's return** thresholded at +0.15%, so the model is evaluated on a strictly forward-looking task using only information available by the current day's close.

Indicative ranges observed across runs (`random_state=42` is used wherever scikit-learn exposes it, but `GridSearchCV` with `n_jobs=-1` introduces small parallel-execution variance, so exact metrics may shift by ~1-2%):

- **Held-out test set (~239 samples)**: accuracy ~0.50-0.55, ROC-AUC ~0.52-0.58
- **5-fold CV (mean across folds)**: accuracy ~0.50-0.55, ROC-AUC ~0.52-0.58, with fold-level standard deviation in the low single digits
- **Train set**: AUC ~1.0, reflecting the in-sample fit of gradient boosting; the large train-test gap is consistent with the well-documented difficulty of forecasting next-day equity direction from technical features alone

The exact numbers used in the report PDF are written there directly from the kernel at execution time, so the PDF and notebook outputs always agree with each other for any given run.

### Final Feature Set (15 Features)

1. ATR_14 (volatility)
2. BB_Position (volatility)
3. CloseOpenSpread (trend)
4. Deviation_7d (mean reversion)
5. MACD_Histogram (momentum)
6. Momentum_3d (momentum)
7. Momentum_7d (momentum)
8. Price_to_Max (mean reversion)
9. Returns_Lag_2 (lagged returns)
10. Returns_Lag_3 (lagged returns)
11. Stochastic_D (momentum)
12. Stochastic_K (momentum)
13. TR (volatility)
14. Volume_Change (volume)
15. Volume_MA_7d (volume)

### Backtesting Results (illustrative, from latest run)

- Total Trades: ~50 (optimized strategy with 55% confidence threshold)
- Win Rate: ~60-65%
- Final Portfolio Value: ~$110k-$120k on $100k initial capital
- **Net Return: roughly +10% to +18% over the test window**
- Average Holding Period: ~3 days

The exact figures in the PDF correspond to the most recent end-to-end notebook run. The contrast between modest classification AUC (~0.55) and positive backtest P&L reflects the fact that even a small directional edge, when combined with a confidence threshold and short holding period, can translate into meaningful cumulative returns over hundreds of bars.

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
