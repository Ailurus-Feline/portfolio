# CQF Exam 3 - Machine Learning

This folder contains the working materials for **CQF Exam 3**, which studies whether short-term positive moves in **QQQ** (Nasdaq 100 ETF) can be predicted with a supervised learning pipeline.

The submission is organized to match the exam structure:

- **Section A**: Entropy in classification
- **Section B**: Feature selection with a funneling approach
- **Section C**: Model building, tuning, and evaluation
- **Section D**: Optional backtesting of trading signals

## Submission Summary

- Main analytical notebook: [report.ipynb](report.ipynb)
- Raw cached data: `data/raw/QQQ_5years.csv`
- Processed feature set: `data/processed/QQQ_features_processed.csv`
- Saved figures: `figures/*.png`

The notebook is the primary analytical artifact. It contains the full explanation, code, outputs, and final conclusions required for the exam.

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

- Trains a `GradientBoostingClassifier` with 108 hyperparameter combinations
- Tunes hyperparameters with `GridSearchCV` using 5-fold cross-validation.
- **Hyperparameter Grid**: Tests n_estimators {80, 120, 160}, learning_rate {0.05, 0.08, 0.1}, max_depth {3, 4, 5}, min_samples_split {5, 8}, min_samples_leaf {2, 3}.
- **Cross-Validation Stability**: Reports metrics across 5 folds (accuracy, precision, recall, F1) with standard deviation to assess model consistency.
- **Overfitting Diagnosis**: Compares train vs. test metrics to diagnose overfitting (Train AUC: 0.9999, Test AUC: 0.9211, gap: 0.0788).
- **Learning Curves Analysis**: Plots training and validation AUC-ROC against increasing dataset sizes to diagnose overfitting risk; shows zero train-validation gap.
- **Feature Importance Interpretation**: Maps model importance scores to financial meanings (momentum, volatility, mean reversion, volume) and identifies primary drivers.
- Evaluates the model using ROC-AUC, accuracy, precision, recall, F1-score, and confusion matrices.
- **Statistical Significance Testing**: Applies t-tests and bootstrap confidence intervals to validate the robustness of backtesting results.
- Generates interpretation plots for reporting (ROC curves, confusion matrices, learning curves, feature importance).

### Section D - Optional: Backtesting Trading Strategy (Bonus)

Implements practical backtesting to evaluate whether the ML predictions translate into profitable trading signals:

**Strategy Design:**

- Buy signals: Generated when model predicts uptrend probability > 50%
- Position: Long-only, single-day hold (enter and exit next day)
- Costs: 0.05% transaction cost + 0.02% slippage per trade
- Capital: $100,000 initial portfolio

**Performance Metrics:**

- Total Trades Executed: 122
- Winning Trades: 67 (54.9% win rate)
- Final Portfolio Value: $97,982.95
- Total Return: -2.02%
- Cumulative and annualized returns
- Risk metrics (volatility, max drawdown, Sharpe/Sortino/Calmar ratios)
- Trade statistics (profit factor, average trade return)
- Comparison vs. buy-and-hold benchmark

**Outputs:**

- Comprehensive performance table
- Visualizations: portfolio growth, drawdowns, metrics comparison
- Summary report with key insights and limitations

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

### Model Performance (5-Fold Cross-Validation)

**Test Set Metrics:**
- ROC-AUC: 0.9211
- Accuracy: 0.8407 ± 0.0210
- F1-score: 0.8384 ± 0.0147

**Train Set Metrics:**
- ROC-AUC: 0.9999
- Accuracy: 0.9960 ± 0.0014
- Train-Test Gap: 0.0788 (indicates overfitting)

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

### Backtesting Results

- Total Trades: 122
- Winning Trades: 67
- Win Rate: 54.9%
- Final Portfolio Value: $97,982.95
- Total Return: -2.02%
- Gross Return (before costs): +5.82%
- Trading Costs Impact: -7.84%

Saved figures include:

- `roc_curves.png` - Training and test ROC curves comparing model performance
- `confusion_matrices.png` - Train/test confusion matrices for classification diagnostics
- `feature_importance.png` - Top features ranked by model importance scores
- `prediction_distributions.png` - Histogram of predicted probabilities on train/test sets
- `learning_curves.png` - Training and validation AUC-ROC vs. dataset size for overfitting diagnosis
- `backtesting_analysis.png` (optional section) - Strategy vs. buy-and-hold performance comparison

## Folder Structure

```text
exam3/
├── README.md
├── .gitignore
├── report.ipynb
├── data/
│   ├── raw/
│   │   └── QQQ_5years.csv
│   └── processed/
│       └── QQQ_features_processed.csv
└── figures/
    ├── roc_curves.png
    ├── confusion_matrices.png
    ├── feature_importance.png
    ├── prediction_distributions.png
    └── backtesting_analysis.png
```

## Reproducibility

Open the notebook and run it from top to bottom in the provided Python environment.

Recommended packages:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- yfinance

## Notes for Submission

- PDF output is excluded from version control through `.gitignore`.
- The notebook contains the full analytical write-up and can be exported to PDF for submission if required by the exam instructions.
- If the notebook is regenerated, rerun all cells in order so that the saved outputs remain consistent.

## Author
f
Mao Yikai
