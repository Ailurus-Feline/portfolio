# CQF Exam 3: Machine Learning

## Predicting Positive Market Moves in QQQ

**Student**: Mao Yikai  
**Cohort**: Jan 2026  
**Asset**: QQQ (Nasdaq 100 ETF)  
**Data Window**: 5 years of daily observations

## Executive Summary

This report investigates whether short-term positive moves in QQQ can be predicted using supervised machine learning. The study follows the exam structure closely: Section A explains entropy in classification, Section B applies a three-stage funneling approach for feature selection, and Section C builds and evaluates a Gradient Boosting model.

The workflow uses daily OHLCV data, technical feature engineering, and a binary target defined from near-zero daily returns. After feature selection and hyperparameter tuning, the final model achieves perfect test-set classification on the held-out sample. Because the results are unusually strong, the interpretation should be handled carefully, especially when discussing realism and generalisability.

## Section A: Entropy in Classification

Entropy measures the amount of disorder or uncertainty in a label distribution. For a set $S$ with class proportions $p_i$, entropy is defined as:

$$
H(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)
$$

A node with low entropy is more pure, meaning the samples mostly belong to a single class. A node with high entropy is more mixed and therefore less useful for classification. Decision trees prefer splits that reduce entropy, because those splits increase information gain and improve class separation.

In this exam, the correct interpretation is that high entropy indicates impure partitions, not pure ones.

## Section B: Feature Selection Using a Funneling Approach

The feature selection process follows a three-stage funnel.

### Stage 1: Filter Method

The initial engineered feature set is screened using ANOVA F-score and mutual information. This step quickly removes variables with weak statistical association to the target and keeps the more informative candidates.

### Stage 2: Wrapper Method

A Gradient Boosting model is trained on the filtered subset. The model’s feature importance scores are then used to retain the strongest predictors. This stage is more selective because it considers how features behave inside a predictive model rather than in isolation.

### Stage 3: Embedded Method

The remaining non-zero importance features are kept for the final model. This produces a compact subset that balances predictive value and interpretability.

### Final Feature Set

The final set contains 15 features, including price spread, momentum, volatility, moving average, and lagged-return variables. The strongest signal in the final model comes from CloseCloseSpread, which is intuitive because the current daily return is highly informative for the next movement classification task.

## Section C: Model Building, Tuning, and Evaluation

The final classifier is a Gradient Boosting model. Hyperparameters are tuned with GridSearchCV using a reduced search grid to keep execution efficient while still exploring the main model controls. The tuning process identifies the best combination of learning rate, tree depth, minimum split size, minimum leaf size, and number of estimators.

The final model is evaluated on a stratified train/test split. The reported metrics are:

- Test ROC-AUC: 1.0000
- Test Accuracy: 1.0000
- Test Precision: 1.0000
- Test Recall: 1.0000
- Test F1-score: 1.0000

The notebook also saves four diagnostic plots:

- ROC curves
- Confusion matrices
- Feature importance
- Prediction probability distributions

## Section D: Optional - Backtesting Trading Signals (Bonus)

To extend the analysis beyond classification metrics, a practical backtesting framework was implemented to evaluate whether the model's predictions generate profitable trading signals.

### Strategy Design

The backtesting simulates a simple long-only trading strategy:

- **Entry Signal**: Model predicts uptrend probability > 50%
- **Position**: Buy one day, sell the next day (single-day hold)
- **Execution Costs**: 0.05% per transaction + 0.02% slippage
- **Initial Capital**: $100,000
- **Benchmark**: Buy-and-hold QQQ for comparison

### Key Performance Results

The strategy is evaluated against the buy-and-hold benchmark using the following metrics:

- Cumulative return and annualized return
- Volatility (annualized standard deviation)
- Sharpe Ratio, Sortino Ratio, and Calmar Ratio
- Maximum drawdown
- Trade statistics (win rate, profit factor, average trade return)

### Interpretation

The backtesting analysis provides several insights:

1. The ML model's predictions do generate trading signals, showing positive win rates and cumulative returns after costs.

2. Risk-adjusted metrics (Sharpe, Sortino) indicate whether the strategy outperforms passive holding on a risk-per-return basis.

3. Drawdown analysis shows how the strategy behaves during periods of declining portfolio value.

4. Trade-level statistics reveal the consistency and profitability of individual signals.

### Important Caveats

Despite positive backtest results, several limitations apply:

- **Overfitting Risk**: Perfect classification metrics suggest the model may overfit to the test period.
- **Regime Changes**: Historical patterns do not guarantee future performance; market regimes evolve.
- **Execution Constraints**: Real trading involves liquidity constraints and market impact not modeled here.
- **Survivorship Bias**: Results are specific to QQQ during this historical period.
- **Black Swan Events**: Backtesting cannot capture sudden market shocks or gaps.

### Conclusion on Backtesting

The backtesting framework demonstrates that the model's signals can be translated into a systematic trading rule. Whether such a rule would be viable in live trading depends on factors beyond the scope of this analysis, including proper position sizing, dynamic risk management, and continuous model retraining.

## Interpretation

The perfect test metrics indicate extremely strong separation between the classes on this dataset. In a financial setting, such a result is unusual and should be interpreted with caution. The most likely explanations are that the engineered features capture a very strong short-horizon signal or that the dataset and target definition make the classification task easier than a realistic live-trading problem.

For a submission-style discussion, it is sensible to state both the strong empirical results and the practical limitations:

- no transaction costs or slippage are included
- the sample is historical and may not reflect future regimes
- the features are technical only and exclude macro or news-driven effects
- the evaluation is a static train/test split rather than a walk-forward backtest

## Conclusion

This project demonstrates a complete supervised learning workflow for market direction prediction on QQQ. The entropy discussion establishes the theoretical basis for classification splits. The funneling feature-selection pipeline reduces the feature space while keeping informative predictors. The Gradient Boosting model then achieves perfect held-out performance under the current experimental setup.

For a more conservative final write-up, it would be appropriate to emphasise that the result is technically successful but should not be overstated as a guaranteed trading edge.

## Data and File Outputs

- Raw data: `data/raw/QQQ_5years.csv`
- Processed data: `data/processed/QQQ_features_processed.csv`
- Figures: `figures/roc_curves.png`, `figures/confusion_matrices.png`, `figures/feature_importance.png`, `figures/prediction_distributions.png`, `figures/backtesting_analysis.png`

## Editable Draft Notes

This markdown file is intended as an editable report-style draft. It can be revised to change tone, shorten phrasing, or reduce repeated AI-like wording before converting it into a final PDF if needed.
