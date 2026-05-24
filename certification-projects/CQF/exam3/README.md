# CQF Exam 3 - Machine Learning

This folder contains the working materials for **CQF Exam 3**, which studies whether short-term positive moves in **QQQ** (Nasdaq 100 ETF) can be predicted with a supervised learning pipeline.

The submission is organized to match the exam structure:

- **Section A**: Entropy in classification
- **Section B**: Feature selection with a funneling approach
- **Section C**: Model building, tuning, and evaluation

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
   - Retains statistically relevant engineered features.

2. **Wrapper stage**
   - Gradient boosting model fit on the filtered set.
   - Uses model-derived importance to refine the subset.

3. **Embedded stage**
   - Final selection based on non-zero model importance.
   - Produces the final feature set used in the model.

### Section C - Model Building and Evaluation

- Trains a `GradientBoostingClassifier`.
- Tunes hyperparameters with `GridSearchCV`.
- Evaluates the model using ROC-AUC, accuracy, precision, recall, F1-score, and confusion matrices.
- Generates interpretation plots for reporting.

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

The final notebook reports the following test-set metrics:

- ROC-AUC: 1.0000
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-score: 1.0000

Saved figures include:

- `roc_curves.png`
- `confusion_matrices.png`
- `feature_importance.png`
- `prediction_distributions.png`

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
    └── prediction_distributions.png
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

Mao Yikai
