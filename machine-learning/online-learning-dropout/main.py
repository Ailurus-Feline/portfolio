from pathlib import Path
import pandas as pd
import pandas.api.types as ptypes
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler


# =========================================================
# Configuration
# =========================================================
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
(COMPLETION_DIR := DATA_DIR / "Processed" / "Completion").mkdir(parents=True, exist_ok=True)

FIG_SIZE = (18, 7) 
TOP_K_CAT = 10  # number of top features retained after ranking
RATIO = 0.9     # threshold for type inference / high-cardinality detection
MAPPING = {}    # stores categorical encoding mappings
CARD_TYPE = {}  # Cardinality class per categorical feature (fit on train)

pd.set_option('display.max_columns', None)   # Show all columns
pd.set_option('display.width', None)         # Prevent automatic line wrapping
pd.set_option('display.max_colwidth', None)  # Do not truncate column content


# =========================================================
# Data Loading
# =========================================================
completion_path  = DATA_DIR / "Course_Completion_Prediction.csv"
consumption_path = DATA_DIR / "online_learning_course_consumption_dataset.csv"
usage_path       = DATA_DIR / "online_courses_uses.csv"

df_completion  = pd.read_csv(completion_path)
df_consumption = pd.read_csv(consumption_path)
df_usage       = pd.read_csv(usage_path)

print("Completion :", df_completion.shape)
print("Consumption:", df_consumption.shape)
print("Usage      :", df_usage.shape)


# =========================================================
# Utility: Column Type Split
# =========================================================
num_cols = []
str_cols = []
id_cols = []

def get_cols(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Split dataframe columns into numeric and non-numeric.

    Returns:
        num_cols (list): numeric feature names
        str_cols (list): non-numeric feature names
    """
    num_cols = []
    str_cols = []

    for l in df.columns:
        if ptypes.is_numeric_dtype(df[l]):
            num_cols.append(l)
            continue
        else:
            str_cols.append(l)
    
    return num_cols, str_cols



# =========================================================
# Step 0 — Initial Distribution Snapshot
# - quick sanity check on raw data distribution
# =========================================================
def show(s: pd.DataFrame) -> None: 
    tmp_col = s["Completed"].value_counts()
    _, ax = plt.subplots(figsize=FIG_SIZE)
    bars = ax.bar(tmp_col.index, tmp_col.values)
    ax.bar_label(bars)
    ax.set_title("Completion Distribution")
    plt.show()

    numerics = s.select_dtypes(include=["int64", "float64"]).columns
    s[numerics].hist(bins=15, figsize=FIG_SIZE)
    plt.suptitle("Numerical Features Distribution")
    plt.tight_layout()
    plt.show()

    for l in s.select_dtypes(exclude=["int64", "float64"]).columns:
        if s[l].nunique() > 10:
            continue
        if l == "Completed":
            continue

        tmp_col = s[l].value_counts()
        _, ax = plt.subplots(figsize=FIG_SIZE)
        bars = ax.bar(tmp_col.index, tmp_col.values)
        ax.bar_label(bars)
        ax.set_title(l + " Distribution")
        plt.show()

show(df_completion)



# =========================================================
# Step 1 — Structural Cleaning
# - remove duplicates
# - infer types (numeric / datetime)
# - drop ID-like columns
# =========================================================
df_completion = df_completion.drop_duplicates()

for l in df_completion.columns.to_list():
    tmp_col = df_completion[l]

    # skip numeric columns
    if pd.api.types.is_numeric_dtype(tmp_col):
        continue

    # attempt numeric coercion
    tmp_numeric = pd.to_numeric(tmp_col, errors="coerce")
    numeric_ratio = tmp_numeric.notna().mean()
    if numeric_ratio > RATIO: 
        df_completion[l] = tmp_numeric
        continue

    # attempt datetime parsing and feature extraction
    tmp_date = pd.to_datetime(
            tmp_col, 
            errors="coerce", 
            cache=True,
            format="mixed"
        )
    date_ratio = tmp_date.notna().mean()
    if date_ratio > RATIO:
        df_completion[l + "_year"] = tmp_date.dt.year
        df_completion[l + "_month"] = tmp_date.dt.month
        df_completion[l + "_day"] = tmp_date.dt.day
        df_completion[l + "_weekday"] = tmp_date.dt.weekday

        df_completion.drop(columns=[l], inplace=True)
        continue

    # detect ID-like columns (high uniqueness)
    if df_completion[l].nunique() > len(df_completion) * RATIO:
        id_cols.append(l)

# drop ID columns
df_completion = df_completion.drop(columns=id_cols)
print(f"\n\n\nID-like columns: {id_cols}")

# enforce target availability
df_completion = df_completion.dropna(subset=["Completed"])
df_completion.to_csv(f"{COMPLETION_DIR}/Completion_Cleaned.csv", index=False)


# =========================================================
# Dataset Split
# =========================================================
train_df, test_df = train_test_split(
    df_completion,
    test_size=0.2,
    random_state=42,
    stratify=df_completion["Completed"]
)



# =========================================================
# Step 2 — Missing Value Handling
# =========================================================
num_cols, str_cols = get_cols(train_df)

# store statistics from train
median_map = {}

# numeric: median imputation
print("\n\n\n")
for l in num_cols:
    median_map[l] = train_df[l].median()
    print(f"Train {l} Numeric NA: {train_df[l].isna().sum()}")
    train_df[l] = train_df[l].fillna(median_map[l])

# categorical: mark missing explicitly
print("\n\n\n")
for l in str_cols:
    print(f"Train {l} String NA: {train_df[l].isna().sum()}")
    train_df[l] = train_df[l].fillna(pd.NA)

# apply same transformation to test
print("\n\n\n")
for l in num_cols:
    print(f"Test {l} Numeric NA: {test_df[l].isna().sum()}")
    test_df[l] = test_df[l].fillna(median_map[l])

print("\n\n\n")
for l in str_cols:
    print(f"Test {l} String NA: {test_df[l].isna().sum()}")
    test_df[l] = test_df[l].fillna(pd.NA)

# remove empty rows/columns (based on train only)
train_df = train_df.dropna(axis=1, how="all")
train_df = train_df.dropna(axis=0, how="all")

# align test columns with train
test_df = test_df[train_df.columns]
train_df.to_csv(f"{COMPLETION_DIR}/Train_Cleaned.csv", index=False)
test_df.to_csv(f"{COMPLETION_DIR}/Test_Cleaned.csv", index=False)


num_cols, str_cols = get_cols(train_df)



# =========================================================
# Step 3 — Quick Distribution Check
# - visualize target and feature distributions
# - used for sanity check
# =========================================================

show(train_df)



# =========================================================
# Step 4 — Feature Scaling and Encoding Rules
# =========================================================
def which_distribution(col: pd.Series) -> str:
    """
    Classify numeric columns by value range and empirical distribution.
    """
    min_val, max_val = col.min(), col.max()
    
    if 0 <= min_val and max_val <= 1:
        return "ratio"
    if 0 <= min_val and max_val <= 5:
        return "rating"
    if 0 <= min_val and max_val <= 6:
        return "weekday"
    if 0 <= min_val and max_val <= 30:
        return "date"
    if 0 <= min_val and max_val <= 11:
        return "month"
    if 0 <= min_val and max_val <= 100:
        return "percentage"
    if 1990 <= min_val and max_val <= 2100:
        return "year"
        
    skew = col.skew()
    unique_ratio = col.nunique() / len(col)
        
    if unique_ratio < (1 - RATIO) and ptypes.is_integer_dtype(col):
        return "discrete"
        
    if skew > 1:
        return "long_tail"
    
    if skew > -0.5 and skew < 0.5:
        return "normal"
    
    return "general"

def which_card(col: pd.Series) -> str:
    """
    Classify categorical columns by cardinality level.
    """
    unique = col.nunique(dropna=True)
    
    if unique <= 10:
        return "low_card"
    
    if unique <= 50 and unique / len(col) < 0.7:
        return "medium_card"
    
    return "high_card"



print("\n\n\n")
for l in num_cols:
    s = train_df[l]
    distr = which_distribution(s)
    print(f"Train {l} Numeric Distribution: {distr}")
    
    def get_range(s: pd.Series) -> tuple[float, float]:
        """
        Compute IQR-based clipping bounds.
        """
        Q1 = s.quantile(0.25)
        Q3 = s.quantile(0.75)
        IQR = Q3 - Q1
        
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        return lower, upper

    # apply fixed-range clipping for bounded numeric features
    if distr == "ratio":
        MAPPING[l] = (0, 1)
        train_df[l] = s.clip(0, 1)
        continue
        
    if distr == "rating":
        MAPPING[l] = (0, 5)
        train_df[l] = s.clip(0, 5)
        continue
        
    if distr == "percentage":
        MAPPING[l] = (0, 100)
        train_df[l] = s.clip(0, 100)
        continue
        
    if distr == "discrete":
        MAPPING[l] = (0, None)
        train_df[l] = s.clip(lower=0)
        continue
        
    # stabilize long-tail features before clipping
    if distr == "long_tail":
        s_log = np.log1p(s.clip(lower=0))
        _, upper = get_range(s_log)
        MAPPING[l] = (0, upper, "log")
        train_df[l] = np.expm1(s_log.clip(0, upper))
        continue
    
    # apply IQR-based clipping for general numeric features
    lower, upper = get_range(s)  
    MAPPING[l] = (lower, upper)
    train_df[l] = s.clip(lower, upper)

# apply same clipping to test
for l in num_cols:
    s = test_df[l]

    rule = MAPPING[l]

    if len(rule) == 2:
        lower, upper = rule
        test_df[l] = s.clip(lower, upper)
    else:
        _, upper, _ = rule
        s_log = np.log1p(s.clip(lower=0))
        test_df[l] = np.expm1(s_log.clip(0, upper))


# categorical encoding
print("\n\n\n")
for l in str_cols:
    s = train_df[l].fillna("Missing")
    stand = which_card(s)
    print(f"Train {l} Category Distribution: {stand}")
    CARD_TYPE[l] = stand

    # use ordinal category codes for low- and medium-cardinality features
    if stand in ["low_card", "medium_card"]:
        train_df[l] = train_df[l].astype("category")
        MAPPING[l] = {cat: i for i, cat in enumerate(train_df[l].cat.categories)}
        
        train_df[l] = train_df[l].cat.codes
    
    # use frequency encoding for high-cardinality features
    else:
        freq = s.value_counts(normalize=True)
        MAPPING[l] = freq
        train_df[l] = s.map(freq)

      
# apply same mapping to test
for l in str_cols:
    s = test_df[l].fillna("Missing")
    stand = CARD_TYPE[l]

    if stand in ["low_card", "medium_card"]:
        test_df[l] = test_df[l].map(MAPPING[l])

    else:
        test_df[l] = s.map(MAPPING[l])


# validate target and feature matrix consistency
assert train_df["Completed"].isin([0,1]).all()
assert not train_df.isna().any().any()
assert all(ptypes.is_numeric_dtype(train_df[c]) for c in train_df.columns)

train_df = train_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)

train_df.to_csv(f"{COMPLETION_DIR}/Train_Scaled.csv", index=False)
test_df.to_csv(f"{COMPLETION_DIR}/Test_Scaled.csv", index=False)



# =========================================================
# Step 5 — Feature Engineering
# =========================================================

# drop constant columns
constant_cols = [l for l in train_df.columns if train_df[l].nunique() <= 1]
print(f"\n\n\nConstant Columns: {constant_cols}")
train_df = train_df.drop(columns=constant_cols)


# =========================================================
# Remove Perfectly Redundant Features
# =========================================================

# detect columns with identical values
corr_matrix = train_df.corr()

redundant_cols = set()
cols = corr_matrix.columns


for i in range(len(cols) - 1):
    if cols[i] in redundant_cols:
        continue

    for j in range(i + 1, len(cols)):
        if cols[j] in redundant_cols:
            continue

        c1, c2 = cols[i], cols[j]

        # linear redundancy
        if np.isclose(abs(corr_matrix.iloc[i, j]), 1.0):
            redundant_cols.add(c2)
            continue

        # functional redundancy: c1 → c2
        if train_df.groupby(c1)[c2].nunique().max() == 1:
            redundant_cols.add(c2)
            continue

        # functional redundancy: c2 → c1
        if train_df.groupby(c2)[c1].nunique().max() == 1:
            redundant_cols.add(c2)
            break

redundant_cols = list(redundant_cols)
print(f"\n\n\nRedundant Columns: {redundant_cols}")

# separate features and target
train_df = train_df.drop(columns=redundant_cols)
test_df = test_df[train_df.columns]

train_df.to_csv(f"{COMPLETION_DIR}/Train_Dropped.csv", index=False)
test_df.to_csv(f"{COMPLETION_DIR}/Test_Dropped.csv", index=False)

X_train = train_df.drop(columns=["Completed"])
y_train = train_df["Completed"]

X_train.to_csv(f"{COMPLETION_DIR}/X_train.csv", index=False)
y_train.to_csv(f"{COMPLETION_DIR}/y_train.csv", index=False)

X_test  = test_df.drop(columns=["Completed"])
y_test  = test_df["Completed"]

X_test.to_csv(f"{COMPLETION_DIR}/X_test.csv", index=False)
y_test.to_csv(f"{COMPLETION_DIR}/y_test.csv", index=False)



# =========================================================
# Numeric Feature Scaling
# =========================================================
scaler = StandardScaler()
num_train, _ = get_cols(X_train)
num_test, _ = get_cols(X_test)

X_train[num_train] = scaler.fit_transform(X_train[num_train])
X_test[num_test]  = scaler.transform(X_test[num_test])

X_train.to_csv(f"{COMPLETION_DIR}/X_train_Transformed.csv", index=False)
X_test.to_csv(f"{COMPLETION_DIR}/X_test_Transformed.csv", index=False)


# =========================================================
# Step 6 — EDA-based Feature Pre-screening
# =========================================================

numerics, _ = get_cols(train_df)


# =========================================================
# Feature Ranking
# =========================================================

# compute class-wise mean (binary target)
group_mean = train_df.groupby("Completed")[numerics].mean()
print(f"\n\n\nGroup Mean: {group_mean}")

# compute global std (avoid division by 0)
std = train_df[numerics].std()

# standardized mean difference (scale-invariant)
delta = (group_mean.loc[0] - group_mean.loc[1]).abs() / std

# format output
num_df = pd.DataFrame({
    "0_mean": group_mean.loc[0],
    "1_mean": group_mean.loc[1],
    "Std": std,
    "Std_Delta": delta
}).sort_values(by="Std_Delta", ascending=False)

features = num_df[num_df["Std_Delta"] > 0.05].index.to_list()

# ensure features exist in processed dataset
features = [f for f in features if f in train_df.columns and f != "Completed"]

print("\n\n\nSelected Features: ")
print(features)



# =========================================================
# Step 7 — Baseline Model (Logistic Regression)
# =========================================================

# use selected features only
X_train_sel = X_train[features]
X_test_sel  = X_test[features]

X_train_sel.to_csv(f"{COMPLETION_DIR}/X_train_Features.csv", index=False)
X_test_sel.to_csv(f"{COMPLETION_DIR}/X_test_Features.csv", index=False)

# initialize model
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",   # handle potential imbalance
    random_state=42
)

# train
model.fit(X_train_sel, y_train)


# =========================================================
# Prediction
# =========================================================

y_pred = model.predict(X_test_sel)
y_prob = model.predict_proba(X_test_sel)[:, 1]   # probability of class 1


# =========================================================
# Evaluation
# =========================================================

acc = accuracy_score(y_test, y_pred)
cm  = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\n\n========== Model Performance ==========")
print(f"Accuracy: {acc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(report)



# =========================================================
# Visualization — Confusion Matrix
# =========================================================

fig, ax = plt.subplots(figsize=FIG_SIZE)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Completed", "Completed"],
    yticklabels=["Not Completed", "Completed"],
    ax=ax
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
plt.show()



# =========================================================
# Feature Importance (Logistic Regression Coefficients)
# =========================================================

coef_df = pd.DataFrame({
    "Feature": X_train_sel.columns,
    "Coefficient": model.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

print("\n\nTop Positive Features:")
print(coef_df.head(10))

print("\nTop Negative Features:")
print(coef_df.tail(10))