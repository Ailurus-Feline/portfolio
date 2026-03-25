from pathlib import Path
import pandas as pd
import pandas.api.types as ptypes
import numpy as np

from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler



# =========================================================
# Configuration
# =========================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

FIG_SIZE = (18, 7)  # default plot size
RATIO = 0.85        # threshold for type inference & distribution identification


# =========================================================
# Data Utils
# =========================================================

def init_df(s: pd.DataFrame, name: str, dir:str, label:str = None) -> None:
    """
    Attach metadata to df.

    Parameters:
        s (DataFrame): dataset
        name (str): dataset name
        dir (Path): output directory
        label (str): target column
    """
    s._name = name
    s._dir = dir
    s._label = label
    s._dir.mkdir(parents=True, exist_ok=True)

def load_data(name: str, file: str) -> pd.DataFrame:
    """
    Load raw CSV and initialize metadata.

    Parameters:
        name (str): dataset name
        file (str): file name
    """
    file_DIR = DATA_DIR / "Processed" / name
    path  = DATA_DIR / "Raw" / file
    df = pd.read_csv(path)
    
    init_df(df, name, file_DIR)
    print(f"{df._name} shape :", df.shape)

    return df

def save_data(s: pd.DataFrame, status: str = '') -> None:
    """
    Save DataFrame with a status suffix.

    Parameters:
        s (DataFrame)
        status (str): suffix tag
    """
    # reset index before saving
    if status:
      status = '_' + status
    s.to_csv(f"{s._dir}/{s._name}{status}.csv", index=False)


# =========================================================
# Load datasets
# =========================================================

df_completion = load_data("Completion", "Course_Completion_Prediction.csv")
df_consumption = load_data("Consumption", "online_learning_course_consumption_dataset.csv")
df_usage = load_data("Usage", "online_courses_uses.csv")

df_completion._label = "Completed"
df_consumption._label = "Completion_Status"
df_usage._label = "Category"

dfs = [df_completion, df_consumption, df_usage]


# =========================================================
# Preview datasets
# =========================================================

def preview_df(s: pd.DataFrame, n: int = 5) -> None:
    """
    Preview DataFrame with basic structure and sample rows.

    Parameters:
        s (DataFrame)
        n (int): number of rows to display
    """
    print(f"\n========== {s._name} Dataset ==========")
    print("Shape: ", s.shape)
    print("\nColumns:")
    print(s.columns.tolist())
    print("\nFirst few rows:")
    display(s.head(n))


for df in dfs:
    preview_df(df)

# =========================================================
# Column Type Recognition
# =========================================================

def get_cols(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Recognize numeric & non-numeric columns.

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
# - quick check on raw data distribution
# =========================================================
def show(s: pd.DataFrame) -> None: 
    """
    Quick EDA visualization for dataset inspection.
    """
    num_col, str_col = get_cols(s)

    # Label Distribution
    tmp_col = s[s._label].value_counts()
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    fig.canvas.manager.set_window_title(s._name)

    bars = ax.bar(tmp_col.index, tmp_col.values)
    ax.bar_label(bars)
    ax.set_title(f"{s._label} Distribution")
    plt.show()

    # Numerical Distribution
    ax = s[num_col].hist(bins=15, figsize=FIG_SIZE)
    fig = ax[0][0].figure
    fig.canvas.manager.set_window_title(s._name)

    plt.suptitle("Numerical Features Distribution")
    plt.tight_layout()
    plt.show()

    # Categorical Distribution
    for l in str_col:
        if s[l].nunique() > 10:
            continue
        if l == s._label:
            continue

        tmp_col = s[l].value_counts()
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        fig.canvas.manager.set_window_title(s._name)

        bars = ax.bar(tmp_col.index, tmp_col.values)
        ax.bar_label(bars)
        ax.set_title(l + " Distribution")
        plt.show()

for df in dfs:
    show(df)



# =========================================================
# Step 1 — Structural Cleaning & Type Inference
# - remove duplicates
# - infer numeric / datetime columns
# - drop ID-like columns
# =========================================================

def set_col_type(s: pd.DataFrame) -> list[str]:
    """
    Infer column types and detect ID-like columns.
    """
    s.drop_duplicates(inplace=True)

    id_cols = []

    for l in s.columns.to_list():
        tmp_col = s[l]

        # skip numeric columns
        if pd.api.types.is_numeric_dtype(tmp_col):
            continue

        # attempt numeric coercion
        tmp_numeric = pd.to_numeric(tmp_col, errors="coerce")
        numeric_ratio = tmp_numeric.notna().mean()
        if numeric_ratio > RATIO: 
            s[l] = tmp_numeric
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
            s[l + "_year"] = tmp_date.dt.year
            s[l + "_month"] = tmp_date.dt.month
            s[l + "_day"] = tmp_date.dt.day
            s[l + "_weekday"] = tmp_date.dt.weekday

            s.drop(columns=[l], inplace=True)
            continue

        # detect ID-like columns (high uniqueness)
        if tmp_col.nunique() > len(s) * RATIO:
            id_cols.append(l)
    
    return id_cols

for df in dfs:
    id = set_col_type(df)
    df.drop(columns=id, inplace=True)
    df.dropna(subset=[df._label], inplace=True)
    print(f"\n\n\n{df._name} ID-like columns: {id}")
    save_data(df, "Cleaned")



# =========================================================
# Dataset Split
# - create train/test split on primary dataset
# - prevent data leakage
# =========================================================
train_df, test_df = train_test_split(
    df_completion,
    test_size=0.2,
    random_state=42,
    stratify=df_completion[df_completion._label]
)

init_df(train_df, "Train", DATA_DIR / "Processed" / "Train", df_completion._label)
init_df(test_df, "Test", DATA_DIR / "Processed" / "Test", df_completion._label)

save_data(train_df)
save_data(test_df)



# =========================================================
# Step 2 — Missing Value Handling & Alignment
# - fill NA
# - remove empty rows & columns
# - align test schema with train
# =========================================================

def dealna(s: pd.DataFrame, map: dict[str, float] = None) -> dict[str, float]:
    """
    Fill missing values.

    Parameters:
        s (DataFrame)
        map (dict): transformation rules from train_df

    Returns:
        median_map (dict)
    """
    num_col, str_col = get_cols(s)

    medianmap = {}

    # numeric: median imputation
    print("\n\n\n")
    for l in num_col:
        print(f"{s._name} {l} Numeric NA: {s[l].isna().sum()}")

        if not map:
            medianmap[l] = s[l].median()
            s[l] = s[l].fillna(medianmap[l])
        else:
            s[l] = s[l].fillna(map[l])

    # categorical: mark missing explicitly
    print("\n\n\n")
    for l in str_col:
        print(f"{s._name} {l} String NA: {s[l].isna().sum()}")
        s[l] = s[l].fillna(pd.NA)
    
    return medianmap

median_map = dealna(train_df)

# apply to other df
dealna(test_df, median_map)
dealna(df_consumption)
dealna(df_usage)


def removena(s: pd.DataFrame) -> pd.DataFrame:
    """
    Remove empty rows and columns.
    """
    s.dropna(axis=1, how="all", inplace=True)
    s.dropna(axis=0, how="all", inplace=True)

    return s

removena(train_df)
removena(df_consumption)
removena(df_usage)

# align test columns with train
test_df.drop(columns=test_df.columns.difference(train_df.columns), inplace=True)
test_df.dropna(axis=0, how="all", inplace=True)

save_data(train_df, "No_NA")
save_data(test_df, "No_NA")
save_data(df_consumption, "No_NA")
save_data(df_usage, "No_NA")

# quick check
dfs[0] = train_df
for i in dfs:
    show(i)



# =========================================================
# Step 3 — Feature Scaling & Encoding
# - classify feature types
# - define transformation strategy
# - apply scaling and encoding
# =========================================================

def which_distribution(l: pd.Series) -> str:
    """
    Classify numeric columns by value range and empirical distribution.
    """
    min_val, max_val = l.min(), l.max()
    
    # fixed-range detection
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
    
    # statistical properties
    skew = l.skew()
    unique_ratio = l.nunique() / len(l)
        
    if unique_ratio < (1 - RATIO) and ptypes.is_integer_dtype(l):
        return "discrete"
        
    if skew > 1:
        return "right_skewed"
    
    if skew > -0.5 and skew < 0.5:
        return "normal"
    
    return "general"

def which_card(l: pd.Series) -> str:
    """
    Classify categorical columns by cardinality level.
    """
    unique = l.nunique(dropna=True)
    
    if unique <= 10:
        return "low_card"
    
    if unique <= 50 and unique / len(l) < RATIO:
        return "medium_card"
    
    return "high_card"

def get_range(s: pd.Series) -> tuple[float, float]:
    """
    Compute IQR-based clipping bounds with Tukey's fences.
    """
    Q1 = s.quantile(0.25)
    Q3 = s.quantile(0.75)
    IQR = Q3 - Q1
        
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    return lower, upper

def scale(s: pd.DataFrame, map: dict[str, tuple] = None) -> dict[str, tuple]:
    """
    Apply numeric scaling rules.

    Parameters:
        s (DataFrame)
        map (dict): transformation rules from train_df

    Returns:
        scalemap (dict)
    """
    num_col, _ = get_cols(s)
    scalemap = {}
    
    if map:
        # for test_df
        for l in num_col:
            col = s[l]
            rule = map[l]
            
            # simple clipping
            if len(rule) == 2:
                lower, upper = rule
                s[l] = col.clip(lower, upper)
            
            # log + clipping
            else:
                _, upper, _ = rule
                col_log = np.log1p(col.clip(lower=0))
                s[l] = np.expm1(col_log.clip(0, upper))
        
        return scalemap

    print("\n\n\n")
    for l in num_col:
        col = s[l]
        distr = which_distribution(col)
        print(f"{s._name} {l} Numeric Distribution: {distr}")
    
        # fixed range
        if distr == "ratio":
            scalemap[l] = (0, 1)
            s[l] = col.clip(0, 1)        
            continue
        
        if distr == "rating":
            scalemap[l] = (0, 5)
            s[l] = col.clip(0, 5)
            continue
        
        if distr == "percentage":
            scalemap[l] = (0, 100)
            s[l] = col.clip(0, 100)
            continue
        
        if distr == "discrete":
            scalemap[l] = (0, None)
            s[l] = col.clip(lower=0)
            continue
        
        # right_skewed stabilization
        if distr == "right_skewed":
            col_log = np.log1p(col.clip(lower=0))
            _, upper = get_range(col_log)
            scalemap[l] = (0, upper, "log")
            s[l] = np.expm1(col_log.clip(0, upper))
            continue
    
        # general numeric features
        lower, upper = get_range(col)  
        scalemap[l] = (lower, upper)
        s[l] = col.clip(lower, upper)
    
    return scalemap

scale_map = scale(train_df)

# apply to other df
scale(test_df, scale_map)
scale(df_consumption)
df_consumption[df_consumption._label] = (df_consumption[df_consumption._label] != "Dropped").astype(int)
scale(df_usage)


def encoding(s: pd.DataFrame, map: dict = None, type: dict[str, str] = None) -> tuple:
    """
    Encode categorical features.

    Parameters:
        s (DataFrame)
        map (dict): transformation rules from train_df
        type (dict): cardinality type of each column

    Returns:
        encodingmap (dict), card_type (dict)
    """
    _, str_col = get_cols(s)
    encodingmap = {}
    card_type = {}
    
    if map:
        # for test_df
        for l in str_col:
            s[l] = s[l].fillna("Missing").map(map[l])

        return encodingmap, card_type
        
    print("\n\n\n")
    for l in str_col:
        col = s[l].fillna("Missing")
        stand = which_card(col)
        print(f"{s._name} {l} Category Distribution: {stand}")
        card_type[l] = stand

        # ordinal category for low- and medium-cardinality features
        if stand in ["low_card", "medium_card"]:
            s[l] = s[l].astype("category")
            encodingmap[l] = {cat: i for i, cat in enumerate(s[l].cat.categories)}
        
            s[l] = s[l].astype("category").cat.codes
    
        # frequency encoding for high-cardinality features
        else:
            freq = col.value_counts(normalize=True)
            encodingmap[l] = freq
            s[l] = s[l].map(freq)
    
    return encodingmap, card_type

encoding_map, card_type = encoding(train_df)

# apply to other df
encoding(test_df, encoding_map, card_type)
test_df.drop(columns=test_df.columns.difference(train_df.columns), inplace=True)
encoding(df_consumption)
encoding(df_usage)

save_data(train_df, "Scaled")
save_data(test_df, "Scaled")
save_data(df_consumption, "Scaled")
save_data(df_usage, "Scaled")

# quick check
for i in dfs:
    show(i)



# =========================================================
# Step 4 — Feature Engineering
# - remove non-informative columns
# - remove redundant columns
# =========================================================

def drop_constant(s: pd.DataFrame) -> pd.DataFrame:
    """
    Remove constant columns.
    """
    constant_col = [l for l in s.columns if s[l].nunique() <= 1]
    print(f"\n\n\n{s._name} Constant Columns: {constant_col}")
    s.drop(columns=constant_col, inplace=True)

    return s

drop_constant(train_df)
drop_constant(df_consumption)
drop_constant(df_usage)


def drop_redundant(s: pd.DataFrame) -> pd.DataFrame:
    """
    Remove redundant features.
    """
    corr_matrix = s.corr()
    redundant_col = set()
    col = corr_matrix.columns

    for i in range(len(col) - 1):
        if col[i] in redundant_col:
            continue

        for j in range(i + 1, len(col)):
            if col[j] in redundant_col:
                continue

            c1, c2 = col[i], col[j]

            # linear redundancy
            if np.isclose(abs(corr_matrix.iloc[i, j]), 1.0):
                redundant_col.add(c2)
                continue

            # functional redundancy
            if s.groupby(c1)[c2].nunique().max() == 1 and s.groupby(c2)[c1].nunique().max() == 1:
                redundant_col.add(c2)
                continue

    redundant_col = list(redundant_col)
    print(f"\n\n\n{s._name} Redundant Columns: {redundant_col}")
    s.drop(columns=redundant_col, inplace=True)
    
    return s

drop_redundant(train_df)

# apply to other df
test_df.drop(columns=test_df.columns.difference(train_df.columns), inplace=True)
drop_redundant(df_consumption)
drop_redundant(df_usage)

save_data(train_df, "Dropped")
save_data(test_df, "Dropped")
save_data(df_consumption, "Dropped")
save_data(df_usage, "Dropped")

# quick check
for i in dfs:
    show(i)



# =========================================================
# Step 5 — Dataset Split
# - separate features and label
# =========================================================

X_train = train_df.drop(columns=[train_df._label])
y_train = train_df[train_df._label]

init_df(X_train, "X_train", train_df._dir / "X_train")
init_df(y_train, "y_train", train_df._dir / "y_train")

save_data(X_train)
save_data(y_train)

X_test  = test_df.drop(columns=[test_df._label])
y_test  = test_df[test_df._label]

init_df(X_test, "X_test", test_df._dir / "X_test")
init_df(y_test, "y_test", test_df._dir / "y_test")

save_data(X_test)
save_data(y_test)



# =========================================================
# Step 6 — Numeric Standardization
# - normalize feature scale for model stability
# =========================================================

num_train, _ = get_cols(X_train)
num_test, _ = get_cols(X_test)

scaler = StandardScaler()
X_train[num_train] = scaler.fit_transform(X_train[num_train])
X_test[num_test]  = scaler.transform(X_test[num_test])

save_data(X_train, "Transformed")
save_data(X_test, "Transformed")



# =========================================================
# Step 7 — Feature Selection (EDA-driven)
# - select features based on class separation
# =========================================================

def feature_rank(s: pd.DataFrame) -> list[str]:
    """
    Rank numeric features by standardized mean difference.

    Returns:
        feature (str)
    """
    num_col, _ = get_cols(s)

    # class-wise mean
    group_mean = s.groupby(s._label)[num_col].mean()
    print(f"\n\n\n{s._name} Group Mean: \n{group_mean}")

    std = s[num_col].std()

    # standardized mean difference
    delta = (group_mean.loc[0] - group_mean.loc[1]).abs() / std

    num_df = pd.DataFrame({
        "0_mean": group_mean.loc[0],
        "1_mean": group_mean.loc[1],
        "Std": std,
        "Std_Delta": delta
    }).sort_values(by="Std_Delta", ascending=False)
    print(f"\n\n\n{s._name} Feature Contribution: \n{num_df["Std_Delta"].T}")


    # filter weak features
    feature = num_df[num_df["Std_Delta"] > 0.05].index.to_list()

    # ensure valid columns
    feature = [f for f in feature if f in s.columns and f != s._label]

    print(f"\n\n\n{s._name} Selected Features: ")
    print(feature)

    return feature

feature_main = feature_rank(train_df)
feature_consumption = feature_rank(df_consumption)
feature_usage = feature_rank(df_usage)



# =========================================================
# Step 8 — Baseline Model (Logistic Regression)
# =========================================================

# use selected features only
X_train_feature = X_train[feature_main]
X_test_feature = X_test[feature_main]

X_train_feature.to_csv(f"{X_train._dir}/X_train_Features.csv", index=False)
X_test_feature.to_csv(f"{X_test._dir}/X_test_Features.csv", index=False)

# initialize model
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",   # handle potential imbalance
    random_state=42
)

# train
model.fit(X_train_feature, y_train)



# =========================================================
# Step 9 — Prediction & Evaluation
# =========================================================

# Prediction
y_pred = model.predict(X_test_feature)

# Evaluation
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
# Step 10 — Visualization & Feature Importance
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
fig.canvas.manager.set_window_title("Logistic Regression")
plt.show()


# Feature Importance (Logistic Regression Coefficients)
coef_df = pd.DataFrame({
    "Feature": X_train_feature.columns,
    "Coefficient": model.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

print("\n\nTop Positive Features:")
print(coef_df.head(10))

print("\nTop Negative Features:")
print(coef_df.tail(10))



# =========================================================
# Step 11 — Advanced Model (Random Forest)
# =========================================================

# initialize model
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# train
rf_model.fit(X_train_feature, y_train)



# =========================================================
# Step 12 — Prediction & Evaluation (RF)
# =========================================================

# Prediction
y_pred_rf = rf_model.predict(X_test_feature)

# Evaluation
acc_rf = accuracy_score(y_test, y_pred_rf)
cm_rf  = confusion_matrix(y_test, y_pred_rf)
report_rf = classification_report(y_test, y_pred_rf)

print("\n\n========== Random Forest Performance ==========")
print(f"Accuracy: {acc_rf:.4f}")

print("\nConfusion Matrix:")
print(cm_rf)

print("\nClassification Report:")
print(report_rf)



# =========================================================
# Step 13 — Visualization & Feature Importance (RF)
# =========================================================

fig, ax = plt.subplots(figsize=FIG_SIZE)
sns.heatmap(
    cm_rf,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["Not Completed", "Completed"],
    yticklabels=["Not Completed", "Completed"],
    ax=ax
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Random Forest Confusion Matrix")
fig.canvas.manager.set_window_title("Random Forest")
plt.show()


# Feature Importance (Random Forest)
importance_df = pd.DataFrame({
    "Feature": X_train_feature.columns,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\n\nTop Important Features (RF):")
print(importance_df.head(10))



# =========================================================
# Step 14 — Advanced Model (XGBoost)
# =========================================================

# initialize model
xgb_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)

# train
xgb_model.fit(X_train_feature, y_train)



# =========================================================
# Step 15 — Prediction & Evaluation (XGBoost)
# =========================================================

# Prediction
y_pred_xgb = xgb_model.predict(X_test_feature)

# Evaluation
acc_xgb = accuracy_score(y_test, y_pred_xgb)
cm_xgb  = confusion_matrix(y_test, y_pred_xgb)
report_xgb = classification_report(y_test, y_pred_xgb)

print("\n\n========== XGBoost Performance ==========")
print(f"Accuracy: {acc_xgb:.4f}")

print("\nConfusion Matrix:")
print(cm_xgb)

print("\nClassification Report:")
print(report_xgb)



# =========================================================
# Step 16 — Visualization & Feature Importance (XGB)
# =========================================================

fig, ax = plt.subplots(figsize=FIG_SIZE)
sns.heatmap(
    cm_xgb,
    annot=True,
    fmt="d",
    cmap="Oranges",
    xticklabels=["Not Completed", "Completed"],
    yticklabels=["Not Completed", "Completed"],
    ax=ax
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("XGBoost Confusion Matrix")
plt.show()


# Feature Importance (XGBoost)
importance_df_xgb = pd.DataFrame({
    "Feature": X_train_feature.columns,
    "Importance": xgb_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\n\nTop Important Features (XGB):")
print(importance_df_xgb.head(10))