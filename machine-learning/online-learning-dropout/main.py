from pathlib import Path
import pandas as pd
import pandas.api.types as ptypes
import numpy as np

from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler



# =========================================================
# Step 1 — Setup & Data Loading
# - configure paths and utilities
# - load datasets and initialize metadata
# =========================================================

# Configuration
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

FIG_SIZE = (18, 7)  # default plot size
RATIO = 0.85        # threshold for type inference & distribution identification


# Data Utils
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

    return df


def save_data(s: pd.DataFrame, status: str = '') -> None:
    """
    Save DataFrame with a status suffix.

    Parameters:
        s (DataFrame)
        status (str): suffix tag
    """
    # Reset index before saving
    if status:
        status = '_' + status
    s.to_csv(f"{s._dir}/{s._name}{status}.csv", index=False)


# Load datasets and assign labels
df_completion = load_data("Completion", "Course_Completion_Prediction.csv")
df_consumption = load_data("Consumption", "online_learning_course_consumption_dataset.csv")
df_usage = load_data("Usage", "online_courses_uses.csv")

df_completion._label = "Completed"
df_consumption._label = "Completion_Status"
df_usage._label = "Completion_Rate (%)"

dfs = [df_completion, df_consumption, df_usage]



# =========================================================
# Step 2 — Data Exploration & Visualization
# - preview dataset structure and content
# - visualize class and feature distributions
# =========================================================

def preview(s: pd.DataFrame, n: int = 5) -> None:
    """
    Preview df with basic structure, data quality, and sample rows.

    Parameters:
        s (DataFrame)
        n (int): number of rows to display
    """
    print(f"========== Dataset {s._name} ==========")

    # Show basic information
    row_cnt, col_cnt = s.shape
    print(f"Shape: {row_cnt:,} rows × {col_cnt} columns")
    print("Columns:")
    print(s.columns.tolist())
    
    # Quick data quality metrics
    missing_total = s.isnull().sum().sum()
    complete_ratio = (1 - missing_total / (row_cnt * col_cnt)) * 100
    print(f"Completeness: {complete_ratio:.2f}%")
    
    print("First few rows:")
    display(s.head(n))

for df in dfs:
    preview(df)
    print("\n\n\n")


def get_cols(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Recognize numeric & non-numeric columns.

    Returns:
        num_cols (list): numeric feature names
        str_cols (list): non-numeric feature names
    """
    num_cols, str_cols = [], []

    for l in df.columns:
        if ptypes.is_numeric_dtype(df[l]):
            num_cols.append(l)
            continue
        else:
            str_cols.append(l)
    
    return num_cols, str_cols


def show(s: pd.DataFrame) -> None: 
    """
    Quick EDA visualization for dataset inspection.
    Displays class distribution, numerical features, and categorical features.
    """
    num_col, str_col = get_cols(s)

    # Label distribution
    # Special handling for df_usage (continuous to binary conversion for visualization)
    if "Usange" not in s._name and s[s._label].nunique() > 2:
        viz_label = (s[s._label] >= s[s._label].median()).astype(int)
        tmp_col = viz_label.value_counts()
    else:
        tmp_col = s[s._label].value_counts()
    
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    fig.canvas.manager.set_window_title(s._name)

    bars = ax.bar(tmp_col.index, tmp_col.values)
    ax.bar_label(bars)
    ax.set_title(f"{s._label} Distribution")
    plt.show()

    # Numerical distribution
    ax = s[num_col].hist(bins=15, figsize=FIG_SIZE)
    fig = ax[0][0].figure
    fig.canvas.manager.set_window_title(s._name)

    plt.suptitle("Numerical Features Distribution")
    plt.tight_layout()
    plt.show()

    # Non-numeric distribution
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
        ax.set_title(f"{l} Distribution")
        plt.show()

for df in dfs:
    show(df)



# =========================================================
# Step 3 — Data Quality Assessment
# - evaluate completeness and missing values
# - analyze label balance and feature diversity
# =========================================================

def check_balance(label_cnt: pd.Series, name: str) -> None:
    """
    Show label distribution and check class balance.
    
    Parameters:
        label_cnt (pd.Series): value counts of label
        name (str): dataset name
    """
    # Show label distribution
    total_cnt = label_cnt.sum()

    print(f"Label Distribution ({name}):")
    for val, count in label_cnt.items():
        pct = count / total_cnt * 100
        print(f"    - {val}: {count:,} ({pct:.1f}%)")
    
    # Class balance check
    if len(label_cnt) == 2:
        ratio = label_cnt.iloc[0] / label_cnt.iloc[1]
        imbalance = max(ratio, 1 / ratio)
        print(f"Class imbalance ratio: {imbalance:.2f}:1")
        if imbalance > 2:
            print(f"Significant class imbalance in {name}!")


def assess_quality(s: pd.DataFrame) -> None:
    """
    Detailed data quality report.
    Evaluates missing patterns, target balance, and feature diversity.
    """
    print(f"\n\n\n========== {s._name} Quality Assessment ==========")
    row_cnt = s.shape[0]
    
    # Missing value patterns
    missing_cnt = s.isnull().sum()
    total_missing = missing_cnt.sum()
    print(f"\nMissing Values: {total_missing:,}")
    
    if total_missing > 0:
        missing_col = missing_cnt[missing_cnt > 0].sort_values(ascending=False)
        for col, count in missing_col.head(10).items():
            print(f"    - {col}: {count:,} ({count / row_cnt * 100:.1f}%)")
    
    # Handle labels
    if "Usange" not in s._name and s[s._label].nunique() > 2:
        pass
    else:
        label_cnt = s[s._label].value_counts()
        # Class balance check
        check_balance(label_cnt, s._name)

    # Feature diversity
    num_col, str_col = get_cols(s)
    print(f"Feature Diversity:")
    print(f"    - Numeric features: {len(num_col)}")
    print(f"    - Categorical features: {len(str_col)}")
    

for df in dfs:
    assess_quality(df)



# =========================================================
# Step 4 — Label Handling
# - convert labels to suitable formats for modeling
# - handle continuous and categorical labels appropriately
# - binarize all labels BEFORE train/test split for proper stratification
# =========================================================

# df_completion: string "Completed" → binary 0/1
df_completion[df_completion._label] = (df_completion[df_completion._label].str.lower() == "completed").astype(int)
save_data(df_completion, "Binary_Label")

# df_consumption: drop "in progress", then string "Completed" → binary 0/1
df_consumption.drop(df_consumption[df_consumption[df_consumption._label].str.lower() == "in progress"].index, inplace=True)
df_consumption[df_consumption._label] = (df_consumption[df_consumption._label].str.lower() == "completed").astype(int)
save_data(df_consumption, "Binary_Label")

# df_usage: continuous percentage → binary using median threshold
threshold_usage = df_usage[df_usage._label].median()
df_usage[df_usage._label] = (df_usage[df_usage._label] >= threshold_usage).astype(int)
save_data(df_usage, "Binary_Label")

# Verify distributions after binarization
print("\n\n\n")
for df in dfs:
    check_balance(df[df._label].value_counts(), df._name)
    print()



# =========================================================
# Step 5 — Structural Cleaning & Type Inference
# - remove duplicates
# - infer numeric / datetime columns
# - drop ID-like columns
# - create train/test split on primary dataset
# =========================================================

def set_col_type(s: pd.DataFrame) -> list[str]:
    """
    Infer column types and detect ID-like columns.
    
    Transforms column types by:
    - Converting numeric strs to numeric types
    - Extracting datetime features (year, month, day, weekday)
    - Identifying and removing ID-like columns with high uniqueness
    
    Returns:
        id_cols (list): ID-like columns
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

# Apply structural cleaning to all datasets
for df in dfs:
    id_cols = set_col_type(df)
    df.drop(columns=id_cols, inplace=True)
    df.dropna(subset=[df._label], inplace=True)

    print(f"\n\n\n========== {df._name} Structural Cleaning ==========")
    print(f"ID-like columns removed: \n{id_cols}")
    save_data(df, "ID_dropped")


# Create train/test split on primary dataset
# Prevent data leakage
train_completion, test_completion = train_test_split(
    df_completion,
    test_size=0.2,
    random_state=42,
    stratify=df_completion[df_completion._label]
)

init_df(train_completion, "Train_Completion", DATA_DIR / "Processed" / "Completion" / "Train", df_completion._label)
init_df(test_completion, "Test_Completion", DATA_DIR / "Processed" / "Completion" / "Test", df_completion._label)

save_data(train_completion)
save_data(test_completion)

# Create train/test split on supplementary datasets
train_consumption, test_consumption = train_test_split(
    df_consumption,
    test_size=0.2,
    random_state=42,
    stratify=df_consumption[df_consumption._label]
)

init_df(train_consumption, "Train_Consumption", DATA_DIR / "Processed" / "Consumption" / "Train", df_consumption._label)
init_df(test_consumption, "Test_Consumption", DATA_DIR / "Processed" / "Consumption" / "Test", df_consumption._label)

save_data(train_consumption)
save_data(test_consumption)

train_usage, test_usage = train_test_split(
    df_usage,
    test_size=0.2,
    random_state=42,
    stratify=df_usage[df_usage._label]
)

init_df(train_usage, "Train_Usage", DATA_DIR / "Processed" / "Usage" / "Train", df_usage._label)
init_df(test_usage, "Test_Usage", DATA_DIR / "Processed" / "Usage" / "Test", df_usage._label)

save_data(train_usage)
save_data(test_usage)

dfs = [train_completion, train_consumption, train_usage]



# =========================================================
# Step 6 — Missing Value Handling & Alignment  
# - fill NA, remove empty rows & columns
# - align test schema with train
# =========================================================

def deal_na(s: pd.DataFrame, rule: dict[str, float] = None) -> dict[str, float]:
    """
    Missing value handling: fill numeric/categorical and remove empty rows/columns.
    
    Strategy:
    - Numeric: median imputation
    - Categorical: explicit NA marker
    - Empty rows/columns: removal
    
    Parameters:
        s (DataFrame)
        rule (dict): pre-computed fill values from training set
    """
    print(f"\n\n\n========== {s._name} NA Handling ==========")

    num_col, str_col = get_cols(s)
    if not rule:
        rule = {}
    
    # Numeric: median imputation
    print("Numeric NA counts: ")
    for l in num_col:
        na_cnt = s[l].isna().sum()
        if na_cnt > 0:
            print(f"   - {l}: {na_cnt}")
        
        if l not in rule:
            rule[l] = s[l].median()
        s[l] = s[l].fillna(rule[l])
    
    # Categorical: mark missing explicitly
    print("\nCategorical NA counts: ")
    for l in str_col:
        na_cnt = s[l].isna().sum()
        if na_cnt > 0:
            print(f"    - {l}: {na_cnt}")
        s[l] = s[l].fillna(pd.NA)
    
    # Remove completely empty rows/columns
    s.dropna(axis=1, how="all", inplace=True)
    s.dropna(axis=0, how="all", inplace=True)
    
    return rule

# Compute NA rules from training sets
na_rule_completion = deal_na(train_completion)
na_rule_consumption = deal_na(train_consumption)
na_rule_usage = deal_na(train_usage)

# Apply same rules to corresponding test sets
deal_na(test_completion, na_rule_completion)
deal_na(test_consumption, na_rule_consumption)
deal_na(test_usage, na_rule_usage)

# Align test columns with train
test_completion.drop(columns=test_completion.columns.difference(train_completion.columns), inplace=True)
test_consumption.drop(columns=test_consumption.columns.difference(train_consumption.columns), inplace=True)
test_usage.drop(columns=test_usage.columns.difference(train_usage.columns), inplace=True)

save_data(train_completion, "No_NA")
save_data(test_completion, "No_NA")
save_data(train_consumption, "No_NA")
save_data(test_consumption, "No_NA")
save_data(train_usage, "No_NA")
save_data(test_usage, "No_NA")



# =========================================================
# Step 7 — Feature Enrichment
# - extract aggregated statistics from primary dataset only
# - create behavioral features for high-granularity data
# =========================================================

def enrich(s: pd.DataFrame, stats: dict = None) -> dict:
    """
    Enrich dataset with behavioral features and interaction terms.
    Flexible column matching for different dataset schemas.
    
    Parameters:
        s (DataFrame): dataset to enrich
        stats (dict): pre-computed statistics from train_df
    """
    print(f"\n\n\n========== {s._name} Feature Enrichment ==========")
    
    if stats is None:
        stats = {}
        
        # df_completion engagement metrics
        if all(l in s.columns for l in ["Login_Frequency", "Discussion_Participation", "Assignments_Submitted"]):
            stats["max_login"] = s["Login_Frequency"].max()
            stats["max_discuss"] = s["Discussion_Participation"].max()
            stats["max_assign"] = s["Assignments_Submitted"].max()
        
        # Supplementary datasets stats
        if "Hours_Spent_Per_Week" in s.columns:
            stats["avg_hours"] = s["Hours_Spent_Per_Week"].mean()
        if "Course_Duration_Weeks" in s.columns:
            stats["avg_duration"] = s["Course_Duration_Weeks"].mean()
        if "Completion_Percentage" in s.columns:
            stats["avg_completion"] = s["Completion_Percentage"].mean()
        if "Satisfaction_Score" in s.columns:
            stats["avg_satisfaction"] = s["Satisfaction_Score"].mean()
    
    # ===== df_completion Features =====
    # Engagement score
    if all(l in s.columns for l in ["Login_Frequency", "Discussion_Participation", "Assignments_Submitted"]):
        s["engagement_score"] = (
            s["Login_Frequency"] / max(stats.get("max_login", 1), 1) * 0.3 +
            s["Discussion_Participation"] / max(stats.get("max_discuss", 1), 1) * 0.3 +
            s["Assignments_Submitted"] / max(stats.get("max_assign", 1), 1) * 0.4
        ).round(3)
        print("engagement_score")
    
    # Learning efficiency
    if all(l in s.columns for l in ["Quiz_Score_Avg", "Progress_Percentage"]):
        s["efficiency"] = (
            s["Quiz_Score_Avg"] / 100 * s["Progress_Percentage"] / 100
        ).fillna(0).round(3)
        print("efficiency")
    
    # Instructor-Course Level interaction
    if all(l in s.columns for l in ["Instructor_Rating", "Course_Level"]):
        level_multiplier = s["Course_Level"].map({
            "Beginner": 1,
            "Intermediate": 2,
            "Advanced": 3
        }).fillna(1)
        s["instructor_level_interaction"] = (s["Instructor_Rating"] * level_multiplier).round(3)
        print("instructor_level_interaction")
    
    # Session-based engagement proxy
    if all(l in s.columns for l in ["Average_Session_Duration_Min", "Login_Frequency"]):
        s["total_session_time"] = (
            s["Average_Session_Duration_Min"] * s["Login_Frequency"]
        ).round(2)
        print("total_session_time")
    
    # Video progress vs Quiz performance
    if all(l in s.columns for l in ["Video_Completion_Rate", "Quiz_Score_Avg"]):
        s["video_quiz_alignment"] = (
            (s["Video_Completion_Rate"] / 100) * (s["Quiz_Score_Avg"] / 100)
        ).round(3)
        print("video_quiz_alignment")
    
    # ===== df_consumption Features =====
    # Time Invested
    if all(l in s.columns for l in ["Hours_Spent_Per_Week", "Course_Duration_Weeks"]):
        s["time_invested"] = (
            s["Hours_Spent_Per_Week"] * s["Course_Duration_Weeks"]
        ).round(2)
        print("time_invested")
    
    # Completion-Satisfaction Alignment
    if all(l in s.columns for l in ["Completion_Percentage", "Satisfaction_Score"]):
        s["completion_satisfaction_alignment"] = (
            (s["Completion_Percentage"] / 100) * (s["Satisfaction_Score"] / 5)
        ).round(3)
        print("completion_satisfaction_alignment")
    
    # Experience-Workload Fit
    if all(l in s.columns for l in ["Experience_Level", "Hours_Spent_Per_Week"]):
        exp_multiplier = s["Experience_Level"].map({
            "Fresher": 1,
            "Student": 2,
            "Working Professional": 3
        }).fillna(1)
        avg_hours = stats.get("avg_hours", s["Hours_Spent_Per_Week"].mean())
        s["workload_alignment"] = (
            (s["Hours_Spent_Per_Week"] / max(avg_hours, 1)) * exp_multiplier
        ).round(3)
        print("workload_alignment")
    
    # Engagement Intensity
    if all(l in s.columns for l in ["Completion_Percentage", "Hours_Spent_Per_Week"]):
        avg_completion = stats.get("avg_completion", s["Completion_Percentage"].mean())
        s["engagement_intensity"] = (
            (s["Completion_Percentage"] / max(avg_completion, 1)) * 
            (s["Hours_Spent_Per_Week"] / max(stats.get("avg_hours", 1), 1))
        ).round(3)
        print("engagement_intensity")
    
    # ===== df_usage Features =====
    # Price-Duration Efficiency
    if all(l in s.columns for l in ["Price ($)", "Duration (hours)"]):
        s["price_per_hour"] = (
            s["Price ($)"] / (s["Duration (hours)"] + 1)
        ).round(3)
        print("price_per_hour")
    
    # Course Value Proposition (Rating weighted by enrollment)
    if all(l in s.columns for l in ["Enrolled_Students", "Rating (out of 5)", "Completion_Rate (%)"]):
        s["value_score"] = (
            (s["Enrolled_Students"] / max(stats.get("max_enrolled", s["Enrolled_Students"].max()), 1)) * 0.3 +
            (s["Rating (out of 5)"] / 5) * 0.4 +
            (s["Completion_Rate (%)"] / 100) * 0.3
        ).round(3)
        print("value_score")
    
    return stats

# Compute enrichment statistics from training sets
train_stats_completion = enrich(train_completion)
train_stats_consumption = enrich(train_consumption)
train_stats_usage = enrich(train_usage)

# Apply same rules to corresponding test datasets
enrich(test_completion, train_stats_completion)
enrich(test_consumption, train_stats_consumption)
enrich(test_usage, train_stats_usage)


save_data(train_completion, "Enriched")
save_data(test_completion, "Enriched")
save_data(train_consumption, "Enriched")
save_data(test_consumption, "Enriched")
save_data(train_usage, "Enriched")
save_data(test_usage, "Enriched")



# =========================================================
# Step 8 — Feature Transformation (Scaling & Encoding)
# - classify feature types and distributions
# - define transformation strategy for each feature type
# - apply scaling and encoding to stabilize feature distributions
# =========================================================

def which_distribution(l: pd.Series) -> str:
    """
    Classify numeric columns by value range and empirical distribution.
    Identifies special ranges (ratio, percentage, etc.) before statistical analysis.
    """
    min_val, max_val = l.min(), l.max()
    
    # Fixed-range features
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
    
    # Statistical properties for general numeric features
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
    Used to select appropriate encoding strategy.
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
    Returns outlier thresholds for robust scaling.
    """
    Q1 = s.quantile(0.25)
    Q3 = s.quantile(0.75)
    IQR = Q3 - Q1
        
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    return lower, upper

def scale(s: pd.DataFrame, map: dict[str, tuple] = None) -> dict[str, tuple]:
    """
    Apply numeric scaling rules based on feature distributions.
    
    Parameters:
        s (DataFrame)
        map (dict): transformation rules from train_df

    Returns:
        map (dict): transformation rules for this dataset
    """
    num_col, _ = get_cols(s)
    
    if map:
        # Apply previously computed rules to test/supplementary datasets
        for l in num_col:
            col = s[l]
            rule = map[l]
            
            # simple clipping
            if len(rule) == 2:
                lower, upper = rule
                s[l] = col.clip(lower, upper)
            
            # log transformation + clipping
            else:
                lower, upper, _ = rule
                col_log = np.log1p(col.clip(lower=0))
                s[l] = np.expm1(col_log.clip(0, upper))
        
        return map

    # Compute rules from training data
    map = {}
    
    print(f"\n\n\n========== {s._name} Feature Scaling ==========")
    print("Distribution: ")

    for l in num_col:
        col = s[l]
        distr = which_distribution(col)
        print(f"    - {l}: {distr}")
    
        # Fixed range features - clip to valid range
        if distr == "ratio":
            map[l] = (0, 1)
            s[l] = col.clip(0, 1)        
            continue
        
        if distr == "rating":
            map[l] = (0, 5)
            s[l] = col.clip(0, 5)
            continue
        
        if distr == "percentage":
            map[l] = (0, 100)
            s[l] = col.clip(0, 100)
            continue
        
        if distr == "discrete":
            map[l] = (0, None)
            s[l] = col.clip(lower=0)
            continue
        
        # Right-skewed features: log transformation for stabilization
        if distr == "right_skewed":
            col_log = np.log1p(col.clip(lower=0))
            _, upper = get_range(col_log)
            map[l] = (0, upper, "log")
            s[l] = np.expm1(col_log.clip(0, upper))
            continue
    
        # General numeric features: IQR-based clipping
        lower, upper = get_range(col)  
        map[l] = (lower, upper)
        s[l] = col.clip(lower, upper)
    
    return map

# Compute scaling rules from training sets
scale_map_completion = scale(train_completion)
scale_map_consumption = scale(train_consumption)
scale_map_usage = scale(train_usage)

# Apply same rules to corresponding test datasets
scale(test_completion, scale_map_completion)
scale(test_consumption, scale_map_consumption)
scale(test_usage, scale_map_usage)


def encoding(s: pd.DataFrame, map: dict = None) -> tuple:
    """
    Encode categorical features based on cardinality.
    
    Strategy:
    - Low/Medium cardinality: ordinal encoding
    - High cardinality: frequency encoding
    
    Parameters:
        s (DataFrame)
        map (dict): transformation rules from train_df
    """
    _, str_col = get_cols(s)
    str_col = [l for l in str_col if l != s._label]
    
    if map:
        # Apply previous encoding rules
        for l in str_col:
            mapping = map[l]
            s[l] = s[l].fillna("Missing").map(mapping)
        return map
    
    # Compute encoding rules from training data
    map = {}
    
    print(f"\n\n\n========== {s._name} Feature Encoding ==========")
    print("Cardinality: ")

    for l in str_col:
        col = s[l].fillna("Missing").astype(str)
        card = which_card(col)
        print(f"    - {l}: {card}")

        # Ordinal encoding for low- and medium-cardinality features
        if card in ["low_card", "medium_card"]:
            unique_cats = col.unique()
            map[l] = {cat: i for i, cat in enumerate(sorted(unique_cats))}
            s[l] = col.map(map[l])
    
        # Frequency encoding for high-cardinality features
        else:
            freq = col.value_counts(normalize=True)
            map[l] = freq
            s[l] = col.map(freq)
    
    return map

# Compute encoding rules from training sets
encoding_map_completion = encoding(train_completion)
encoding_map_consumption = encoding(train_consumption)
encoding_map_usage = encoding(train_usage)

# Apply same rules to corresponding test datasets
encoding(test_completion, encoding_map_completion)
encoding(test_consumption, encoding_map_consumption)
encoding(test_usage, encoding_map_usage)

save_data(train_completion, "Scaled")
save_data(test_completion, "Scaled")
save_data(train_consumption, "Scaled")
save_data(test_consumption, "Scaled")
save_data(train_usage, "Scaled")
save_data(test_usage, "Scaled")



# =========================================================
# Step 9 — Feature Engineering (Filtering & Selection)
# - remove constant columns (zero variance)
# - remove redundant/correlated columns
# - select high-value features based on class separation strength
# =========================================================

def drop_constant(s: pd.DataFrame) -> pd.DataFrame:
    """
    Remove constant columns with no variance.
    These columns provide no discriminative information for modeling.
    """
    print(f"\n\n\n========== {s._name} Constant Columns ==========")

    constant_col = [l for l in s.columns if s[l].nunique() <= 1]
    if constant_col:
        print(f"Columns removed: \n{constant_col}")
        s.drop(columns=constant_col, inplace=True)
    
    return s

# Remove constant columns from all datasets
drop_constant(train_completion)
drop_constant(test_completion)
drop_constant(train_consumption)
drop_constant(test_consumption)
drop_constant(train_usage)
drop_constant(test_usage)


def drop_redundant(s: pd.DataFrame) -> pd.DataFrame:
    """
    Remove redundant features that duplicate information.
    
    Identifies two types of redundancy:
    - Linear: perfectly correlated features (r ≈ ±1)
    - Functional: deterministic relationships between features
    """
    print(f"\n\n\n========== {s._name} Redundant Columns ==========")

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

            # Linear redundancy: perfect correlation
            if np.isclose(abs(corr_matrix.iloc[i, j]), 1.0):
                redundant_col.add(c2)
                continue

            # Functional redundancy: deterministic mapping
            if s.groupby(c1)[c2].nunique().max() == 1 and s.groupby(c2)[c1].nunique().max() == 1:
                redundant_col.add(c2)
                continue

    redundant_col = list(redundant_col)
    if redundant_col:
        print(f"Columns removed: \n{redundant_col}")
        s.drop(columns=redundant_col, inplace=True)
    
    return s

# Remove redundant columns from all datasets
drop_redundant(train_completion)
drop_redundant(test_completion)
drop_redundant(train_consumption)
drop_redundant(test_consumption)

# Ensure test schema matches train for each dataset
test_completion.drop(columns=test_completion.columns.difference(train_completion.columns), inplace=True)
test_consumption.drop(columns=test_consumption.columns.difference(train_consumption.columns), inplace=True)
test_usage.drop(columns=test_usage.columns.difference(train_usage.columns), inplace=True)

save_data(train_completion, "Dropped")
save_data(test_completion, "Dropped")
save_data(train_consumption, "Dropped")
save_data(test_consumption, "Dropped")
save_data(train_usage, "Dropped")
save_data(test_usage, "Dropped")



# =========================================================
# Step 10 — Feature-Label Separation & Standardization
# - separate features and labels
# - normalize feature scale for model stability
# =========================================================

# ===== DATASET 1: COMPLETION =====
# Separate features from labels
X_train_completion = train_completion.drop(columns=[train_completion._label])
y_train_completion = train_completion[train_completion._label]

init_df(X_train_completion, "X_train_completion", train_completion._dir / "X_train")
init_df(y_train_completion, "y_train_completion", train_completion._dir / "y_train")

X_test_completion = test_completion.drop(columns=[test_completion._label])
y_test_completion = test_completion[test_completion._label]

init_df(X_test_completion, "X_test_completion", test_completion._dir / "X_test")
init_df(y_test_completion, "y_test_completion", test_completion._dir / "y_test")

save_data(X_train_completion)
save_data(y_train_completion)
save_data(X_test_completion)
save_data(y_test_completion)


# ===== DATASET 2: CONSUMPTION =====
# Separate features from labels
X_train_consumption = train_consumption.drop(columns=[train_consumption._label])
y_train_consumption = train_consumption[train_consumption._label]

init_df(X_train_consumption, "X_train_consumption", train_consumption._dir / "X_train")
init_df(y_train_consumption, "y_train_consumption", train_consumption._dir / "y_train")

X_test_consumption = test_consumption.drop(columns=[test_consumption._label])
y_test_consumption = test_consumption[test_consumption._label]

init_df(X_test_consumption, "X_test_consumption", test_consumption._dir / "X_test")
init_df(y_test_consumption, "y_test_consumption", test_consumption._dir / "y_test")

save_data(X_train_consumption)
save_data(y_train_consumption)
save_data(X_test_consumption)
save_data(y_test_consumption)


# ===== DATASET 3: USAGE =====
# Separate features from labels
X_train_usage = train_usage.drop(columns=[train_usage._label])
y_train_usage = train_usage[train_usage._label]

init_df(X_train_usage, "X_train_usage", train_usage._dir / "X_train")
init_df(y_train_usage, "y_train_usage", train_usage._dir / "y_train")

X_test_usage = test_usage.drop(columns=[test_usage._label])
y_test_usage = test_usage[test_usage._label]

init_df(X_test_usage, "X_test_usage", test_usage._dir / "X_test")
init_df(y_test_usage, "y_test_usage", test_usage._dir / "y_test")

save_data(X_train_usage)
save_data(y_train_usage)
save_data(X_test_usage)
save_data(y_test_usage)


# Feature standardization for all datasets
# ===== COMPLETION =====
num_train_completion, _ = get_cols(X_train_completion)
num_test_completion, _ = get_cols(X_test_completion)

completion_scaler = StandardScaler()
X_train_completion[num_train_completion] = completion_scaler.fit_transform(X_train_completion[num_train_completion])
X_test_completion[num_test_completion] = completion_scaler.transform(X_test_completion[num_test_completion])

save_data(X_train_completion, "Transformed")
save_data(X_test_completion, "Transformed")

# ===== CONSUMPTION =====
num_train_consumption, _ = get_cols(X_train_consumption)
num_test_consumption, _ = get_cols(X_test_consumption)

consumption_scaler = StandardScaler()
X_train_consumption[num_train_consumption] = consumption_scaler.fit_transform(X_train_consumption[num_train_consumption])
X_test_consumption[num_test_consumption] = consumption_scaler.transform(X_test_consumption[num_test_consumption])

save_data(X_train_consumption, "Transformed")
save_data(X_test_consumption, "Transformed")

# ===== USAGE =====
num_train_usage, _ = get_cols(X_train_usage)
num_test_usage, _ = get_cols(X_test_usage)

usage_scaler = StandardScaler()
X_train_usage[num_train_usage] = usage_scaler.fit_transform(X_train_usage[num_train_usage])
X_test_usage[num_test_usage] = usage_scaler.transform(X_test_usage[num_test_usage])

save_data(X_train_usage, "Transformed")
save_data(X_test_usage, "Transformed")

# =========================================================
# Step 11 — Feature Selection (EDA-driven)
# - select features based on class separation strength (Std_Delta)
# - prepare datasets for model training
# =========================================================

def feature_rank(s: pd.DataFrame) -> list[str]:
    """
    Rank numeric features by standardized mean difference.
    
    Interpretation:
    - Std_Delta measures how well each feature separates completion vs dropout groups
    - Higher values indicate stronger predictive potential
    - Threshold balances model complexity vs information retention
    """
    print(f"\n\n\n========== {s._name} Feature Selection ==========")

    num_col, _ = get_cols(s)
    num_col = [l for l in num_col if l != s._label]
    threshold = 0.02

    # Compute class-wise mean for each feature
    group_mean = s.groupby(s._label)[num_col].mean()
    print(f"Group Mean (by class):\n{group_mean}")

    std = s[num_col].std()

    # Standardized mean difference: scale-invariant measure of class separation
    delta = (group_mean.loc[0] - group_mean.loc[1]).abs() / std

    num_df = pd.DataFrame({
        "Group_0_Mean": group_mean.loc[0],
        "Group_1_Mean": group_mean.loc[1],
        "Std": std,
        "Std_Delta": delta
    }).sort_values(by="Std_Delta", ascending=False)
    
    print(f"\nFeature Contribution: \n{num_df["Std_Delta"].T}")

    # Select features with significant separation
    features = num_df[num_df["Std_Delta"] > threshold].index.to_list()

    # Validate selected features
    features = [f for f in features if f in s.columns and f != s._label]

    print(f"\n{s._name} Selected {len(features)} features")
    print(f"Selection threshold: Std_Delta > {threshold}")
    print(features)

    return features

# Compute feature rankings for all datasets
feature_completion = feature_rank(train_completion)
feature_consumption = feature_rank(train_consumption)
feature_usage = feature_rank(train_usage)

# Visualization
fig, ax = plt.subplots(figsize=FIG_SIZE)
info = pd.DataFrame({
    "Dataset": ["Completion", "Consumption", "Usage"],
    "Total Features": [len(train_completion.columns) - 1, len(train_consumption.columns) - 1, len(train_usage.columns) - 1],
    "Selected Features": [len(feature_completion), len(feature_consumption), len(feature_usage)]
})
info["Retention Rate"] = (info["Selected Features"] / info["Total Features"] * 100).round(1)


print("\n\n\n========== Feature Selection Summary ==========")
print(info.to_string(index=False))

ax.axis("off")
table = ax.table(cellText=info.values, colLabels=info.columns, 
                cellLoc="center", loc="center", colWidths=[0.25, 0.25, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)
ax.set_title("Feature Selection Across Datasets", fontsize=14, fontweight="bold", pad=20)
plt.tight_layout()
plt.show()



# =========================================================
# Step 12 — Model Visualization & Evaluation Utilities
# =========================================================

def plot_confusion_matrix(cm: np.ndarray, name: str, cmap: str = "Blues") -> None:
    """
    Plot confusion matrix heatmap for model evaluation.
    Visualizes classification performance across classes.
    
    Parameters:
        cm (np.ndarray): confusion matrix from sklearn.metrics
        name (str): model name for window title
        cmap (str): colormap name
    """
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap=cmap,
        xticklabels=["Not Completed", "Completed"],
        yticklabels=["Not Completed", "Completed"],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{name} Confusion Matrix")
    fig.canvas.manager.set_window_title(name)
    plt.show()


def plot_feature_importance(features: list[str], scores: np.ndarray, name: str, cmap_color: str = "steelblue", K: int = 10) -> None:
    """
    Plot feature importance/coefficients for model interpretation.
    Displays top K features and their contribution scores.
    
    Parameters:
        features (list): feature names
        scores (np.ndarray): importance scores or coefficients
        name (str): model name for visualization
        cmap_color (str): bar color
        K (int) : top K features
    """
    # Create DataFrame and sort
    feat_df = pd.DataFrame({
        "Feature": features,
        "Score": scores
    }).sort_values(by="Score", ascending=False)
    
    # Plot top K
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.barh(feat_df.head(K)["Feature"], feat_df.head(K)["Score"], color=cmap_color)
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top {K} Features - {name}")
    ax.invert_yaxis()  # Highest at top
    fig.canvas.manager.set_window_title(f"{name} Feature Importance")
    plt.tight_layout()
    plt.show()
    
    return feat_df


def print_evaluation(name: str, accuracy: float, report: str) -> None:
    """
    Print standardized model evaluation metrics summary.
    """
    print("\nPerformance: ")
    print(f"    Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)



# =========================================================
# Step 13 — Model Preparation (Feature Selection & CV Setup)
# - select high-value features from feature ranking
# - configure cross-validation for stable evaluation
# =========================================================

# ===== DATASET 1: COMPLETION (primary) =====
# Feature selection
X_train_completion_feature = X_train_completion[feature_completion]
X_test_completion_feature = X_test_completion[feature_completion]

init_df(X_train_completion_feature, "X_train_completion_feature", X_train_completion._dir)
init_df(X_test_completion_feature, "X_test_completion_feature", X_test_completion._dir)

save_data(X_train_completion_feature, "Features")
save_data(X_test_completion_feature, "Features")


# ===== DATASET 2: CONSUMPTION =====
# Feature selection
X_train_consumption_feature = X_train_consumption[feature_consumption]
X_test_consumption_feature = X_test_consumption[feature_consumption]

init_df(X_train_consumption_feature, "X_train_consumption_feature", X_train_consumption._dir)
init_df(X_test_consumption_feature, "X_test_consumption_feature", X_test_consumption._dir)

save_data(X_train_consumption_feature, "Features")
save_data(X_test_consumption_feature, "Features")


# ===== DATASET 3: USAGE =====
# Feature selection
X_train_usage_feature = X_train_usage[feature_usage]
X_test_usage_feature = X_test_usage[feature_usage]

init_df(X_train_usage_feature, "X_train_usage_feature", X_train_usage._dir)
init_df(X_test_usage_feature, "X_test_usage_feature", X_test_usage._dir)

save_data(X_train_usage_feature, "Features")
save_data(X_test_usage_feature, "Features")

print(f"\n========== Cross-Validation Datasets ==========")
print(f"Completion: {X_train_completion_feature.shape[0]} train, {X_test_completion_feature.shape[0]} test, {len(feature_completion)} features")
print(f"Consumption: {X_train_consumption_feature.shape[0]} train, {X_test_consumption_feature.shape[0]} test, {len(feature_consumption)} features")
print(f"Usage: {X_train_usage_feature.shape[0]} train, {X_test_usage_feature.shape[0]} test, {len(feature_usage)} features")

# Cross-validation configuration
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



# =========================================================
# Step 14 — Multi-Dataset Model Training Framework
# - unified pipeline for all datasets
# - abstract model training into reusable function
# =========================================================

def train_single(model, X_train, y_train, X_test, y_test, model_name: str, 
                       dataset_name: str, cv = None) -> dict:
    """
    Train a single model and return comprehensive metrics.
    
    Parameters:
        model: sklearn model instance
        X_train, y_train, X_test, y_test: data splits
        model_name (str)
        dataset_name (str)
        cv: cross-validation strategy (optional)
    
    Returns:
        dict: with cv_scores, test_accuracy, predictions, cm, report, importance
    """
    result = {"model": model, "name": model_name}
    
    # Cross-validation
    if cv is not None:
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        result["cv_mean"] = cv_scores.mean()
        result["cv_std"] = cv_scores.std()
        print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Train
    model.fit(X_train, y_train)
    
    # Test predictions
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print_evaluation(f"{dataset_name} - {model_name}", acc, report)
    plot_confusion_matrix(cm, f"{dataset_name} - {model_name}", 
                         {"lr": "Blues", "rf": "Greens", "xgb": "Oranges", "voting": "RdYlGn"}.get(model_name.lower().split()[0][:2], "Blues"))
    
    # Feature importance/coefficients
    if hasattr(model, 'coef_'):  # LR
        importance_df = plot_feature_importance(X_train.columns, model.coef_[0],
                                               f"{dataset_name} - {model_name}", "steelblue")
    elif hasattr(model, 'feature_importances_'):  # RF, XGB
        importance_df = plot_feature_importance(X_train.columns, model.feature_importances_,
                                               f"{dataset_name} - {model_name}",
                                               {"rf": "forestgreen", "xgb": "coral"}.get(model_name.lower().split()[0][:2], "steelblue"))
    elif hasattr(model, 'named_estimators_'):  # Voting - skip visualization
        importance_df = pd.DataFrame()
    else:
        importance_df = pd.DataFrame()
    
    if not importance_df.empty:
        print("Top Features:")
        print(importance_df.head(10))
    
    result["test_acc"] = acc
    result["cm"] = cm
    result["report"] = report
    result["y_pred"] = y_pred
    result["importance"] = importance_df
    
    return result


def train_all(X_train, y_train, X_test, y_test, name: str, cv = None) -> dict:
    """
    Train all 4 models (LR, RF, XGB, Voting) on a dataset.
    
    Returns:
        dict: keyed by model type with results
    """
    print(f"\n\n\nDataset: {name}")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    results = {}
    
    # ===== Logistic Regression =====
    print(f"========== Logistic Regression ==========")
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    results["lr"] = train_single(lr, X_train, y_train, X_test, y_test, 
                                       "Logistic Regression", name, cv)
    
    # ===== Random Forest =====
    print(f"\n========== Random Forest ==========")
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_split=4, 
                               min_samples_leaf=2, max_features="sqrt", 
                               class_weight="balanced", random_state=42, n_jobs=-1)
    results["rf"] = train_single(rf, X_train, y_train, X_test, y_test,
                                       "Random Forest", name, cv)
    
    # ===== XGBoost =====
    print(f"\n========== XGBoost ==========")
    xgb = XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=5, 
                        subsample=0.9, colsample_bytree=0.9, scale_pos_weight=1,
                        random_state=42, eval_metric="logloss", n_jobs=-1)
    results["xgb"] = train_single(xgb, X_train, y_train, X_test, y_test,
                                        "XGBoost", name, cv)
    
    # ===== Voting Classifier =====
    print(f"\n========== Voting Classifier ==========")
    voting = VotingClassifier(
        estimators=[("lr", results["lr"]["model"]), 
                   ("rf", results["rf"]["model"]), 
                   ("xgb", results["xgb"]["model"])],
        voting="soft"
    )
    results["voting"] = train_single(voting, X_train, y_train, X_test, y_test,
                                          "Voting Classifier", name, cv)
    
    return results

results_completion = train_all(X_train_completion_feature, y_train_completion, X_test_completion_feature, y_test_completion, 
                                      "Completion", cv=skf)
results_consumption = train_all(X_train_consumption_feature, y_train_consumption, X_test_consumption_feature, y_test_consumption,
                                       "Consumption", cv=skf)
results_usage = train_all(X_train_usage_feature, y_train_usage, X_test_usage_feature, y_test_usage,
                                 "Usage", cv=skf)



# =========================================================
# Step 15 — Cross-Dataset Model Comparison
# - compile results from all three datasets
# - comparative analysis and best practices identification
# =========================================================

print(f"\n\n{'═'*80}")
print(f"CROSS-DATASET MODEL COMPARISON")
print(f"{'═'*80}\n")

# Build comparison dataframe
comparison_data = []
for name, results in [("Completion", results_completion), 
                              ("Consumption", results_consumption), 
                              ("Usage", results_usage)]:
    for model_key, result in results.items():
        row = {
            "Dataset": name,
            "Model": result["name"],
            "Test_Accuracy": result["test_acc"],
            "CV_Mean": result.get("cv_mean", "N/A"),
            "CV_Std": result.get("cv_std", "N/A")
        }
        comparison_data.append(row)

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# Best model per dataset
print(f"\n\n{'─'*80}\nBEST MODEL PER DATASET:\n{'─'*80}")
for dataset_name in ["Completion", "Consumption", "Usage"]:
    dataset_results = comparison_df[comparison_df["Dataset"] == dataset_name]
    best_idx = dataset_results["Test_Accuracy"].idxmax()
    best_row = dataset_results.loc[best_idx]
    print(f"  {dataset_name:12s} → {best_row['Model']:18s} (Acc: {best_row['Test_Accuracy']:.4f})")

# ROC-AUC analysis for all models across datasets
print(f"\n\n{'─'*80}\nROC-AUC SCORES:\n{'─'*80}\n")

roc_data = []
for dataset_name, X_test, y_test, results in [
    ("Completion", X_test_completion_feature, y_test_completion, results_completion),
    ("Consumption", X_test_consumption_feature, y_test_consumption, results_consumption),
    ("Usage", X_test_usage_feature, y_test_usage, results_usage)
]:
    for model_key, result in results.items():
        if hasattr(result["model"], 'predict_proba'):
            roc_auc = roc_auc_score(y_test, result["model"].predict_proba(X_test)[:, 1])
            roc_data.append({
                "Dataset": dataset_name,
                "Model": result["name"],
                "ROC-AUC": roc_auc
            })

roc_df = pd.DataFrame(roc_data)
for dataset_name in ["Completion", "Consumption", "Usage"]:
    subset = roc_df[roc_df["Dataset"] == dataset_name]
    print(f"{dataset_name}:")
    for _, row in subset.iterrows():
        print(f"  {row['Model']:18s}: {row['ROC-AUC']:.4f}")
    print()

print(f"{'═'*80}\n")



# =========================================================
# Step 16 — Dataset Independence & Dimensionality Analysis
# - diagnose feature independence between datasets
# - explain why separate models are optimal
# =========================================================

print(f"\n\n{'═'*80}")
print(f"Dataset Independence Analysis")
print(f"{'═'*80}\n")

train_cols = set(X_train_completion_feature.columns)
cons_cols = set(X_train_consumption_feature.columns)
usage_cols = set(X_train_usage_feature.columns)

overlap_tc = len(train_cols & cons_cols)
overlap_tu = len(train_cols & usage_cols)
overlap_cu = len(cons_cols & usage_cols)

print(f"Feature Overlap:")
print(f"  Completion ∩ Consumption: {overlap_tc} / {len(train_cols)} ({overlap_tc / len(train_cols) * 100:.1f}%)")
print(f"  Completion ∩ Usage:       {overlap_tu} / {len(train_cols)} ({overlap_tu / len(train_cols) * 100:.1f}%)")
print(f"  Consumption ∩ Usage:      {overlap_cu} / {len(cons_cols)} ({overlap_cu / len(cons_cols) * 100:.1f}%)")
print(f"\nFeature Dimensions:")
print(f"  Completion: {len(train_cols)} | Consumption: {len(cons_cols)} | Usage: {len(usage_cols)}")



# =========================================================
# Step 17 — Executive Summary
# - compile key findings and performance metrics
# - provide final recommendations
# =========================================================

print(f"\n\n{'═'*80}")
print(f"Executive Summary")
print(f"{'═'*80}\n")

# Datasets summary
print("Datasets: Completion ({} feat, {} train) | Consumption ({} feat, {} train) | Usage ({} feat, {} train)".format(
    len(feature_completion), len(X_train_completion_feature),
    len(feature_consumption), len(X_train_consumption_feature),
    len(feature_usage), len(X_train_usage_feature)))

# Model performance
best_completion = comparison_df[comparison_df["Dataset"] == "Completion"]["Test_Accuracy"].max()
best_consumption = comparison_df[comparison_df["Dataset"] == "Consumption"]["Test_Accuracy"].max()
best_usage = comparison_df[comparison_df["Dataset"] == "Usage"]["Test_Accuracy"].max()
avg_acc = (best_completion + best_consumption + best_usage) / 3

print(f"\nBest Accuracy: Completion {best_completion:.4f} | Consumption {best_consumption:.4f} | Usage {best_usage:.4f} | Avg {avg_acc:.4f}")
print(f"\n{'═'*80}")