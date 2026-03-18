from datetime import date
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

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIG_SIZE = (18, 7)
RATIO = 0.9

completion_path  = DATA_DIR / "Course_Completion_Prediction.csv"
consumption_path = DATA_DIR / "online_learning_course_consumption_dataset.csv"
usage_path       = DATA_DIR / "online_courses_uses.csv"

df_completion  = pd.read_csv(completion_path)
df_consumption = pd.read_csv(consumption_path)
df_usage       = pd.read_csv(usage_path)

print("Completion :", df_completion.shape)
print("Consumption:", df_consumption.shape)
print("Usage      :", df_usage.shape)





num_cols = []
str_cols = []
id_cols = []


def get_cols(df):
    num_cols = []
    str_cols = []

    for l in df.columns:
        if ptypes.is_numeric_dtype(df[l]):
            num_cols.append(l)
            continue
        else:
            str_cols.append(l)
    
    return num_cols, str_cols
        


df_completion = df_completion.drop_duplicates()
for l in df_completion.columns.to_list():
    tmp_col = df_completion[l]

    if pd.api.types.is_numeric_dtype(tmp_col):
        continue

    tmp_numeric = pd.to_numeric(tmp_col, errors="coerce")
    numeric_ratio = tmp_numeric.notna().mean()
    if numeric_ratio > RATIO: 
        df_completion[l] = tmp_numeric
        continue

    tmp_date = pd.to_datetime(
            tmp_col, 
            errors="coerce", 
            cache=True
        )
    date_ratio = tmp_date.notna().mean()
    if date_ratio > RATIO:
        df_completion[l + "_year"] = tmp_date.dt.year
        df_completion[l + "_month"] = tmp_date.dt.month
        df_completion[l + "_day"] = tmp_date.dt.day
        df_completion[l + "_weekday"] = tmp_date.dt.weekday

        df_completion.drop(columns=[l], inplace=True)
        continue

    if df_completion[l].nunique() > len(df_completion) * RATIO:
        id_cols.append(l)
        continue


df_completion = df_completion.drop(columns=id_cols)
df_completion = df_completion.dropna(subset=["Completed"])

num_cols, str_cols = get_cols(df_completion)
for l in num_cols:
    df_completion[l] = df_completion[l].fillna(df_completion[l].median())
for l in str_cols:
    df_completion[l] = df_completion[l].fillna(pd.NA)

df_completion = df_completion.dropna(axis=1, how="all")
df_completion = df_completion.dropna(axis=0, how="all")





cmp_status = df_completion["Completed"].value_counts()
fig, ax = plt.subplots(figsize=FIG_SIZE)
bars = ax.bar(cmp_status.index, cmp_status.values)
ax.bar_label(bars)
ax.set_title("Completion Distribution")
plt.show()


num_cols, str_cols = get_cols(df_completion)
df_completion[num_cols].hist(bins=15, figsize=FIG_SIZE)
plt.suptitle("Numerical Features Distribution")
plt.tight_layout()
plt.show()


for l in str_cols:
    if df_completion[l].nunique() > 10:
        continue
    tmp_col = df_completion[l].value_counts()
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    bars = ax.bar(tmp_col.index, tmp_col.values)
    ax.bar_label(bars)
    ax.set_title(l + " Distribution")
    plt.xticks(rotation=30, ha='right')
    plt.show() 
    

'''


group_mean = df_completion.groupby("Completed")[numerics].mean()
col1, col2 = group_mean.T.columns
delta = (group_mean.T[col1] - group_mean.T[col2]).abs()
new_df = pd.DataFrame({
    f"{col1}_mean": group_mean.T[col1],
    f"{col2}_mean": group_mean.T[col2],
    "Delta": delta
}).sort_values(by="Delta", ascending=False).head(10).T
print("\n\n\nTop Features by Mean Difference:")
features = new_df.columns.to_list()
print(new_df[features].T)


result = []
for col in strs:
    if col == "Completed":
        continue
    
    dist = pd.crosstab(
        df_completion[col],
        df_completion["Completed"],
        normalize="columns"
    )
    dist = dist.fillna(0)
    
    if dist.shape[1] != 2:
        continue
    
    col1, col2 = dist.columns
    delta_str = (dist[col1] - dist[col2]).abs().sum()
    result.append((col, delta_str))

result_df = pd.DataFrame(result, columns=["Feature", "L1 Distance"]).set_index("Feature")
result_df = result_df.sort_values(by="L1 Distance", ascending=False).head(10).T
str_feature = result_df.columns.to_list()
features += str_feature
features = list(set(features))
features = [f for f in features if f in df_completion.columns]
print("\n\n\nTop Categorical Features by Distribution Difference:")
print(result_df[str_feature].T)





y = df_completion["Completed"]

x = df_completion[features]

objs = x.select_dtypes(include=["object"]).columns
x = pd.get_dummies(x, columns=objs, drop_first=True,dtype=int)
print("dtype: ", x.dtypes)

print("null: ", x.isnull().sum())
x = x.fillna(0)

print("\n\n\n")
print(x.dtypes.value_counts())
print(x.isnull().sum().sum())

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42
)

# ======== START ADDED (标准化) ========
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# ======== END ADDED ========

mod = LogisticRegression(max_iter=1000)
mod.fit(x_train, y_train)

y_pred = mod.predict(x_test)
y_prob = mod.predict_proba(x_test)[:, 1]

print("\n\nAccuracy Score: ", accuracy_score(y_test, y_pred))
print("\n\nConfusion Matrix: ", confusion_matrix(y_test, y_pred))
print("\n\nClassification Report: ", classification_report(y_test, y_pred))

risk_df = pd.DataFrame({
    "y_true": y_test,
    "y_prob": y_prob
})
high_risk = risk_df[risk_df["y_prob"] > 0.7]

# ======== START FIXED (coef对应feature名) ========
coef = pd.Series(mod.coef_[0], index=x.columns).sort_values()
# ======== END FIXED ========


print(coef.head(10))
print(coef.tail(10))

coef.head(10).plot(kind="barh")
plt.show()

coef.tail(10).plot(kind="barh")
plt.show()

x.loc[high_risk.index].mean().sort_values()'''