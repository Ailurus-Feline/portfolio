from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIG_SIZE = (18, 7)

completion_path  = DATA_DIR / "Course_Completion_Prediction.csv"
consumption_path = DATA_DIR / "online_learning_course_consumption_dataset.csv"
usage_path       = DATA_DIR / "online_courses_uses.csv"

df_completion  = pd.read_csv(completion_path)
df_consumption = pd.read_csv(consumption_path)
df_usage       = pd.read_csv(usage_path)

print("Completion :", df_completion.shape)
print("Consumption:", df_consumption.shape)
print("Usage      :", df_usage.shape)



completion_cnt = df_completion["Completed"].value_counts()
fig, ax = plt.subplots(figsize=FIG_SIZE)
bars = ax.bar(completion_cnt.index, completion_cnt.values)
ax.bar_label(bars)
ax.set_title("Completion Distribution")
plt.show()


numerics = df_completion.select_dtypes(include=["int64", "float64"]).columns
df_completion[numerics].hist(bins=15, figsize=FIG_SIZE)
plt.suptitle("Numerical Features Distribution")
plt.tight_layout()
plt.show()


strs = df_completion.select_dtypes(include=["object"]).columns
print(strs)
for l in strs:
    if df_completion[l].nunique() > 10:
        continue
    tmp_col = df_completion[l].value_counts()
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    bars = ax.bar(tmp_col.index, tmp_col.values)
    ax.bar_label(bars)
    ax.set_title(l + " Distribution")
    plt.xticks(rotation=30, ha='right')
    plt.show()


group_mean = df_completion.groupby("Completed")[numerics].mean()
col1, col2 = group_mean.T.columns
delta = (group_mean.T[col1] - group_mean.T[col2]).abs()
new_df = pd.DataFrame({
    f"{col1}_mean": group_mean.T[col1],
    f"{col2}_mean": group_mean.T[col2],
    "Delta": delta
}).sort_values(by="abs_delta", ascending=False).head(10).T
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

result_df = pd.DataFrame(result, columns=["Feature", "L1 Distance"]).set_index("feature")
result_df = result_df.sort_values(by="delta", ascending=False).head(10).T
str_feature = result_df.columns.to_list()
features += str_feature
print("\n\n\nTop Categorical Features by Distribution Difference:")
print(result_df[str_feature].T)





'''y = df_completion["Completed"]

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

mod = LogisticRegression(max_iter=1000)
mod.fit(x_train, y_train)

y_pred = mod.predict(x_test)
y_prob = mod.predict_proba(x_test)

print("\n\nAccuracy Score: ", accuracy_score(y_test, y_pred))
print("\n\nConfusion Matrix: ", confusion_matrix(y_test, y_pred))
print("\n\nClassification Report: ", classification_report(y_test, y_pred))

risk_df = pd.DataFrame({
    "y_true": y_test,
    "y_prob": y_prob
})
high_risk = risk_df[risk_df["y_prob"] > 0.7]

coef = pd.Series(mod.coef_[0], index=x.columns).sort_values()

print(coef.head(10))
print(coef.tail(10))

coef.head(10).plot(kind="barh")
plt.show()

coef.tail(10).plot(kind="barh")
plt.show()

x_test.loc[high_risk.index].mean().sort_values()'''