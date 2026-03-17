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

completion_path  = DATA_DIR / "Course_Completion_Prediction.csv"
consumption_path = DATA_DIR / "online_learning_course_consumption_dataset.csv"
usage_path       = DATA_DIR / "online_courses_uses.csv"

df_completion  = pd.read_csv(completion_path)
df_consumption = pd.read_csv(consumption_path)
df_usage       = pd.read_csv(usage_path)


print("Completion :", df_completion.shape)
print("Consumption:", df_consumption.shape)
print("Usage      :", df_usage.shape)

print("\n\n\n\n\n")
print(df_completion.head())
print(df_completion.columns)

'''counts = df_completion["Completed"].value_counts()
fig, ax = plt.subplots()
bars = ax.bar(counts.index, counts.values)
ax.bar_label(bars)
plt.show()'''

y = df_completion["Completed"]
features = [
    "Age",
    "Education_Level",
    "Employment_Status",
    "Device_Type",
    "Internet_Connection_Quality",
    "Category",
    "Course_Level",
    "Course_Duration_Days",
    "Instructor_Rating",
    "Login_Frequency",
    "Average_Session_Duration_Min",
    "Video_Completion_Rate",
    "Discussion_Participation",
    "Time_Spent_Hours",
    "Days_Since_Last_Login",
    "Notifications_Checked",
    "Peer_Interaction_Score",
    "Assignments_Submitted",
    "Assignments_Missed",
    "Quiz_Attempts",
    "Quiz_Score_Avg",
    "Project_Grade",
    "Progress_Percentage",
    "Rewatch_Count",
    "Fee_Paid",
    "Discount_Used",
    "Payment_Amount",
    "App_Usage_Percentage",
    "Reminder_Emails_Clicked",
    "Support_Tickets_Raised",
    "Satisfaction_Rating"
]
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


