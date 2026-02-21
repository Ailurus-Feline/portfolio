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

consumption_path = DATA_DIR / "online_learning_course_consumption_dataset.csv"
completion_path  = DATA_DIR / "Course_Completion_Prediction.csv"
usage_path       = DATA_DIR / "online_courses_uses.csv"

df_consumption = pd.read_csv(consumption_path)
df_completion  = pd.read_csv(completion_path)
df_usage       = pd.read_csv(usage_path)

print("Consumption:", df_consumption.shape)
print("Completion :", df_completion.shape)
print("Usage      :", df_usage.shape)