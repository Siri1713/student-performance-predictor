# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Create Sample Dataset
data = {
    'study_hours': [1, 2, 3, 4, 5, 6, 7, 8, 2, 3, 5, 7, 1, 4, 6],
    'attendance':  [40, 50, 60, 70, 80, 90, 95, 99, 45, 55, 75, 85, 35, 65, 88],
    'prev_score':  [30, 40, 50, 60, 70, 75, 85, 90, 35, 55, 65, 80, 25, 60, 78],
    'result':      [0,  0,  0,  1,  1,  1,  1,  1,  0,  0,  1,  1,  0,  1,  1]
}

df = pd.DataFrame(data)
print("✅ Dataset loaded successfully!")
print(df.head())