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

# Step 3: Split Data into Input (X) and Output (y)
X = df[['study_hours', 'attendance', 'prev_score']]  # features
y = df['result']                                      # label (pass/fail)

# Step 4: Split into Training and Testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the Decision Tree Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Step 6: Test the Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("✅ Model trained successfully!")
print(f"🎯 Accuracy: {accuracy * 100:.2f}%")

# Step 7: Show Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fail', 'Pass'],
            yticklabels=['Fail', 'Pass'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
# Step 8: Predict for a NEW Student
print("\n--- 🎓 Student Performance Predictor ---")
study = float(input("Enter study hours per day: "))
attend = float(input("Enter attendance percentage: "))
score = float(input("Enter previous exam score: "))

# Make Prediction
new_student = [[study, attend, score]]
prediction = model.predict(new_student)

if prediction[0] == 1:
    print("\n✅ Result: Student is likely to PASS! 🎉")
else:
    print("\n❌ Result: Student is likely to FAIL. Study more! 📚")