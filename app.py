import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# ---- Train Model ----
data = {
    'study_hours': [1, 2, 3, 4, 5, 6, 7, 8, 2, 3, 5, 7, 1, 4, 6],
    'attendance':  [40, 50, 60, 70, 80, 90, 95, 99, 45, 55, 75, 85, 35, 65, 88],
    'prev_score':  [30, 40, 50, 60, 70, 75, 85, 90, 35, 55, 65, 80, 25, 60, 78],
    'result':      [0,  0,  0,  1,  1,  1,  1,  1,  0,  0,  1,  1,  0,  1,  1]
}
df = pd.DataFrame(data)
X = df[['study_hours', 'attendance', 'prev_score']]
y = df['result']
model = DecisionTreeClassifier()
model.fit(X, y)

# ---- UI ----
st.title("Student Performance Predictor")
st.write("Predict if a student will Pass or Fail")

study = st.slider("Study Hours Per Day", 0, 12, 5)
attend = st.slider("Attendance Percentage", 0, 100, 75)
score = st.slider("Previous Exam Score", 0, 100, 60)

if st.button("Predict"):
    result = model.predict([[study, attend, score]])
    if result[0] == 1:
        st.success("Student will PASS!")
        st.balloons()
    else:
        st.error("Student will FAIL!")