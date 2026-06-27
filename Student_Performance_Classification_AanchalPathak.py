"""
Project 2: Data Classification Using AI


Submitted By: Aanchal Pathak
Role: AI Intern
"""

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Create sample dataset
data = {
    "Study_Hours": [2, 3, 4, 5, 6, 7, 8, 1, 9, 10],
    "Attendance": [50, 55, 60, 65, 70, 80, 85, 45, 90, 95],
    "Result": [
        "Need Improvement",
        "Need Improvement",
        "Need Improvement",
        "Pass",
        "Pass",
        "Pass",
        "Pass",
        "Need Improvement",
        "Pass",
        "Pass"
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

print("Student Dataset")
print(df)

# Features and Target
X = df[["Study_Hours", "Attendance"]]
y = df["Result"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

# Create model
model = DecisionTreeClassifier()

# Train model
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# Test with new student data
new_student = [[7, 75]]

result = model.predict(new_student)

print("\nPrediction for Student")
print("Study Hours:", new_student[0][0])
print("Attendance:", new_student[0][1], "%")
print("Predicted Result:", result[0])