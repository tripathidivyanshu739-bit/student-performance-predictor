import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# STUDENT PERFORMANCE PREDICTOR
# ==========================================

# Create our dataset
data = {
    "study_hours": [2, 3, 4, 5, 6, 7, 8, 9, 1, 3,
                    5, 6, 7, 4, 8, 2, 5, 6, 9, 3,
                    4, 7, 5, 8, 2, 6, 7, 4, 9, 3],

    "attendance": [60, 65, 70, 75, 80, 85, 90, 92, 55, 68,
                   78, 82, 88, 72, 94, 62, 76, 86, 95, 67,
                   73, 89, 79, 91, 58, 83, 87, 69, 96, 64],

    "previous_score": [55, 58, 62, 68, 72, 78, 84, 88, 50, 60,
                       70, 75, 80, 65, 90, 54, 69, 77, 92, 59,
                       64, 82, 71, 86, 52, 74, 81, 61, 94, 57],

    "assignment_score": [58, 62, 65, 70, 75, 80, 86, 90, 52, 64,
                         72, 78, 84, 68, 92, 57, 73, 81, 94, 63,
                         67, 85, 76, 89, 55, 79, 83, 66, 96, 61],

    "sleep_hours": [6, 6, 7, 7, 7, 8, 8, 8, 6, 7,
                    7, 8, 8, 6, 9, 6, 7, 8, 9, 6,
                    7, 8, 7, 8, 6, 7, 8, 6, 9, 7],

    "final_score": [56, 60, 65, 71, 76, 82, 88, 92, 51, 62,
                    73, 79, 85, 68, 94, 57, 72, 81, 96, 61,
                    67, 86, 74, 91, 54, 78, 84, 65, 97, 63]
}

# Convert dictionary into a Pandas DataFrame
data = pd.DataFrame(data)

print("=" * 50)
print("       STUDENT PERFORMANCE PREDICTOR")
print("=" * 50)

# ==========================================
# SELECT FEATURES AND TARGET
# ==========================================

X = data[
    [
        "study_hours",
        "attendance",
        "previous_score",
        "assignment_score",
        "sleep_hours"
    ]
]

y = data["final_score"]

# ==========================================
# SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# CREATE AND TRAIN MODEL
# ==========================================

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel trained successfully!")

# ==========================================
# EVALUATE MODEL
# ==========================================

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Evaluation")
print("-" * 30)
print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 2))

# ==========================================
# GET STUDENT INFORMATION
# ==========================================

print("\nEnter Student Information")
print("-" * 30)

study_hours = float(input("Study hours per day: "))
attendance = float(input("Attendance percentage: "))
previous_score = float(input("Previous exam score: "))
assignment_score = float(input("Assignment score: "))
sleep_hours = float(input("Sleep hours per day: "))

# ==========================================
# MAKE PREDICTION
# ==========================================

new_student = pd.DataFrame(
    [[
        study_hours,
        attendance,
        previous_score,
        assignment_score,
        sleep_hours
    ]],
    columns=[
        "study_hours",
        "attendance",
        "previous_score",
        "assignment_score",
        "sleep_hours"
    ]
)

prediction = model.predict(new_student)

predicted_score = prediction[0]

# Keep score between 0 and 100
predicted_score = max(0, min(100, predicted_score))

# ==========================================
# DISPLAY RESULT
# ==========================================

print("\n" + "=" * 50)
print("              PREDICTION RESULT")
print("=" * 50)

print("Predicted Final Score:", round(predicted_score, 2))

if predicted_score >= 90:
    print("Performance Level: Excellent")
elif predicted_score >= 75:
    print("Performance Level: Good")
elif predicted_score >= 60:
    print("Performance Level: Average")
else:
    print("Performance Level: Needs Improvement")

print("=" * 50)