import pandas as pd
import joblib

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# Load UCI Student Performance dataset
dataset = fetch_ucirepo(id=320)

X = dataset.data.features.copy()
y = dataset.data.targets["G3"].copy()


# Features used by the model
features = [
    "age",
    "address",
    "famsize",
    "Pstatus",
    "Medu",
    "Fedu",
    "Mjob",
    "Fjob",
    "reason",
    "guardian",
    "traveltime",
    "studytime",
    "failures",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "famrel",
    "freetime",
    "goout",
    "health",
    "absences"
]

X = X[features]

# Make sure target is numeric
y = pd.to_numeric(y, errors="coerce")

# Remove invalid rows
valid_rows = y.notna()

X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].copy()


# Numeric features
numeric_features = [
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "famrel",
    "freetime",
    "goout",
    "health",
    "absences"
]


# Categorical features
categorical_features = [
    "address",
    "famsize",
    "Pstatus",
    "Mjob",
    "Fjob",
    "reason",
    "guardian",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet"
]


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)


# Random Forest model
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)


# Complete ML pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Train
pipeline.fit(X_train, y_train)


# Predict
predictions = pipeline.predict(X_test)


# Evaluate
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"MAE: {mae:.2f}")
print(f"R2 Score: {r2:.2f}")


# Save complete pipeline
joblib.dump(pipeline, "student_performance_model.pkl")

print("Model saved successfully.")

print("\nFeatures used:")
for feature in features:
    print("-", feature)