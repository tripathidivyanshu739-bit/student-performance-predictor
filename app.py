import streamlit as st
import joblib
import pandas as pd

model = joblib.load("student_performance_model.pkl")

st.title("Student Performance Predictor")
st.write("Predict a student's final academic grade based on academic and personal factors.")

st.header("Student Information")

age = st.number_input("Age", 15, 25, 18)
Medu = st.slider("Mother's Education Level", 0, 4, 2)
Fedu = st.slider("Father's Education Level", 0, 4, 2)
studytime = st.slider("Weekly Study Time", 1, 4, 2)
failures = st.slider("Past Class Failures", 0, 3, 0)
absences = st.number_input("Number of Absences", 0, 100, 5)
goout = st.slider("Going Out With Friends", 1, 5, 3)
health = st.slider("Current Health Status", 1, 5, 3)

if st.button("Predict Performance"):

    input_data = pd.DataFrame({
        "age": [age],
        "Medu": [Medu],
        "Fedu": [Fedu],
        "studytime": [studytime],
        "failures": [failures],
        "absences": [absences],
        "goout": [goout],
        "health": [health]
    })

    # Fill remaining model features with their average training values
    for feature in model.feature_names_in_:
        if feature not in input_data.columns:
            input_data[feature] = 0

    input_data = input_data[model.feature_names_in_]

    prediction = model.predict(input_data)[0]

    prediction = max(0, min(20, prediction))

    st.success(f"Predicted Final Grade: {prediction:.2f} / 20")

    if prediction >= 15:
        st.info("Expected performance: Excellent")
    elif prediction >= 10:
        st.info("Expected performance: Average")
    else:
        st.info("Expected performance: Needs Improvement")