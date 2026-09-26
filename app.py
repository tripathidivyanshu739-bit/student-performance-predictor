import streamlit as st
import pandas as pd
import joblib

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    .main-title {
        text-align: center;
        font-size: 44px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #777;
        font-size: 18px;
        margin-bottom: 35px;
    }

    .section-header {
        font-size: 26px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    .result-card {
        border: 1px solid #dddddd;
        border-radius: 18px;
        padding: 30px;
        text-align: center;
        margin-top: 25px;
        margin-bottom: 25px;
    }

    .result-label {
        color: #777;
        font-size: 16px;
    }

    .score {
        font-size: 60px;
        font-weight: 800;
        margin: 5px 0;
    }

    .performance {
        font-size: 26px;
        font-weight: 700;
    }

    .result-subtitle {
        color: #777;
        margin-top: 8px;
    }

    /* Predict button */

    div.stButton > button {
        width: 100%;
        height: 55px;
        border-radius: 10px;
        border: none;
        background-color: #e53935;
        color: white;
        font-size: 18px;
        font-weight: 700;
        transition: 0.2s;
    }

    div.stButton > button:hover {
        background-color: #c62828;
        color: white;
        border: none;
    }

    div.stButton > button:focus {
        background-color: #e53935;
        color: white;
    }

    .metric-box {
        text-align: center;
        padding: 15px;
        border: 1px solid #dddddd;
        border-radius: 12px;
    }

    .footer {
        text-align: center;
        color: #888;
        font-size: 13px;
        margin-top: 40px;
    }

</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("student_performance_model.pkl")

# Actual results from current 25-feature model
MAE = 2.07
R2 = 0.19

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("About This Project")

    st.write(
        "Student Performance Predictor is a machine learning application "
        "that estimates a student's final academic performance using "
        "25 academic, demographic, family, social and lifestyle features."
    )

    st.subheader("Dataset")

    st.write(
        "The project uses the UCI Student Performance dataset. "
        "The target variable is G3, the student's final grade on a "
        "0–20 scale."
    )

    st.subheader("Machine Learning Model")

    st.write(
        "A Random Forest Regressor with 300 trees is used to predict "
        "the student's final G3 grade."
    )

    st.subheader("Model Performance")

    st.write(f"MAE: {MAE:.2f}")
    st.write(f"R² Score: {R2:.2f}")

    st.subheader("Training")

    st.write(
        "The dataset is divided into 80% training data and 20% testing data."
    )

    st.subheader("Important Note")

    st.write(
        "The prediction is an estimate based on patterns learned from "
        "the dataset. It should not be considered an official academic result."
    )

# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">Student Performance Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Based Academic Performance Prediction'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "Complete the student profile below to estimate the expected "
    "final academic performance."
)

# ============================================================
# STUDENT INFORMATION
# ============================================================

st.markdown(
    '<div class="section-header">Student Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        min_value=15,
        max_value=25,
        value=18,
        step=1
    )

with col2:
    address_label = st.selectbox(
        "Home Location",
        ["Urban", "Rural"]
    )

with col3:
    famsize_label = st.selectbox(
        "Family Size",
        ["Small", "Large"]
    )

col1, col2, col3 = st.columns(3)

with col1:
    pstatus_label = st.selectbox(
        "Parents' Living Status",
        ["Together", "Apart"]
    )

with col2:
    traveltime_label = st.selectbox(
        "Travel Time to School",
        [
            "Less than 15 minutes",
            "15–30 minutes",
            "30–60 minutes",
            "More than 60 minutes"
        ]
    )

with col3:
    absences = st.number_input(
        "Number of School Absences",
        min_value=0,
        max_value=100,
        value=5,
        step=1
    )

# ============================================================
# FAMILY & EDUCATION
# ============================================================

st.markdown(
    '<div class="section-header">Family & Education</div>',
    unsafe_allow_html=True
)

education_options = [
    "No formal education",
    "Primary education",
    "Middle school",
    "Secondary education",
    "Higher education"
]

col1, col2 = st.columns(2)

with col1:
    medu_label = st.selectbox(
        "Mother's Education",
        education_options
    )

with col2:
    fedu_label = st.selectbox(
        "Father's Education",
        education_options
    )

col1, col2 = st.columns(2)

with col1:
    mjob_label = st.selectbox(
        "Mother's Occupation",
        [
            "Teacher",
            "Healthcare",
            "Civil services",
            "At home",
            "Other"
        ]
    )

with col2:
    fjob_label = st.selectbox(
        "Father's Occupation",
        [
            "Teacher",
            "Healthcare",
            "Civil services",
            "At home",
            "Other"
        ]
    )

col1, col2 = st.columns(2)

with col1:
    reason_label = st.selectbox(
        "Main Reason for Choosing School",
        [
            "Close to home",
            "School reputation",
            "Preference for a specific course",
            "Other"
        ]
    )

with col2:
    guardian_label = st.selectbox(
        "Main Guardian",
        [
            "Mother",
            "Father",
            "Other"
        ]
    )

# ============================================================
# STUDY HABITS
# ============================================================

st.markdown(
    '<div class="section-header">Study Habits & Academic Support</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    studytime_label = st.selectbox(
        "Weekly Study Time",
        [
            "Less than 2 hours",
            "2–5 hours",
            "5–10 hours",
            "More than 10 hours"
        ]
    )

with col2:
    failures_label = st.selectbox(
        "Previous Class Failures",
        [
            "None",
            "1",
            "2",
            "3 or more"
        ]
    )

col1, col2, col3 = st.columns(3)

with col1:
    schoolsup_label = st.selectbox(
        "Extra Educational Support",
        ["Yes", "No"]
    )

with col2:
    famsup_label = st.selectbox(
        "Family Educational Support",
        ["Yes", "No"]
    )

with col3:
    paid_label = st.selectbox(
        "Paid Extra Classes",
        ["Yes", "No"]
    )

col1, col2 = st.columns(2)

with col1:
    activities_label = st.selectbox(
        "Participates in Extracurricular Activities",
        ["Yes", "No"]
    )

with col2:
    nursery_label = st.selectbox(
        "Attended Nursery School",
        ["Yes", "No"]
    )

higher_label = st.selectbox(
    "Wants to Pursue Higher Education",
    ["Yes", "No"]
)

# ============================================================
# SOCIAL & HEALTH
# ============================================================

st.markdown(
    '<div class="section-header">Social Life & Health</div>',
    unsafe_allow_html=True
)

level_options = [
    "Very Low",
    "Low",
    "Moderate",
    "High",
    "Very High"
]

quality_options = [
    "Very Poor",
    "Poor",
    "Average",
    "Good",
    "Excellent"
]

col1, col2, col3 = st.columns(3)

with col1:
    famrel_label = st.selectbox(
        "Family Relationship",
        quality_options
    )

with col2:
    freetime_label = st.selectbox(
        "Free Time",
        level_options
    )

with col3:
    goout_label = st.selectbox(
        "Going Out With Friends",
        level_options
    )

col1, col2 = st.columns(2)

with col1:
    health_label = st.selectbox(
        "Overall Health",
        quality_options
    )

with col2:
    internet_label = st.selectbox(
        "Internet Access at Home",
        ["Yes", "No"]
    )

# ============================================================
# CONVERT HUMAN INPUTS TO DATASET VALUES
# ============================================================

education_map = {
    "No formal education": 0,
    "Primary education": 1,
    "Middle school": 2,
    "Secondary education": 3,
    "Higher education": 4
}

address_map = {
    "Urban": "U",
    "Rural": "R"
}

famsize_map = {
    "Small": "LE3",
    "Large": "GT3"
}

pstatus_map = {
    "Together": "T",
    "Apart": "A"
}

traveltime_map = {
    "Less than 15 minutes": 1,
    "15–30 minutes": 2,
    "30–60 minutes": 3,
    "More than 60 minutes": 4
}

studytime_map = {
    "Less than 2 hours": 1,
    "2–5 hours": 2,
    "5–10 hours": 3,
    "More than 10 hours": 4
}

level_map = {
    "Very Low": 1,
    "Low": 2,
    "Moderate": 3,
    "High": 4,
    "Very High": 5
}

quality_map = {
    "Very Poor": 1,
    "Poor": 2,
    "Average": 3,
    "Good": 4,
    "Excellent": 5
}

failures_map = {
    "None": 0,
    "1": 1,
    "2": 2,
    "3 or more": 3
}

job_map = {
    "Teacher": "teacher",
    "Healthcare": "health",
    "Civil services": "services",
    "At home": "at_home",
    "Other": "other"
}

reason_map = {
    "Close to home": "home",
    "School reputation": "reputation",
    "Preference for a specific course": "course",
    "Other": "other"
}

guardian_map = {
    "Mother": "mother",
    "Father": "father",
    "Other": "other"
}

# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict = st.button("Predict Student Performance")

# ============================================================
# PREDICTION
# ============================================================

if predict:

    input_data = pd.DataFrame({
        "age": [age],
        "address": [address_map[address_label]],
        "famsize": [famsize_map[famsize_label]],
        "Pstatus": [pstatus_map[pstatus_label]],
        "Medu": [education_map[medu_label]],
        "Fedu": [education_map[fedu_label]],
        "Mjob": [job_map[mjob_label]],
        "Fjob": [job_map[fjob_label]],
        "reason": [reason_map[reason_label]],
        "guardian": [guardian_map[guardian_label]],
        "traveltime": [traveltime_map[traveltime_label]],
        "studytime": [studytime_map[studytime_label]],
        "failures": [failures_map[failures_label]],
        "schoolsup": [schoolsup_label.lower()],
        "famsup": [famsup_label.lower()],
        "paid": [paid_label.lower()],
        "activities": [activities_label.lower()],
        "nursery": [nursery_label.lower()],
        "higher": [higher_label.lower()],
        "internet": [internet_label.lower()],
        "famrel": [quality_map[famrel_label]],
        "freetime": [level_map[freetime_label]],
        "goout": [level_map[goout_label]],
        "health": [quality_map[health_label]],
        "absences": [absences]
    })

    # Exact feature order
    input_data = input_data[model.feature_names_in_]

    # Predict
    prediction = float(model.predict(input_data)[0])

    # Keep G3 within valid range
    prediction = max(0, min(20, prediction))

    # Convert /20 to /100
    score_100 = prediction * 5

    # Performance category
    if score_100 >= 80:
        category = "Excellent"
    elif score_100 >= 60:
        category = "Good"
    elif score_100 >= 40:
        category = "Average"
    else:
        category = "Needs Improvement"

    # ========================================================
    # RESULT
    # ========================================================

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Estimated Final Score</div>
            <div class="score">{score_100:.1f} / 100</div>
            <div class="performance">{category}</div>
            <div class="result-subtitle">
                Equivalent predicted G3 grade: {prediction:.1f} / 20
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Predicted Score",
            f"{score_100:.1f}/100"
        )

    with col2:
        st.metric(
            "G3 Grade",
            f"{prediction:.1f}/20"
        )

    with col3:
        st.metric(
            "Performance",
            category
        )

    # ========================================================
    # INPUT SUMMARY
    # ========================================================

    st.subheader("Student Profile")

    summary = pd.DataFrame({
        "Factor": [
            "Age",
            "Home Location",
            "Family Size",
            "Parents' Status",
            "Travel Time",
            "Absences",
            "Mother's Education",
            "Father's Education",
            "Mother's Occupation",
            "Father's Occupation",
            "School Choice Reason",
            "Main Guardian",
            "Weekly Study Time",
            "Previous Failures",
            "Extra Educational Support",
            "Family Support",
            "Paid Extra Classes",
            "Extracurricular Activities",
            "Nursery School",
            "Higher Education",
            "Family Relationship",
            "Free Time",
            "Going Out",
            "Overall Health",
            "Internet Access"
        ],
        "Selected Value": [
            age,
            address_label,
            famsize_label,
            pstatus_label,
            traveltime_label,
            absences,
            medu_label,
            fedu_label,
            mjob_label,
            fjob_label,
            reason_label,
            guardian_label,
            studytime_label,
            failures_label,
            schoolsup_label,
            famsup_label,
            paid_label,
            activities_label,
            nursery_label,
            higher_label,
            famrel_label,
            freetime_label,
            goout_label,
            health_label,
            internet_label
        ]
    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# PROJECT DETAILS
# ============================================================

st.divider()

st.markdown(
    '<div class="section-header">Project Details</div>',
    unsafe_allow_html=True
)

with st.expander("What does this project do?"):
    st.write(
        "This application uses machine learning to estimate a student's "
        "final academic performance from demographic, academic, family, "
        "social and lifestyle information."
    )

with st.expander("Dataset"):
    st.write(
        "The project uses the UCI Student Performance dataset. "
        "The original final grade variable, G3, ranges from 0 to 20."
    )

with st.expander("Features Used"):
    st.write(
        "The model uses 25 features covering student demographics, "
        "family background, education, study habits, academic support, "
        "social life, health and school attendance."
    )

with st.expander("Machine Learning Algorithm"):
    st.write(
        "A Random Forest Regressor with 300 trees is used. "
        "The model combines multiple decision trees to estimate "
        "the final academic grade."
    )

with st.expander("Training and Testing"):
    st.write(
        "The dataset is divided into 80% training data and 20% testing data. "
        "Categorical features are converted using one-hot encoding as part "
        "of the machine learning pipeline."
    )

with st.expander("Model Evaluation"):
    st.write(f"Mean Absolute Error (MAE): {MAE:.2f}")
    st.write(f"R² Score: {R2:.2f}")

    st.write(
        "MAE represents the average absolute difference between predicted "
        "and actual grades. R² indicates how much variation in the target "
        "is explained by the model."
    )

with st.expander("How does the prediction work?"):
    st.write(
        "The user's answers are converted into the format expected by the "
        "dataset. The preprocessing pipeline then encodes categorical "
        "variables and passes all features to the trained Random Forest "
        "model. The predicted G3 grade is finally converted from a 0–20 "
        "scale to a 0–100 score."
    )

with st.expander("Limitations"):
    st.write(
        "The model learns patterns from a specific dataset and cannot "
        "account for every factor affecting academic performance. "
        "Therefore, its output is an estimate rather than an exact result."
    )

with st.expander("Disclaimer"):
    st.write(
        "This application was created as an educational machine learning "
        "project. Predictions should not be considered official academic "
        "evaluations."
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">'
    'Student Performance Predictor | Machine Learning Project'
    '</div>',
    unsafe_allow_html=True
)