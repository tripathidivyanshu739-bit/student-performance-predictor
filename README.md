# Student Performance Predictor

A machine learning project that predicts a student's final academic performance using student-related academic and personal factors.

## Project Overview

This project uses a **Random Forest Regression** model to predict a student's final grade (G3) out of 20.

The model uses numerical student information such as:

* Age
* Mother's education
* Father's education
* Study time
* Past class failures
* Absences
* Going out frequency
* Health status

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Random Forest Regression
* Streamlit
* Joblib
* UCI Machine Learning Repository dataset

## Model Performance

The current model achieved:

* **Mean Absolute Error (MAE): 2.07**
* **R² Score: 0.17**

These results represent the baseline model performance on the test dataset.

## How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application opens in a web browser where users can enter student information and receive a predicted final grade.

## Project Structure

```text
student-performance-predictor/
├── app.py
├── train_model.py
├── student_performance_model.pkl
├── requirements.txt
├── README.md
├── .gitignore
└── data/
```

## Disclaimer

This project is intended for educational purposes and demonstrates the use of machine learning for student performance prediction. Predictions should not be treated as definitive assessments of a student's actual academic ability.


