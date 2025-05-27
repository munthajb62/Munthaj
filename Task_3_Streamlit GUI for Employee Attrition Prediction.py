import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
# Load and preprocess dataset
@st.cache_data
def load_and_train_model():
    df = pd.read_csv(r"C:\Users\User\Desktop\BCAS\anis sir\ML\WA_Fn-UseC_-HR-Employee-Attrition.csv")
    df.drop(['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours'], axis=1, inplace=True)
    df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)
    return model, scaler, X.columns.tolist(), X.mean()
# Load model
model, scaler, feature_names, feature_means = load_and_train_model()
# Streamlit UI
st.title("💼 Employee Attrition Predictor (Logistic Regression)")
st.write("This app predicts whether an employee is likely to leave the company based on their HR information.")

# Input features (simplified for UI)
age = st.slider("Age", 18, 60, 30)
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=30000, value=5000)
job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4], index=2)
distance_from_home = st.slider("Distance From Home (km)", 1, 50, 10)
overtime = st.selectbox("OverTime", ["Yes", "No"])
years_at_company = st.slider("Years at Company", 0, 40, 5)
if st.button("Predict Attrition"):
    # Prepare input
    overtime_val = 1 if overtime == "Yes" else 0
    input_dict = {
        "Age": age,
        "MonthlyIncome": monthly_income,
        "JobSatisfaction": job_satisfaction,
        "DistanceFromHome": distance_from_home,
        "OverTime": overtime_val,
        "YearsAtCompany": years_at_company
    }
    # Fill remaining features with mean
    user_input = pd.DataFrame([input_dict])
    for feat in feature_names:
        if feat not in user_input.columns:
            user_input[feat] = feature_means[feat]
    user_input = user_input[feature_names]
    scaled_input = scaler.transform(user_input)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]
    if prediction == 1:
        st.error(f"⚠️ Employee is likely to leave. Probability: {probability:.2f}")
    else:
        st.success(f"✅ Employee is likely to stay. Probability: {1 - probability:.2f}")
