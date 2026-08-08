"""
GlucoGuard AI - Streamlit App
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="GlucoGuard AI - Diabetes Risk Predictor", page_icon="🩺", layout="wide")

# ---------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("outputs/final_stacked_model.pkl")
    scaler = joblib.load("outputs/scaler.pkl")
    hybrid_features = joblib.load("outputs/hybrid_features.pkl")
    gb_model = joblib.load("outputs/baseline_models.pkl")["Gradient Boosting"]
    return model, scaler, hybrid_features, gb_model

model, scaler, hybrid_features, gb_model = load_artifacts()

@st.cache_resource
def get_explainer(_gb_model):
    return shap.TreeExplainer(_gb_model)

explainer = get_explainer(gb_model)

# All 21 original features (needed to build the full vector before scaling,
# since the scaler was fit on all 21 columns)
ALL_FEATURES = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
                 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
                 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']

st.title("🩺 GlucoGuard AI — Diabetes Risk Predictor")
st.caption("Hybrid Feature Selection + Cost-Sensitive Stacked Ensemble | Explainable with SHAP")

st.markdown("---")

col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("Enter Patient Details")

    age_map = {"18-24": 1, "25-29": 2, "30-34": 3, "35-39": 4, "40-44": 5, "45-49": 6,
               "50-54": 7, "55-59": 8, "60-64": 9, "65-69": 10, "70-74": 11, "75-79": 12, "80+": 13}
    age_label = st.select_slider("Age group", options=list(age_map.keys()), value="45-49")
    age = age_map[age_label]

    bmi = st.slider("BMI", 12.0, 55.0, 27.0, 0.5)
    genhlth = st.select_slider("General Health (1=Excellent, 5=Poor)", options=[1, 2, 3, 4, 5], value=3)
    physhlth = st.slider("Physically unwell days (last 30 days)", 0, 30, 3)
    highbp = st.toggle("High Blood Pressure")
    highchol = st.toggle("High Cholesterol")
    heart = st.toggle("Heart Disease / Heart Attack history")
    diffwalk = st.toggle("Difficulty Walking / Climbing Stairs")

    with st.expander("Additional details (used for full model input)"):
        cholcheck = st.toggle("Cholesterol checked in last 5 years", value=True)
        smoker = st.toggle("Smoker (100+ cigarettes lifetime)")
        stroke = st.toggle("History of stroke")
        physact = st.toggle("Physically active in last 30 days", value=True)
        fruits = st.toggle("Eats fruit daily", value=True)
        veggies = st.toggle("Eats vegetables daily", value=True)
        alcohol = st.toggle("Heavy alcohol consumption")
        healthcare = st.toggle("Has healthcare coverage", value=True)
        nodoc = st.toggle("Skipped doctor due to cost")
        menthlth = st.slider("Mentally unwell days (last 30 days)", 0, 30, 2)
        sex = st.radio("Sex", ["Female", "Male"], horizontal=True)
        education = st.select_slider("Education level (1=none, 6=college grad)", options=[1,2,3,4,5,6], value=4)
        income = st.select_slider("Income level (1=lowest, 8=highest)", options=[1,2,3,4,5,6,7,8], value=5)

    predict_btn = st.button("🔍 Predict Diabetes Risk", type="primary", use_container_width=True)

with col2:
    st.subheader("Prediction Result")

    if predict_btn:
        row = {
            "HighBP": int(highbp), "HighChol": int(highchol), "CholCheck": int(cholcheck),
            "BMI": bmi, "Smoker": int(smoker), "Stroke": int(stroke),
            "HeartDiseaseorAttack": int(heart), "PhysActivity": int(physact),
            "Fruits": int(fruits), "Veggies": int(veggies), "HvyAlcoholConsump": int(alcohol),
            "AnyHealthcare": int(healthcare), "NoDocbcCost": int(nodoc), "GenHlth": genhlth,
            "MentHlth": menthlth, "PhysHlth": physhlth, "DiffWalk": int(diffwalk),
            "Sex": 1 if sex == "Male" else 0, "Age": age, "Education": education, "Income": income
        }
        input_df = pd.DataFrame([row])[ALL_FEATURES]
        input_scaled = scaler.transform(input_df)
        input_scaled_df = pd.DataFrame(input_scaled, columns=ALL_FEATURES)
        input_hybrid = input_scaled_df[hybrid_features]

        pred = model.predict(input_hybrid)[0]
        proba = model.predict_proba(input_hybrid)[0][1]

        if proba < 0.33:
            risk_label, color = "Low Risk", "🟢"
        elif proba < 0.66:
            risk_label, color = "Moderate Risk", "🟡"
        else:
            risk_label, color = "High Risk", "🔴"

        st.metric("Risk Level", f"{color} {risk_label}")
        st.metric("Predicted Probability of Diabetes", f"{proba*100:.1f}%")
        st.progress(min(proba, 1.0))

        st.markdown("#### Why this prediction — SHAP Explanation")
        gb_input = input_hybrid  # Gradient Boosting was trained on same hybrid features
        shap_vals = explainer.shap_values(gb_input)

        fig, ax = plt.subplots(figsize=(6, 3.5))
        vals = shap_vals[0]
        feat_names = hybrid_features
        order = np.argsort(np.abs(vals))
        colors = ["#C94F3D" if v > 0 else "#2F8F7E" for v in vals[order]]
        ax.barh(np.array(feat_names)[order], vals[order], color=colors)
        ax.set_xlabel("SHAP value (impact on prediction)")
        ax.axvline(0, color="black", linewidth=0.8)
        plt.tight_layout()
        st.pyplot(fig)

        st.caption("Red bars increase risk, green bars decrease risk, for this specific patient.")

        if risk_label != "Low Risk":
            st.warning("⚠️ This is a screening tool, not a diagnosis. Please consult a healthcare professional for proper testing.")
    else:
        st.info("Fill in the patient details and click **Predict Diabetes Risk** to see results.")

st.markdown("---")
st.caption("GlucoGuard AI — Cost-Sensitive Stacked Ensemble (Logistic Regression + Random Forest + "
           "Gradient Boosting, meta-learner: Logistic Regression) | Trained on the CDC BRFSS 2015 "
           "Diabetes Health Indicators Dataset (229,474 records) | Explainable with SHAP")
