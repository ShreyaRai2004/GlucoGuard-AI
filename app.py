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
import re
from PIL import Image

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

ALL_FEATURES = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
                 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
                 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']

AGE_MAP = {"18-24": 1, "25-29": 2, "30-34": 3, "35-39": 4, "40-44": 5, "45-49": 6,
           "50-54": 7, "55-59": 8, "60-64": 9, "65-69": 10, "70-74": 11, "75-79": 12, "80+": 13}
AGE_BRACKETS = [(24, "18-24"), (29, "25-29"), (34, "30-34"), (39, "35-39"), (44, "40-44"),
                (49, "45-49"), (54, "50-54"), (59, "55-59"), (64, "60-64"), (69, "65-69"),
                (74, "70-74"), (79, "75-79"), (200, "80+")]

# ---------------------------------------------------------
# Session state defaults (single source of truth for all form fields)
# ---------------------------------------------------------
DEFAULTS = {
    "age_label": "45-49", "bmi": 27.0, "genhlth": 3, "physhlth": 3,
    "highbp": False, "highchol": False, "heart": False, "diffwalk": False,
    "cholcheck": True, "smoker": False, "stroke": False, "physact": True,
    "fruits": True, "veggies": True, "alcohol": False, "healthcare": True,
    "nodoc": False, "menthlth": 2, "sex": "Female", "education": 4, "income": 5,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------
# OCR extraction logic
# ---------------------------------------------------------
def age_to_bracket(age_years):
    for upper, label in AGE_BRACKETS:
        if age_years <= upper:
            return label
    return "80+"

def genhlth_from_text(text):
    text = text.lower()
    if "excellent" in text:
        return 1
    if "very good" in text:
        return 2
    if "good" in text and "very" not in text:
        return 2
    if "fair" in text:
        return 4
    if "poor" in text:
        return 5
    return None

def run_ocr(uploaded_file):
    """Extract raw text from an uploaded image or PDF."""
    import pytesseract
    name = uploaded_file.name.lower()
    text = ""
    if name.endswith(".pdf"):
        from pdf2image import convert_from_bytes
        pages = convert_from_bytes(uploaded_file.read())
        for page in pages:
            text += pytesseract.image_to_string(page) + "\n"
    else:
        image = Image.open(uploaded_file)
        text = pytesseract.image_to_string(image)
    return text

def extract_fields_from_text(text):
    """Parse OCR text and return a dict of session_state updates + a list of
    (field, value, found) for a human-readable summary."""
    updates = {}
    summary = []

    m = re.search(r"BMI[:\s]+(\d+\.?\d*)", text, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        updates["bmi"] = val
        summary.append(("BMI", f"{val}", True))
    else:
        summary.append(("BMI", "not found — kept default", False))

    m = re.search(r"Age[:\s]+(\d+)", text, re.IGNORECASE)
    if m:
        val = int(m.group(1))
        updates["age_label"] = age_to_bracket(val)
        summary.append(("Age", f"{val} years -> {updates['age_label']}", True))
    else:
        summary.append(("Age", "not found — kept default", False))

    m = re.search(r"Blood Pressure[:\s]+(\d+)\s*/\s*(\d+)", text, re.IGNORECASE)
    if m:
        systolic, diastolic = int(m.group(1)), int(m.group(2))
        high_bp = systolic >= 140 or diastolic >= 90
        updates["highbp"] = high_bp
        summary.append(("Blood Pressure", f"{systolic}/{diastolic} -> High BP: {high_bp}", True))
    else:
        m2 = re.search(r"hypertension|high blood pressure", text, re.IGNORECASE)
        if m2:
            updates["highbp"] = True
            summary.append(("Blood Pressure", "mentioned as hypertensive", True))
        else:
            summary.append(("Blood Pressure", "not found — kept default", False))

    m = re.search(r"Cholesterol[:\s]+(\d+\.?\d*)", text, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        high_chol = val >= 200
        updates["highchol"] = high_chol
        updates["cholcheck"] = True
        summary.append(("Cholesterol", f"{val} mg/dL -> High Cholesterol: {high_chol}", True))
    else:
        summary.append(("Cholesterol", "not found — kept default", False))

    m = re.search(r"Heart Disease[^\n:]*[:\s]+(\w+)", text, re.IGNORECASE)
    if m:
        val = "yes" in m.group(1).lower()
        updates["heart"] = val
        summary.append(("Heart Disease History", str(val), True))
    else:
        summary.append(("Heart Disease History", "not found — kept default", False))

    m = re.search(r"Smoker[^\n:]*[:\s]+(\w+)", text, re.IGNORECASE)
    if m:
        val = "yes" in m.group(1).lower()
        updates["smoker"] = val
        summary.append(("Smoker", str(val), True))
    else:
        summary.append(("Smoker", "not found — kept default", False))

    m = re.search(r"Stroke[^\n:]*[:\s]+(\w+)", text, re.IGNORECASE)
    if m:
        val = "yes" in m.group(1).lower()
        updates["stroke"] = val
        summary.append(("Stroke History", str(val), True))
    else:
        summary.append(("Stroke History", "not found — kept default", False))

    m = re.search(r"(?:General Health|Health Assessment)[^\n:]*[:\s]+(\w+\s?\w*)", text, re.IGNORECASE)
    if m:
        gh = genhlth_from_text(m.group(1))
        if gh:
            updates["genhlth"] = gh
            summary.append(("General Health", f"{m.group(1).strip()} -> level {gh}", True))
    m = re.search(r"Sex[:\s]+(\w+)", text, re.IGNORECASE)
    if m:
        val = "Male" if "m" in m.group(1).lower()[0:1] else "Female"
        updates["sex"] = val
        summary.append(("Sex", val, True))

    return updates, summary

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("🩺 GlucoGuard AI — Diabetes Risk Predictor")
st.caption("Hybrid Feature Selection + Cost-Sensitive Stacked Ensemble | Explainable with SHAP")
st.markdown("---")

# ---------------------------------------------------------
# Input method selector
# ---------------------------------------------------------
input_method = st.radio("Choose input method", ["Manual Entry", "Upload Medical Report"], horizontal=True)

if input_method == "Upload Medical Report":
    st.markdown("#### Upload a Medical Report")
    st.caption("Supported: JPG, PNG, PDF. The system will read the report and pre-fill the form below — "
               "please review the extracted values before predicting, since OCR is not always 100% accurate.")
    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png", "pdf"])

    if uploaded_file is not None:
        if st.button("📄 Extract Values from Report", type="secondary"):
            with st.spinner("Reading report..."):
                try:
                    text = run_ocr(uploaded_file)
                    updates, summary = extract_fields_from_text(text)
                    for k, v in updates.items():
                        st.session_state[k] = v
                    st.session_state["_last_extraction_summary"] = summary
                    st.success(f"Extracted {len(updates)} field(s) from the report. Review the form below before predicting.")
                except Exception as e:
                    st.error(f"Could not read this file. Try a clearer image or a different format. ({e})")

    if "_last_extraction_summary" in st.session_state:
        with st.expander("What was extracted from the report"):
            for field, value, found in st.session_state["_last_extraction_summary"]:
                icon = "✅" if found else "⚠️"
                st.write(f"{icon} **{field}:** {value}")

st.markdown("---")

col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("Patient Details (review before predicting)")

    age_label = st.select_slider("Age group", options=list(AGE_MAP.keys()), key="age_label")
    age = AGE_MAP[age_label]

    bmi = st.slider("BMI", 12.0, 55.0, key="bmi", step=0.5)
    genhlth = st.select_slider("General Health (1=Excellent, 5=Poor)", options=[1, 2, 3, 4, 5], key="genhlth")
    physhlth = st.slider("Physically unwell days (last 30 days)", 0, 30, key="physhlth")
    highbp = st.toggle("High Blood Pressure", key="highbp")
    highchol = st.toggle("High Cholesterol", key="highchol")
    heart = st.toggle("Heart Disease / Heart Attack history", key="heart")
    diffwalk = st.toggle("Difficulty Walking / Climbing Stairs", key="diffwalk")

    with st.expander("Additional details (used for full model input)"):
        cholcheck = st.toggle("Cholesterol checked in last 5 years", key="cholcheck")
        smoker = st.toggle("Smoker (100+ cigarettes lifetime)", key="smoker")
        stroke = st.toggle("History of stroke", key="stroke")
        physact = st.toggle("Physically active in last 30 days", key="physact")
        fruits = st.toggle("Eats fruit daily", key="fruits")
        veggies = st.toggle("Eats vegetables daily", key="veggies")
        alcohol = st.toggle("Heavy alcohol consumption", key="alcohol")
        healthcare = st.toggle("Has healthcare coverage", key="healthcare")
        nodoc = st.toggle("Skipped doctor due to cost", key="nodoc")
        menthlth = st.slider("Mentally unwell days (last 30 days)", 0, 30, key="menthlth")
        sex = st.radio("Sex", ["Female", "Male"], horizontal=True, key="sex")
        education = st.select_slider("Education level (1=none, 6=college grad)", options=[1,2,3,4,5,6], key="education")
        income = st.select_slider("Income level (1=lowest, 8=highest)", options=[1,2,3,4,5,6,7,8], key="income")

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
        gb_input = input_hybrid
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
        st.info("Fill in the patient details (or upload a report above) and click **Predict Diabetes Risk** to see results.")

st.markdown("---")
st.caption("GlucoGuard AI — Cost-Sensitive Stacked Ensemble (Logistic Regression + Random Forest + "
           "Gradient Boosting, meta-learner: Logistic Regression) | Trained on the CDC BRFSS 2015 "
           "Diabetes Health Indicators Dataset (229,474 records) | Explainable with SHAP")
