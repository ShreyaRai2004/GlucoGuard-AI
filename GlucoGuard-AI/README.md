# GlucoGuard AI

A machine learning system that predicts diabetes risk — either from manually entered health details, or automatically from an uploaded medical report (image or PDF). Designed to catch more at-risk patients than standard accuracy-driven models, with every prediction explained via SHAP.

## Two Ways to Use It

1. **Manual Entry** — enter BMI, blood pressure, cholesterol, age, etc. yourself using sliders/toggles
2. **Upload Medical Report** — upload a photo or PDF of a real health report; the app reads it using OCR, pre-fills the same form automatically, and shows exactly what it found so you can review before predicting

Either path feeds into the **same trained model** and produces the same kind of result: a diabetes risk prediction with a SHAP explanation of which factors drove it.

## Results

| Model | Accuracy | Recall |
|---|---|---|
| Best baseline (Gradient Boosting) | 85.5% | 15.7% |
| **GlucoGuard AI (proposed)** | **83.6%** | **41.3%** |

The proposed model catches roughly **4 out of 10 actual diabetic patients**, nearly triple the baseline's ~1.5 out of 10 — statistically significant (p < 0.001), at a small accuracy trade-off.

## Method

1. **Hybrid Feature Selection** — Mutual Information + Recursive Feature Elimination, intersected → 8 features
2. **Cost-Sensitive Stacked Ensemble** — Logistic Regression + Random Forest + Gradient Boosting, meta-learner: Logistic Regression, with class weighting to reduce false negatives
3. **OCR Report Extraction** — Tesseract OCR reads uploaded reports and maps detected values (BMI, BP, cholesterol, age, medical history) to model inputs
4. **SHAP** — explains each individual prediction
5. **Statistical validation** — paired t-test / Wilcoxon on cross-validated recall

## Tech Stack

Python, pandas, NumPy, scikit-learn, SHAP, SciPy, Matplotlib, Seaborn, Streamlit, Tesseract OCR (pytesseract), pdf2image

## Dataset

CDC Diabetes Health Indicators (BRFSS 2015) — 253,680 records, 21 features

## Project Files

- `train.py` — trains the full pipeline from scratch
- `test.py` — evaluates the saved model, prints accuracy/recall/confusion matrix
- `app.py` — the Streamlit app (both input modes)
- `sample_reports/` — two demo report images (high-risk and low-risk) for testing the upload feature

## Run It

```bash
pip install -r requirements.txt
streamlit run app.py
```

The trained model is already in `outputs/` — no retraining needed. To retrain from scratch: `python train.py`, then check results with `python test.py`.

> OCR requires Tesseract installed locally (see setup notes below). On Streamlit Cloud, `packages.txt` installs it automatically.

## Live Demo

https://glucoguard-ai-adlwqc3ykwoib5tzey49ql.streamlit.app/
