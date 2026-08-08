# GlucoGuard AI

**A machine learning system that predicts diabetes risk — built to catch more at-risk patients than standard models, and to explain every prediction it makes.**

[![Python](https://img.shields.io/badge/Python-3.10-blue)]()
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-orange)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)]()

---

## The Problem

Most machine learning models for disease prediction are optimized for one number: accuracy. That sounds right until you look closer — in a dataset where 85% of people are healthy and 15% have diabetes, a model can hit 85% accuracy by barely detecting anyone with diabetes at all. In healthcare, that's the wrong failure mode: **missing a sick patient is far more costly than a false alarm.**

GlucoGuard AI is built around that insight. Instead of chasing accuracy, it's designed to maximize recall on the at-risk class — while staying interpretable and statistically validated, not just a black box that "seems to work."

## What It Does

Given a person's health indicators (BMI, blood pressure, cholesterol, age, general health, etc.), GlucoGuard AI predicts their diabetes risk and shows **why** — which specific factors pushed the prediction up or down — through a live, interactive web app.

## Results

| Model | Accuracy | Recall (catches at-risk patients) | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 85.0% | 14.2% | 0.805 |
| Random Forest | 85.3% | 13.0% | 0.809 |
| SVM | 85.1% | 8.6% | 0.654 |
| Gradient Boosting (best baseline) | 85.5% | 15.7% | 0.813 |
| **GlucoGuard AI (proposed)** | **83.6%** | **41.3%** | 0.811 |

The proposed model **nearly triples recall** over the best standard model, at a ~2-point cost in accuracy — a deliberate, measured trade-off, statistically confirmed with a paired t-test (p < 0.001) rather than just eyeballed.

## How It Works

```
Raw survey data (253,680 records)
        |
        v
  Cleaning & EDA  ->  229,474 records, 0 missing values
        |
        v
  Hybrid Feature Selection
  (Mutual Information (intersect) Recursive Feature Elimination)  ->  8 features
        |
        v
  Cost-Sensitive Stacked Ensemble
  (Logistic Regression + Random Forest + Gradient Boosting -> meta Logistic Regression)
        |
        v
  SHAP Explainability  +  Statistical Validation (paired t-test, Wilcoxon)
        |
        v
  Streamlit Web App -- live prediction + live explanation
```

**Why this architecture, briefly:**
- **Hybrid feature selection** (not just one method) — a filter method (Mutual Information) and a wrapper method (RFE) rarely agree on everything; keeping only the features both agree on gives a cleaner, more defensible feature set than trusting either alone.
- **Stacking, not voting** — a meta-learner learns *how* to weigh each base model's opinion, rather than averaging them blindly.
- **Cost-sensitive weighting, not SMOTE** — class weighting adjusts the model's decision boundary directly, without synthesizing artificial data points that can introduce noise.
- **SHAP over generic feature importance** — SHAP explains individual predictions, not just global trends, so each user gets a personalized explanation.

## Tech Stack

- **Language:** Python 3.10
- **ML:** scikit-learn (feature selection, ensemble models, StackingClassifier)
- **Explainability:** SHAP
- **Statistics:** SciPy (paired t-test, Wilcoxon signed-rank test)
- **Data:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **App:** Streamlit
- **Dataset:** CDC Diabetes Health Indicators (BRFSS 2015) — 253,680 records, 21 features

## Project Structure

```
GlucoGuard-AI/
├── data/
│   └── diabetes_binary_health_indicators_BRFSS2015.csv
├── outputs/                                    # trained models, results, figures
├── 01_data_preprocessing_eda.py                # cleaning + exploratory analysis
├── 02_feature_selection_baseline_models.py     # hybrid feature selection + baselines
├── 03_cost_sensitive_stacking_ensemble.py      # proposed model + ablation study
├── 04_explainability_and_validation.py         # SHAP + significance testing
├── app.py                                      # Streamlit live demo
├── requirements.txt
└── README.md
```

## Running It

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. The trained model is already included in `outputs/` — no retraining needed to try the demo. To reproduce the full pipeline from scratch, run the numbered scripts in order.

> **Note:** `requirements.txt` pins exact library versions. The trained models were serialized with these versions — installing different numpy/scikit-learn versions can cause deserialization errors when loading them.

## Ablation Study

Cost-weight ratio (diabetic : healthy penalty) was tuned rather than guessed:

| Weight Ratio | Accuracy | Recall | F1-Score |
|---|---|---|---|
| 1:1 (no cost-sensitivity) | 85.4% | 19.7% | 0.292 |
| **2:1 (chosen)** | **83.6%** | **41.3%** | **0.434** |
| 3:1 | 80.0% | 56.8% | 0.465 |
| 4:1 | 76.8% | 65.8% | 0.464 |

2:1 was selected because F1-score gains plateau beyond this point — further weighting trades meaningful accuracy for comparatively small additional recall.

## Live Demo

https://glucoguard-ai-adlwqc3ykwoib5tzey49ql.streamlit.app/

---

*Built as an academic project exploring cost-sensitive learning and explainable AI in healthcare ML.*
