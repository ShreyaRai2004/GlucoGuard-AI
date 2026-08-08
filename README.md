# GlucoGuard AI
### A Hybrid Feature Selection and Cost-Sensitive Stacked Ensemble Framework for Diabetes Risk Prediction

Pure Python, scikit-learn based machine learning pipeline with a Streamlit live demo.

---

## Project Structure

```
GlucoGuard-AI/
├── data/
│   └── diabetes_binary_health_indicators_BRFSS2015.csv
├── outputs/                                    # trained models, results, all report figures
├── 01_data_preprocessing_eda.py                # data cleaning + exploratory analysis
├── 02_feature_selection_baseline_models.py     # hybrid feature selection + baseline models
├── 03_cost_sensitive_stacking_ensemble.py      # proposed model + cost-weight ablation study
├── 04_explainability_and_validation.py         # SHAP + statistical significance testing
├── app.py                                      # Streamlit live demo app
├── requirements.txt
└── README.md
```

## Dataset

CDC Diabetes Health Indicators Dataset (BRFSS 2015) — 253,680 original records,
229,474 after cleaning, 21 health features, binary target (0 = no diabetes, 1 = diabetes).

## How to Run

**1. Install dependencies (exact versions matter — see note below):**
```bash
pip install -r requirements.txt
```

**2. Launch the live demo app (uses the already-trained model in `outputs/`):**
```bash
streamlit run app.py
```
This opens in your browser automatically (usually `http://localhost:8501`).

**3. (Optional) Re-run the full pipeline from scratch, in order:**
```bash
python 01_data_preprocessing_eda.py
python 02_feature_selection_baseline_models.py
python 03_cost_sensitive_stacking_ensemble.py
python 04_explainability_and_validation.py
```
Each script saves its results into `outputs/`. You only need to do this if you want
to reproduce or modify the training — everything needed for the app demo is already
included and trained.

> **Note on versions:** `requirements.txt` pins exact library versions
> (`numpy==2.2.6`, `scikit-learn==1.6.1`, etc.). This matters — the trained models in
> `outputs/` were saved with these exact versions, and loading them with different
> numpy/sklearn versions can throw pickle/deserialization errors. Install exactly as
> specified.

## Method Summary

**1. Hybrid Feature Selection**
Filter method (Mutual Information) ∩ Wrapper method (Recursive Feature Elimination
with Random Forest) → 8 final features out of 21: Age, BMI, DiffWalk, GenHlth,
HeartDiseaseorAttack, HighBP, HighChol, PhysHlth.

**2. Baseline Models** (accuracy-optimized, standard approach)
Logistic Regression, Random Forest, SVM, Gradient Boosting.

**3. Proposed Model — Cost-Sensitive Stacked Ensemble**
Base learners (Logistic Regression, Random Forest, Gradient Boosting) with
class-weighting so missing a diabetic patient is penalized more heavily than a
false alarm, combined via a Logistic Regression meta-learner.

**4. Explainability** — SHAP, both global feature importance and per-prediction
explanations (shown live in the app).

**5. Statistical Validation** — paired t-test and Wilcoxon signed-rank test on
5-fold cross-validated recall, proposed vs best baseline.

## Key Results

| Model | Accuracy | Recall | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 85.0% | 14.2% | 0.805 |
| Random Forest | 85.3% | 13.0% | 0.809 |
| SVM | 85.1% | 8.6% | 0.654 |
| Gradient Boosting (best baseline) | 85.5% | 15.7% | 0.813 |
| **Proposed Cost-Sensitive Stacked Ensemble (2:1 weight)** | **83.6%** | **41.3%** | 0.811 |

### Ablation Study: Cost-Weight Sensitivity

| Weight Ratio (Diabetic:Healthy) | Accuracy | Recall | F1-Score |
|---|---|---|---|
| 1:1 (no cost-sensitivity) | 85.4% | 19.7% | 0.292 |
| **2:1 (chosen — best balance)** | **83.6%** | **41.3%** | **0.434** |
| 3:1 | 80.0% | 56.8% | 0.465 |
| 4:1 | 76.8% | 65.8% | 0.464 |

2:1 was selected as the primary model: it keeps accuracy close to baseline levels
while more than doubling recall, and F1-score gains plateau beyond this point — 3:1
and 4:1 trade significant accuracy for comparatively small further recall gains.
(Full ablation results in `outputs/day3b_ablation_results.csv` if a use case
prioritizes recall above all else.)

## Research Gap

Existing machine learning models for diabetes risk prediction primarily optimize
overall accuracy or ROC-AUC while assuming equal misclassification costs. In
clinical practice, false negatives are significantly more critical because
undiagnosed diabetic patients may experience delayed treatment. Many existing
studies also rely on a single feature selection technique. Limited work integrates
hybrid feature selection with cost-sensitive stacked ensemble learning while also
validating improvements through explainable AI and statistical significance testing.
This project addresses these limitations.
