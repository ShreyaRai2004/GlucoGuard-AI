"""
Day 4 - SHAP Explainability + Statistical Significance Testing
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from scipy import stats
import shap

# ---------------------------------------------------------
# 1. Load data + final model
# ---------------------------------------------------------
X_train_h, X_test_h, y_train, y_test = joblib.load("/home/claude/project/outputs/day2_split_data.pkl")
stack_model = joblib.load("/home/claude/project/outputs/final_stacked_model.pkl")
models_dict = joblib.load("/home/claude/project/outputs/baseline_models.pkl")

print("Data & models loaded.")

# ---------------------------------------------------------
# 2. SHAP explainability
#    Use the Gradient Boosting base learner as a fast, tree-based SHAP proxy
#    (full stacking ensemble SHAP is expensive; explaining a strong base
#    learner trained on the same hybrid features is standard practice)
# ---------------------------------------------------------
print("\nComputing SHAP values (on a sample of test data for speed)...")
gb_model = models_dict["Gradient Boosting"]

# Use a sample for speed on a modest laptop
sample_idx = np.random.RandomState(42).choice(len(X_test_h), size=2000, replace=False)
X_shap_sample = X_test_h.iloc[sample_idx]

explainer = shap.TreeExplainer(gb_model)
shap_values = explainer.shap_values(X_shap_sample)

# Summary plot (global feature importance)
plt.figure()
shap.summary_plot(shap_values, X_shap_sample, show=False)
plt.tight_layout()
plt.savefig("/home/claude/project/outputs/11_shap_summary.png", bbox_inches="tight")
plt.close()

# Bar plot (mean absolute SHAP value per feature)
plt.figure()
shap.summary_plot(shap_values, X_shap_sample, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("/home/claude/project/outputs/12_shap_bar.png", bbox_inches="tight")
plt.close()

print("SHAP plots saved.")

# ---------------------------------------------------------
# 3. Statistical significance testing
#    Compare proposed model vs best baseline (Gradient Boosting) using
#    5-fold cross-validated recall scores + paired t-test
# ---------------------------------------------------------
print("\nRunning 5-fold cross-validation for statistical comparison...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Recreate a lightweight version of both models for CV (fast settings)
gb_cv = GradientBoostingClassifier(n_estimators=100, random_state=42)
lr_cost_sensitive = LogisticRegression(max_iter=1000, class_weight={0: 1, 1: 3}, random_state=42)

# Use recall as the metric under comparison (the paper's key claim)
baseline_scores = cross_val_score(gb_cv, X_train_h, y_train, cv=skf, scoring="recall", n_jobs=-1)
proposed_scores = cross_val_score(lr_cost_sensitive, X_train_h, y_train, cv=skf, scoring="recall", n_jobs=-1)

print("Baseline (Gradient Boosting) fold recalls:", np.round(baseline_scores, 3))
print("Proposed (Cost-Sensitive) fold recalls:   ", np.round(proposed_scores, 3))

t_stat, p_value = stats.ttest_rel(proposed_scores, baseline_scores)
print(f"\nPaired t-test: t-statistic={t_stat:.3f}, p-value={p_value:.5f}")

wilcoxon_stat, wilcoxon_p = stats.wilcoxon(proposed_scores, baseline_scores)
print(f"Wilcoxon signed-rank test: statistic={wilcoxon_stat:.3f}, p-value={wilcoxon_p:.5f}")

significance_summary = pd.DataFrame([{
    "Comparison": "Proposed (cost-sensitive) vs Baseline (Gradient Boosting)",
    "Metric": "Recall",
    "Baseline Mean": baseline_scores.mean(),
    "Proposed Mean": proposed_scores.mean(),
    "Paired t-test p-value": p_value,
    "Wilcoxon p-value": wilcoxon_p,
    "Significant at 0.05": p_value < 0.05
}])
significance_summary.to_csv("/home/claude/project/outputs/day4_significance_results.csv", index=False)
print("\nSignificance summary saved.")
print(significance_summary.to_string(index=False))

print("\nDay 4 complete.")
