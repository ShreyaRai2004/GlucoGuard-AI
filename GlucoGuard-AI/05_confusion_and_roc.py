"""
Generates Fig. 4 (confusion matrices: best baseline vs proposed) and
Fig. 5 (ROC curves for all evaluated models), matching the paper.
"""
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

X_train_h, X_test_h, y_train, y_test = joblib.load("/home/claude/project/outputs/day2_split_data.pkl")
baseline_models = joblib.load("/home/claude/project/outputs/baseline_models.pkl")
proposed_model = joblib.load("/home/claude/project/outputs/final_stacked_model_2to1.pkl")

gb_model = baseline_models["Gradient Boosting"]

# ---------------------------------------------------------
# Fig 4: Confusion matrices
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 4))

cm_gb = confusion_matrix(y_test, gb_model.predict(X_test_h))
sns.heatmap(cm_gb, annot=True, fmt="d", cmap="Reds", ax=axes[0],
            xticklabels=["No Diabetes", "Diabetes"], yticklabels=["No Diabetes", "Diabetes"])
axes[0].set_title("Best Baseline (Gradient Boosting)")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

cm_prop = confusion_matrix(y_test, proposed_model.predict(X_test_h))
sns.heatmap(cm_prop, annot=True, fmt="d", cmap="Greens", ax=axes[1],
            xticklabels=["No Diabetes", "Diabetes"], yticklabels=["No Diabetes", "Diabetes"])
axes[1].set_title("Proposed Cost-Sensitive Ensemble")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.savefig("/home/claude/project/outputs/09_confusion_matrices.png")
plt.close()
print("Confusion matrices saved.")
print("Baseline CM:\n", cm_gb)
print("Proposed CM:\n", cm_prop)

# ---------------------------------------------------------
# Fig 5: ROC curves for all models
# ---------------------------------------------------------
plt.figure(figsize=(6, 5))

for name, model in baseline_models.items():
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test_h)[:, 1]
    else:
        probs = model.decision_function(X_test_h)
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})", linewidth=1.3)

probs_prop = proposed_model.predict_proba(X_test_h)[:, 1]
fpr_p, tpr_p, _ = roc_curve(y_test, probs_prop)
auc_p = auc(fpr_p, tpr_p)
plt.plot(fpr_p, tpr_p, label=f"Proposed Ensemble (AUC={auc_p:.3f})", linewidth=2.2, color="black")

plt.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=0.8)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(fontsize=8, loc="lower right")
plt.tight_layout()
plt.savefig("/home/claude/project/outputs/10_roc_curves.png")
plt.close()
print("ROC curves saved. Proposed AUC:", auc_p)
