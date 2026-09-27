"""
Day 3b - Ablation Study: Cost-Weight Sensitivity Analysis
Tests multiple class_weight ratios to find the best accuracy/recall trade-off.
"""

import pandas as pd
import numpy as np
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

X_train_h, X_test_h, y_train, y_test = joblib.load("/home/claude/project/outputs/day2_split_data.pkl")

weight_settings = [1, 2, 3, 4]
ablation_results = []

for w in weight_settings:
    print(f"\nTraining stacking ensemble with cost-weight ratio {w}:1 ...")
    COST_WEIGHT = {0: 1, 1: w}

    base_learners = [
        ("lr", LogisticRegression(max_iter=1000, class_weight=COST_WEIGHT, random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=100, max_depth=12, class_weight=COST_WEIGHT, random_state=42, n_jobs=-1)),
        ("gb", GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)),
    ]
    meta_learner = LogisticRegression(max_iter=1000, class_weight=COST_WEIGHT, random_state=42)

    t0 = time.time()
    model = StackingClassifier(estimators=base_learners, final_estimator=meta_learner, cv=3, n_jobs=-1)
    model.fit(X_train_h, y_train)
    train_time = time.time() - t0

    y_pred = model.predict(X_test_h)
    y_proba = model.predict_proba(X_test_h)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"  Weight {w}:1 -> Accuracy={acc:.3f} Precision={prec:.3f} Recall={rec:.3f} F1={f1:.3f} ROC-AUC={auc:.3f} ({train_time:.1f}s)")

    ablation_results.append({
        "Cost Weight (Diabetic:Healthy)": f"{w}:1",
        "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "ROC-AUC": auc
    })

    # Save the 2:1 model as the new "recommended" primary model
    if w == 2:
        joblib.dump(model, "/home/claude/project/outputs/final_stacked_model_2to1.pkl")

ablation_df = pd.DataFrame(ablation_results)
ablation_df.to_csv("/home/claude/project/outputs/day3b_ablation_results.csv", index=False)
print("\nAblation study results:")
print(ablation_df.to_string(index=False))

# ---------------------------------------------------------
# Plot: Accuracy/Recall trade-off curve across weight settings
# ---------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(8, 5))
x = np.arange(len(weight_settings))

ax1.plot(x, ablation_df["Accuracy"], marker="o", color="#8FA6A2", linewidth=2, label="Accuracy")
ax1.plot(x, ablation_df["Recall"], marker="o", color="#C94F3D", linewidth=2, label="Recall")
ax1.plot(x, ablation_df["F1"], marker="o", color="#0E7C7B", linewidth=2, linestyle="--", label="F1-Score")
ax1.set_xticks(x)
ax1.set_xticklabels([f"{w}:1" for w in weight_settings])
ax1.set_xlabel("Cost Weight Ratio (Diabetic : Healthy)")
ax1.set_ylabel("Score")
ax1.set_title("Ablation Study: Effect of Cost-Weight Ratio\non Accuracy vs Recall Trade-off", fontweight="bold")
ax1.axvline(x[1], color="gray", linestyle=":", alpha=0.6)
ax1.text(x[1], 0.95, "  Recommended (2:1)", fontsize=9, color="gray")
ax1.legend()
plt.tight_layout()
plt.savefig("/home/claude/project/outputs/13_ablation_tradeoff.png")
plt.close()

print("\nAblation study complete. Plot saved.")
