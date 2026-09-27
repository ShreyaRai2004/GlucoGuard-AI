"""
Day 2 - Hybrid Feature Selection + Baseline Models
"""

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

# ---------------------------------------------------------
# 1. Load cleaned data
# ---------------------------------------------------------
df = pd.read_csv("/home/claude/project/outputs/diabetes_clean.csv")
X = df.drop("Diabetes_binary", axis=1)
y = df["Diabetes_binary"]

print("Features:", list(X.columns))
print("X shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 2. Hybrid Feature Selection
#    Filter: Mutual Information
#    Wrapper: RFE with Random Forest
# ---------------------------------------------------------
print("\nRunning Mutual Information (filter method)...")
mi_scores = mutual_info_classif(X_train_scaled, y_train, random_state=42)
mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
top_mi_features = set(mi_series.head(12).index)
print("Top 12 features by Mutual Information:")
print(mi_series.head(12))

print("\nRunning RFE with Random Forest (wrapper method)... this takes a minute")
# Use a stratified subsample for RFE (standard practice — RFE refits repeatedly,
# full 183k rows x multiple refits is unnecessarily slow for feature ranking)
X_rfe_sub, _, y_rfe_sub, _ = train_test_split(
    X_train_scaled, y_train, train_size=30000, random_state=42, stratify=y_train
)
rfe_estimator = RandomForestClassifier(n_estimators=40, max_depth=10, random_state=42, n_jobs=-1)
rfe = RFE(estimator=rfe_estimator, n_features_to_select=12, step=3)
rfe.fit(X_rfe_sub, y_rfe_sub)
top_rfe_features = set(X.columns[rfe.support_])
print("Top 12 features by RFE:")
print(sorted(top_rfe_features))

# Hybrid = intersection of both methods
hybrid_features = sorted(top_mi_features & top_rfe_features)
print(f"\nHybrid feature selection (intersection): {len(hybrid_features)} features")
print(hybrid_features)

# Save the feature comparison plot
fig, ax = plt.subplots(figsize=(8, 5))
categories = ["Mutual Info only", "RFE only", "Both (Hybrid)"]
mi_only = len(top_mi_features - top_rfe_features)
rfe_only = len(top_rfe_features - top_mi_features)
both = len(hybrid_features)
ax.bar(categories, [mi_only, rfe_only, both], color=["#8FA6A2", "#8FA6A2", "#0E7C7B"])
ax.set_title("Hybrid Feature Selection: Filter vs Wrapper Agreement", fontweight="bold")
ax.set_ylabel("Number of features")
plt.tight_layout()
plt.savefig("/home/claude/project/outputs/06_feature_selection_overlap.png")
plt.close()

# Use hybrid features for all downstream models
X_train_h = pd.DataFrame(X_train_scaled, columns=X.columns)[hybrid_features]
X_test_h = pd.DataFrame(X_test_scaled, columns=X.columns)[hybrid_features]

# ---------------------------------------------------------
# 3. Baseline Models (plain, accuracy-driven, no cost-sensitivity)
# ---------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
    "SVM": SVC(probability=False, random_state=42, kernel="rbf"),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=3),
}

# SVM has O(n^2)-O(n^3) training complexity — infeasible on 183k rows in reasonable
# time. Train it on a stratified subsample (documented, standard practice for SVM
# on large datasets), keep full data for the other three models.
X_train_svm, _, y_train_svm, _ = train_test_split(
    X_train_h, y_train, train_size=10000, random_state=42, stratify=y_train
)

results = []
for name, model in models.items():
    t0 = time.time()
    if name == "SVM":
        model.fit(X_train_svm, y_train_svm)
    else:
        model.fit(X_train_h, y_train)
    train_time = time.time() - t0

    y_pred = model.predict(X_test_h)
    if name == "SVM":
        y_proba = model.decision_function(X_test_h)  # for AUC only, not a probability
    else:
        y_proba = model.predict_proba(X_test_h)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results.append({
        "Model": name, "Accuracy": acc, "Precision": prec,
        "Recall": rec, "F1": f1, "ROC-AUC": auc, "Train Time (s)": train_time
    })
    print(f"\n{name} (trained in {train_time:.1f}s):")
    print(f"  Accuracy={acc:.3f}  Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}  ROC-AUC={auc:.3f}")

results_df = pd.DataFrame(results)
results_df.to_csv("/home/claude/project/outputs/day2_baseline_results.csv", index=False)

# ---------------------------------------------------------
# 4. Visualize: Accuracy vs Recall gap (THE key argument of your paper)
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(results_df))
width = 0.35
ax.bar(x - width/2, results_df["Accuracy"], width, label="Accuracy", color="#8FA6A2")
ax.bar(x + width/2, results_df["Recall"], width, label="Recall (catching diabetics)", color="#C94F3D")
ax.set_xticks(x)
ax.set_xticklabels(results_df["Model"], rotation=15)
ax.set_ylabel("Score")
ax.set_title("Baseline Models: Accuracy vs Recall Gap", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("/home/claude/project/outputs/07_accuracy_vs_recall_baseline.png")
plt.close()

# Save trained baseline models + scaler + feature list for later days
import joblib
joblib.dump(models, "/home/claude/project/outputs/baseline_models.pkl")
joblib.dump(scaler, "/home/claude/project/outputs/scaler.pkl")
joblib.dump(hybrid_features, "/home/claude/project/outputs/hybrid_features.pkl")
joblib.dump((X_train_h, X_test_h, y_train, y_test), "/home/claude/project/outputs/day2_split_data.pkl")

print("\nDay 2 complete. Results saved.")
print(results_df)
