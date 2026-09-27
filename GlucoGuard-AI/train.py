"""
train.py — GlucoGuard AI full training pipeline

Runs, in order: data cleaning -> hybrid feature selection -> baseline models
-> cost-sensitive stacked ensemble -> saves all artifacts to outputs/.

Usage:
    python train.py
"""

import pandas as pd
import numpy as np
import joblib
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score)

sns.set_style("whitegrid")
os.makedirs("outputs", exist_ok=True)

DATA_PATH = "data/diabetes_binary_health_indicators_BRFSS2015.csv"
COST_WEIGHT = {0: 1, 1: 2}  # diabetic:healthy penalty ratio -- chosen via ablation study


def load_and_clean_data():
    print("[1/5] Loading and cleaning data...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Raw shape: {df.shape}")
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"  After removing duplicates: {df.shape}")
    df.to_csv("outputs/diabetes_clean.csv", index=False)
    return df


def hybrid_feature_selection(X_train_scaled, y_train, feature_names):
    print("[2/5] Running hybrid feature selection (Mutual Information + RFE)...")

    mi_scores = mutual_info_classif(X_train_scaled, y_train, random_state=42)
    mi_series = pd.Series(mi_scores, index=feature_names).sort_values(ascending=False)
    top_mi = set(mi_series.head(12).index)

    X_rfe_sub, _, y_rfe_sub, _ = train_test_split(
        X_train_scaled, y_train, train_size=30000, random_state=42, stratify=y_train
    )
    rfe_estimator = RandomForestClassifier(n_estimators=40, max_depth=10, random_state=42, n_jobs=-1)
    rfe = RFE(estimator=rfe_estimator, n_features_to_select=12, step=3)
    rfe.fit(X_rfe_sub, y_rfe_sub)
    top_rfe = set(np.array(feature_names)[rfe.support_])

    hybrid = sorted(top_mi & top_rfe)
    print(f"  Selected {len(hybrid)} features: {hybrid}")
    return hybrid


def train_baselines(X_train, y_train, X_test, y_test):
    print("[3/5] Training baseline models...")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        "SVM": SVC(probability=False, random_state=42, kernel="rbf"),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
    }
    X_train_svm, _, y_train_svm, _ = train_test_split(
        X_train, y_train, train_size=10000, random_state=42, stratify=y_train
    )

    results = []
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train_svm if name == "SVM" else X_train, y_train_svm if name == "SVM" else y_train)
        y_pred = model.predict(X_test)
        acc, rec = accuracy_score(y_test, y_pred), recall_score(y_test, y_pred)
        print(f"  {name}: accuracy={acc:.3f} recall={rec:.3f} ({time.time()-t0:.1f}s)")
        results.append({"Model": name, "Accuracy": acc, "Recall": rec})

    joblib.dump(models, "outputs/baseline_models.pkl")
    joblib.dump(models["Gradient Boosting"], "outputs/gb_model_only.pkl")
    pd.DataFrame(results).to_csv("outputs/baseline_results.csv", index=False)
    return models


def train_proposed_model(X_train, y_train, X_test, y_test):
    print("[4/5] Training cost-sensitive stacked ensemble (proposed model)...")
    base_learners = [
        ("lr", LogisticRegression(max_iter=1000, class_weight=COST_WEIGHT, random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=100, max_depth=12, class_weight=COST_WEIGHT, random_state=42, n_jobs=-1)),
        ("gb", GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)),
    ]
    meta_learner = LogisticRegression(max_iter=1000, class_weight=COST_WEIGHT, random_state=42)
    model = StackingClassifier(estimators=base_learners, final_estimator=meta_learner, cv=3, n_jobs=-1)

    t0 = time.time()
    model.fit(X_train, y_train)
    print(f"  Trained in {time.time()-t0:.1f}s")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(f"  Accuracy={accuracy_score(y_test,y_pred):.3f} "
          f"Precision={precision_score(y_test,y_pred):.3f} "
          f"Recall={recall_score(y_test,y_pred):.3f} "
          f"F1={f1_score(y_test,y_pred):.3f} "
          f"ROC-AUC={roc_auc_score(y_test,y_proba):.3f}")

    joblib.dump(model, "outputs/final_stacked_model.pkl")
    return model


def main():
    df = load_and_clean_data()
    X = df.drop("Diabetes_binary", axis=1)
    y = df["Diabetes_binary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, "outputs/scaler.pkl")

    hybrid_features = hybrid_feature_selection(X_train_scaled, y_train, X.columns)
    joblib.dump(hybrid_features, "outputs/hybrid_features.pkl")

    X_train_h = pd.DataFrame(X_train_scaled, columns=X.columns)[hybrid_features]
    X_test_h = pd.DataFrame(X_test_scaled, columns=X.columns)[hybrid_features]
    joblib.dump((X_train_h, X_test_h, y_train, y_test), "outputs/split_data.pkl")

    train_baselines(X_train_h, y_train, X_test_h, y_test)
    train_proposed_model(X_train_h, y_train, X_test_h, y_test)

    print("[5/5] Done. All artifacts saved to outputs/.")


if __name__ == "__main__":
    main()
