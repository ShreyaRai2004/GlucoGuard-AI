"""
test.py — Evaluate the trained GlucoGuard AI model on the held-out test set.
Run this after train.py to verify results without retraining.

Usage:
    python test.py
"""

import joblib
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix, classification_report)

def main():
    print("Loading trained model and test data...")
    model = joblib.load("outputs/final_stacked_model.pkl")
    X_train, X_test, y_train, y_test = joblib.load("outputs/split_data.pkl")

    print(f"Test set size: {len(X_test)} records\n")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print("=== GlucoGuard AI — Proposed Model Results ===")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC-AUC   : {auc:.4f}")

    print("\n=== Confusion Matrix ===")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                 Predicted No    Predicted Yes")
    print(f"Actual No        {cm[0][0]:<15} {cm[0][1]}")
    print(f"Actual Yes       {cm[1][0]:<15} {cm[1][1]}")

    print("\n=== Full Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))

    print(f"Diabetic patients correctly identified: {cm[1][1]} out of {cm[1][0]+cm[1][1]} "
          f"({rec*100:.1f}% recall)")

if __name__ == "__main__":
    main()
