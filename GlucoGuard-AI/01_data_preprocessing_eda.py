"""
Day 1 - Data Loading, Cleaning & Exploratory Data Analysis
Project: Hybrid Feature Selection and Cost-Sensitive Stacked Ensemble
         Framework for Diabetes Risk Prediction
Dataset: CDC Diabetes Health Indicators (BRFSS 2015) - 253,680 rows, 21 features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------
df = pd.read_csv("/home/claude/data/diabetes_binary_health_indicators_BRFSS2015.csv")
print("Shape:", df.shape)
print("Missing values total:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())

# Drop exact duplicate rows (common in BRFSS survey data)
df = df.drop_duplicates().reset_index(drop=True)
print("Shape after removing duplicates:", df.shape)

df.to_csv("/home/claude/project/outputs/diabetes_clean.csv", index=False)

# ---------------------------------------------------------
# 2. Class distribution (justifies cost-sensitive approach)
# ---------------------------------------------------------
counts = df["Diabetes_binary"].value_counts()
pct = df["Diabetes_binary"].value_counts(normalize=True) * 100

fig, ax = plt.subplots(figsize=(6, 4.5))
bars = ax.bar(["No Diabetes", "Diabetes"], counts.values, color=["#2F8F7E", "#C94F3D"])
for bar, p in zip(bars, pct.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2000,
            f"{p:.1f}%", ha="center", fontweight="bold")
ax.set_title("Class Distribution — Diabetes Health Indicators Dataset", fontweight="bold")
ax.set_ylabel("Number of respondents")
plt.tight_layout()
plt.savefig("/home/claude/project/outputs/01_class_distribution.png")
plt.close()

# ---------------------------------------------------------
# 3. Correlation heatmap
# ---------------------------------------------------------
plt.figure(figsize=(12, 10))
corr = df.corr()
sns.heatmap(corr, cmap="RdBu_r", center=0, annot=False, linewidths=0.3)
plt.title("Feature Correlation Heatmap", fontweight="bold")
plt.tight_layout()
plt.savefig("/home/claude/project/outputs/02_correlation_heatmap.png")
plt.close()

# ---------------------------------------------------------
# 4. Correlation of each feature with target (sorted)
# ---------------------------------------------------------
target_corr = corr["Diabetes_binary"].drop("Diabetes_binary").sort_values()
plt.figure(figsize=(7, 8))
colors = ["#C94F3D" if v > 0 else "#2F8F7E" for v in target_corr.values]
plt.barh(target_corr.index, target_corr.values, color=colors)
plt.title("Feature Correlation with Diabetes Outcome", fontweight="bold")
plt.xlabel("Correlation coefficient")
plt.axvline(0, color="black", linewidth=0.8)
plt.tight_layout()
plt.savefig("/home/claude/project/outputs/03_target_correlation.png")
plt.close()

# ---------------------------------------------------------
# 5. BMI distribution by class
# ---------------------------------------------------------
plt.figure(figsize=(7, 4.5))
sns.kdeplot(data=df, x="BMI", hue="Diabetes_binary", fill=True, common_norm=False,
            palette={0: "#2F8F7E", 1: "#C94F3D"}, alpha=0.5)
plt.title("BMI Distribution by Diabetes Status", fontweight="bold")
plt.tight_layout()
plt.savefig("/home/claude/project/outputs/04_bmi_distribution.png")
plt.close()

# ---------------------------------------------------------
# 6. Age group vs diabetes rate
# ---------------------------------------------------------
age_diab = df.groupby("Age")["Diabetes_binary"].mean() * 100
plt.figure(figsize=(8, 4.5))
plt.plot(age_diab.index, age_diab.values, marker="o", color="#0E7C7B", linewidth=2)
plt.title("Diabetes Rate by Age Group (BRFSS Age Category)", fontweight="bold")
plt.xlabel("Age Category (1=18-24 ... 13=80+)")
plt.ylabel("% with Diabetes")
plt.tight_layout()
plt.savefig("/home/claude/project/outputs/05_age_vs_diabetes.png")
plt.close()

print("\nAll EDA plots saved to /home/claude/project/outputs/")
print("\nTop 5 features positively correlated with diabetes:")
print(target_corr.sort_values(ascending=False).head(5))
print("\nTop 5 features negatively correlated with diabetes:")
print(target_corr.sort_values().head(5))
