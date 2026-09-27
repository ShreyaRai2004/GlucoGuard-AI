import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(4.2, 9))
ax.set_xlim(0, 10)
ax.set_ylim(0, 20)
ax.axis("off")

stages = [
    "Raw Dataset\n(253,680 records, 21 features)",
    "Data Cleaning\n(duplicate removal, validation)\n→ 229,474 records",
    "Feature Scaling\n(StandardScaler)",
    "Hybrid Feature Selection\n(Mutual Information ∩ RFE)\n→ 8 features",
    "Baseline Model Training\n(LR, RF, SVM, GB)",
    "Cost-Sensitive Stacked Ensemble\n(LR + RF + GB → Meta-LR)",
    "SHAP Explainability\n+ Statistical Validation",
    "Deployment\n(Streamlit Web App:\nManual Entry / OCR Report Upload)",
]
colors = ["#e8edf3"] * 5 + ["#1f8a70", "#e8edf3", "#1a3c6e"]
text_colors = ["#1a2a3a"] * 5 + ["white", "#1a2a3a", "white"]

n = len(stages)
box_h = 1.7
gap = 0.75
total_h = n * box_h + (n - 1) * gap
y = 20 - (20 - total_h) / 2 - box_h

ys = []
for i, (stage, c, tc) in enumerate(zip(stages, colors, text_colors)):
    rect = patches.FancyBboxPatch((0.7, y), 8.6, box_h,
                                   boxstyle="round,pad=0.08,rounding_size=0.15",
                                   linewidth=1, edgecolor="#334", facecolor=c)
    ax.add_patch(rect)
    ax.text(5, y + box_h / 2, stage, ha="center", va="center", fontsize=8.3, color=tc, wrap=True)
    ys.append(y)
    y -= (box_h + gap)

for i in range(n - 1):
    y_top_of_current = ys[i]
    y_top_of_next = ys[i + 1] + box_h
    ax.annotate("", xy=(5, y_top_of_next), xytext=(5, y_top_of_current),
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.4))

plt.tight_layout()
plt.savefig("/home/claude/project/outputs/00_architecture.png", dpi=150, bbox_inches="tight")
print("Architecture diagram saved.")
