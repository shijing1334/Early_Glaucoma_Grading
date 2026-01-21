import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# 相对地址：checkpoints\jepa_inference_epoch1_val_predictions.csv
base_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录
csv_file_path = os.path.join(base_dir, "checkpoints", "jepa_inference_epoch1_val_predictions.csv")

df = pd.read_csv(csv_file_path)

true_labels = df["true_label"]
pred_labels = df["final_prediction"]

classes = sorted(df["true_label"].unique())
class_names = [str(c) for c in classes]

cm = confusion_matrix(true_labels, pred_labels, labels=classes)

plt.figure(figsize=(16, 13))
sns.set_theme(style="white")

ax = sns.heatmap(
    cm,
    annot=True, fmt="d",
    cmap="Blues",
    xticklabels=class_names, yticklabels=class_names,
    annot_kws={"size": 34, "weight": "bold"},
    cbar=True
)

ax.set_title("Confusion Matrix", fontsize=30, pad=18)
ax.set_xlabel("Predicted Label", fontsize=34, fontstyle="italic", labelpad=14)
ax.set_ylabel("True Label", fontsize=34, fontstyle="italic", labelpad=14)

ax.tick_params(axis="both", labelsize=30)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
plt.setp(ax.get_yticklabels(), rotation=45, va="center")
for t in ax.get_xticklabels():
    t.set_fontstyle("italic")
for t in ax.get_yticklabels():
    t.set_fontstyle("italic")

cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=28)
cbar.set_label("Count", fontsize=32, fontstyle="italic")
for t in cbar.ax.get_yticklabels():
    t.set_fontstyle("italic")

threshold = cm.max() * 0.5
for text in ax.texts:
    try:
        val = float(text.get_text())
    except ValueError:
        continue
    text.set_color("white" if val >= threshold else "black")

plt.tight_layout()

out_png = os.path.join(base_dir, "checkpoints", "jepa_confusion_matrix.png")
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.show()

print("CSV:", csv_file_path)
print("混淆矩阵:")
print(cm)
print("\n分类报告:")
print(classification_report(true_labels, pred_labels, target_names=class_names))