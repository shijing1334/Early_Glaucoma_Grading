import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import timm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report


# =========================
# Config（相对路径：你自己填）
# =========================
VAL_CSV = r"val_image.csv"
TEST_CSV = r"test_image.csv"
MODEL_CKPT = r"CFP_checkpoints/best_efficientnet_b3_seed_288.pth"

OUT_DIR = r"checkpoints"  # 输出目录（相对本脚本）
OUT_VAL_PRED_CSV = r"val_predictions.csv"
OUT_TEST_PRED_CSV = r"test_predictions.csv"
OUT_VAL_CM_PNG = r"val_confusion_matrix.png"
OUT_TEST_CM_PNG = r"test_confusion_matrix.png"

BATCH_SIZE = 10
NUM_CLASSES = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SELECTED_MODEL = "efficientnet_b3"
MODEL_OPTIONS = {"efficientnet_b3": {"input_size": 300}}
INPUT_SIZE = MODEL_OPTIONS[SELECTED_MODEL]["input_size"]


# 以“本脚本所在目录”为基准拼绝对路径，避免从不同工作目录运行出错
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VAL_CSV = os.path.join(BASE_DIR, VAL_CSV)
TEST_CSV = os.path.join(BASE_DIR, TEST_CSV)
MODEL_CKPT = os.path.join(BASE_DIR, MODEL_CKPT)

OUT_DIR = os.path.join(BASE_DIR, OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

OUT_VAL_PRED_CSV = os.path.join(OUT_DIR, OUT_VAL_PRED_CSV)
OUT_TEST_PRED_CSV = os.path.join(OUT_DIR, OUT_TEST_PRED_CSV)
OUT_VAL_CM_PNG = os.path.join(OUT_DIR, OUT_VAL_CM_PNG)
OUT_TEST_CM_PNG = os.path.join(OUT_DIR, OUT_TEST_CM_PNG)


def set_seed(seed=288):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, csvfile, transform=None):
        self.df = pd.read_csv(csvfile)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["path"]
        label = int(row["class"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, img_path


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, hidden_dim, p=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(p)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(p)

    def forward(self, x):
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop1(h)
        h = self.fc2(h)
        h = self.drop2(h)
        return x + h


class AdvancedClassifier(nn.Module):
    def __init__(self, model_name, num_classes=3, pretrained=False, out_indices=(2, 3, 4)):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )
        feat_channels = self.backbone.feature_info.channels()
        fusion_dim = sum(feat_channels)

        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.4),
            ResidualMLPBlock(1024, 2048, p=0.2),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            ResidualMLPBlock(512, 1024, p=0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        feats = self.backbone(x)
        pooled = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in feats]
        fused = torch.cat(pooled, dim=1)
        return self.classifier(fused)


@torch.no_grad()
def infer_to_df(model, loader, split_name: str):
    model.eval()
    rows = []
    all_true, all_pred = [], []

    for imgs, labels, paths in tqdm(loader, desc=f"Infer {split_name}"):
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(imgs)
        preds = logits.argmax(dim=1)

        all_true.append(labels.cpu().numpy())
        all_pred.append(preds.cpu().numpy())

        for i in range(len(paths)):
            rows.append({
                "split": split_name,
                "path": paths[i],
                "true_label": int(labels[i].cpu().item()),
                "final_prediction": int(preds[i].cpu().item()),
            })

    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)

    acc = (all_true == all_pred).mean()
    kappa = cohen_kappa_score(all_true, all_pred, weights="quadratic")
    print(f"{split_name}: Samples={len(rows)} | Acc={acc*100:.3f}% | Kappa={kappa:.4f}")

    return pd.DataFrame(rows), all_true, all_pred


def plot_and_save_cm(true_labels, pred_labels, out_png, title="Confusion Matrix"):
    classes = sorted(np.unique(np.concatenate([true_labels, pred_labels])))
    class_names = [str(c) for c in classes]

    cm = confusion_matrix(true_labels, pred_labels, labels=classes)

    plt.figure(figsize=(16, 13))
    sns.set_theme(style="white")

    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 34, "weight": "bold"},
        cbar=True,
    )

    ax.set_title(title, fontsize=30, pad=18)
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

    threshold = cm.max() * 0.5 if cm.size else 0
    for text in ax.texts:
        try:
            val = float(text.get_text())
        except ValueError:
            continue
        text.set_color("white" if val >= threshold else "black")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved CM PNG:", out_png)
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(true_labels, pred_labels, target_names=class_names))
    return cm


def main():
    set_seed(288)

    if not os.path.exists(MODEL_CKPT):
        raise FileNotFoundError(f"Checkpoint not found: {MODEL_CKPT}")
    if not os.path.exists(VAL_CSV):
        raise FileNotFoundError(f"VAL_CSV not found: {VAL_CSV}")
    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(f"TEST_CSV not found: {TEST_CSV}")

    tfm = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_loader = DataLoader(MyDataset(VAL_CSV, transform=tfm), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(MyDataset(TEST_CSV, transform=tfm), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = AdvancedClassifier(model_name=SELECTED_MODEL, num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
    state = torch.load(MODEL_CKPT, map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    print(f"Loaded checkpoint: {MODEL_CKPT}")

    # 1) 推理并保存 CSV
    df_val, yv_true, yv_pred = infer_to_df(model, val_loader, "val")
    df_test, yt_true, yt_pred = infer_to_df(model, test_loader, "test")

    df_val.to_csv(OUT_VAL_PRED_CSV, index=False, encoding="utf-8-sig")
    df_test.to_csv(OUT_TEST_PRED_CSV, index=False, encoding="utf-8-sig")
    print("Saved:", OUT_VAL_PRED_CSV)
    print("Saved:", OUT_TEST_PRED_CSV)

    # 2) 直接生成混淆矩阵图
    plot_and_save_cm(yv_true, yv_pred, OUT_VAL_CM_PNG, title="Confusion Matrix")
    plot_and_save_cm(yt_true, yt_pred, OUT_TEST_CM_PNG, title="Confusion Matrix")


if __name__ == "__main__":
    main()