import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score
import timm
import random
import shutil



def set_seed(seed=288):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# Config
TRAIN_CSV = "train_image.csv"
VAL_CSV   = "val_image.csv"
TEST_CSV  = "test_image.csv"

BATCH_SIZE = 10
NUM_CLASSES = 3
EPOCHS = 100
LR = 2e-5
HEAD_LR_MULT = 30
WEIGHT_DECAY = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Early stopping
PATIENCE = 10
MIN_DELTA = 1e-4
EARLY_STOP_METRIC = "acc"  # 'kappa' or 'acc'

# Checkpoints folder
CKPT_DIR = "CFP_checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

# Model
MODEL_OPTIONS = {
    "efficientnet_b0": {"input_size": 224, "description": "EfficientNet-B0 (5.3M params)"},
    "efficientnet_b1": {"input_size": 240, "description": "EfficientNet-B1 (7.8M params)"},
    "efficientnet_b2": {"input_size": 260, "description": "EfficientNet-B2 (9.2M params)"},
    "efficientnet_b3": {"input_size": 300, "description": "EfficientNet-B3 (12.2M params)"},
    "efficientnet_b4": {"input_size": 380, "description": "EfficientNet-B4 (19.3M params)"},
    "tf_efficientnetv2_s": {"input_size": 300, "description": "EfficientNetV2-S (TF pretrained, 21.5M params)"},
    "tf_efficientnetv2_m": {"input_size": 336, "description": "EfficientNetV2-M (TF pretrained, 54.1M params)"},
    "tf_efficientnetv2_l": {"input_size": 480, "description": "EfficientNetV2-L (TF pretrained, 120.4M params)"},
    "resnet50": {"input_size": 224, "description": "ResNet-50 (25.6M params)"},
    "resnet101": {"input_size": 224, "description": "ResNet-101 (44.5M params)"},
    "densenet121": {"input_size": 224, "description": "DenseNet-121 (8.0M params)"},
    "vit_base_patch16_224": {"input_size": 224, "description": "Vision Transformer Base (86.6M params)"},
    "xcit_tiny_12_p8_224": {"input_size": 224, "description": "XCiT Tiny 12-layer Patch8 224x224 Input"},
    "xcit_small_24_p8_224": {"input_size": 224, "description": "XCiT Small 24-layer Patch8 224x224 Input"},
    "convnext_tiny": {"input_size": 224, "description": "ConvNeXt-Tiny (ImageNet-1K pretrained)"},
    "convnext_small": {"input_size": 224, "description": "ConvNeXt-Small (ImageNet-1K pretrained)"},
    "convnext_base": {"input_size": 224, "description": "ConvNeXt-Base (ImageNet-1K pretrained)"},
    "swin_tiny_patch4_window7_224": {"input_size": 224,
                                     "description": "Swin-Tiny Patch4 Window7 224 (ImageNet pretrained)"},
    "swin_small_patch4_window7_224": {"input_size": 224,
                                      "description": "Swin-Small Patch4 Window7 224 (ImageNet pretrained)"},
    "swin_base_patch4_window7_224": {"input_size": 224,
                                     "description": "Swin-Base Patch4 Window7 224 (ImageNet pretrained)"},
    "deit_tiny_patch16_224": {"input_size": 224, "description": "DeiT-Tiny Patch16 224 (ImageNet pretrained)"},
    "deit_small_patch16_224": {"input_size": 224, "description": "DeiT-Small Patch16 224 (ImageNet pretrained)"},
    "deit_base_patch16_224": {"input_size": 224, "description": "DeiT-Base Patch16 224 (ImageNet pretrained)"},
    "beit_base_patch16_224": {"input_size": 224,
                              "description": "BEiT-Base Patch16 224 (if pretrained weights available)"},
    "coat_tiny": {"input_size": 224, "description": "CoaT-Tiny (ImageNet pretrained if available)"},
    "coat_small": {"input_size": 224, "description": "CoaT-Small (ImageNet pretrained if available)"},
    "maxvit_tiny_rw_224": {"input_size": 224, "description": "MaxViT-Tiny RW 224 (ImageNet pretrained)"},
}
SELECTED_MODEL = "efficientnet_b3"
INPUT_SIZE = MODEL_OPTIONS[SELECTED_MODEL]["input_size"]


def evaluate_test(model, loader, criterion):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Testing"):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    return running_loss / total, correct / total, kappa


class MyDataset(Dataset):
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
        return image, label


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
    def __init__(self, model_name, num_classes=3, pretrained=True, out_indices=(2, 3, 4)):
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


def train_one_epoch(model, loader, optimizer, epoch, criterion, scheduler=None):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    all_preds, all_labels = [], []

    for imgs, labels in tqdm(loader, desc=f"Train Epoch {epoch}"):
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    if scheduler:
        scheduler.step()

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    return running_loss / total, correct / total, kappa


def validate(model, loader, criterion):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Validation"):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    return running_loss / total, correct / total, kappa


def build_layerwise_lr_optimizer(model, base_lr, head_lr_mult=10, weight_decay=0.01):
    backbone_params = set(id(p) for p in model.backbone.parameters())
    head_params = set(id(p) for p in model.classifier.parameters())

    def is_no_decay(n: str):
        n = n.lower()
        return n.endswith(".bias") or ("bn" in n) or ("norm" in n) or ("ln" in n)

    bb_decay, bb_no_decay = [], []
    hd_decay, hd_no_decay = [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        pid = id(p)
        in_backbone = pid in backbone_params
        in_head = pid in head_params

        if not (in_backbone or in_head):
            in_head = True

        group_no_decay = is_no_decay(name)

        if in_backbone:
            (bb_no_decay if group_no_decay else bb_decay).append(p)
        else:
            (hd_no_decay if group_no_decay else hd_decay).append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": bb_decay, "lr": base_lr, "weight_decay": weight_decay},
            {"params": bb_no_decay, "lr": base_lr, "weight_decay": 0.0},
            {"params": hd_decay, "lr": base_lr * head_lr_mult, "weight_decay": weight_decay},
            {"params": hd_no_decay, "lr": base_lr * head_lr_mult, "weight_decay": 0.0},
        ]
    )
    return optimizer


def train_model():
    set_seed(288)

    basic_transform = transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = MyDataset(TRAIN_CSV, transform=basic_transform)
    val_ds = MyDataset(VAL_CSV, transform=basic_transform)
    test_ds = MyDataset(TEST_CSV, transform=basic_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = AdvancedClassifier(model_name=SELECTED_MODEL, num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)

    optimizer = build_layerwise_lr_optimizer(
        model,
        base_lr=LR,
        head_lr_mult=HEAD_LR_MULT,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    best_val_metric = -float("inf")
    patience_counter = 0
    best_epoch = 0

    best_val_acc = 0
    best_val_kappa = 0
    best_test_acc = 0
    best_test_kappa = 0

    best_ckpt_path = os.path.join(CKPT_DIR, f"best_{SELECTED_MODEL}.pth")

    print(f"Selected model: {SELECTED_MODEL}")
    print(f"Description: {MODEL_OPTIONS[SELECTED_MODEL]['description']}")
    print(f"Input size: {INPUT_SIZE}x{INPUT_SIZE}")
    print(f"Device: {DEVICE}")
    print(f"Total epochs: {EPOCHS}")
    print(f"Early stop: patience={PATIENCE}, min_delta={MIN_DELTA}, metric={EARLY_STOP_METRIC}")
    print(f"Checkpoint dir: {CKPT_DIR}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}, Test samples: {len(test_ds)}")
    print(f"Layerwise LR: backbone={LR}, head={LR * HEAD_LR_MULT}")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, train_kappa = train_one_epoch(model, train_loader, optimizer, epoch, criterion, scheduler)
        val_loss, val_acc, val_kappa = validate(model, val_loader, criterion)
        test_loss, test_acc, test_kappa = evaluate_test(model, test_loader, criterion)

        current_lr0 = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch}: "
            f"Train loss/acc/kappa: {train_loss:.4f}/{train_acc:.4f}/{train_kappa:.4f}; "
            f"Val loss/acc/kappa: {val_loss:.4f}/{val_acc:.4f}/{val_kappa:.4f}; "
            f"Test loss/acc/kappa: {test_loss:.4f}/{test_acc:.4f}/{test_kappa:.4f}; "
            f"LR(group0): {current_lr0:.6f}"
        )

        current_val_metric = val_kappa if EARLY_STOP_METRIC == "kappa" else val_acc

        if current_val_metric - best_val_metric > MIN_DELTA:
            best_val_metric = current_val_metric
            best_epoch = epoch

            best_val_acc = val_acc
            best_val_kappa = val_kappa
            best_test_acc = test_acc
            best_test_kappa = test_kappa

            patience_counter = 0
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"New best checkpoint saved (epoch {best_epoch}, val_{EARLY_STOP_METRIC}={best_val_metric:.4f})")
        else:
            patience_counter += 1
            print(f"Early-stopping patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    final_best_model_path = os.path.join(CKPT_DIR, f"best_{SELECTED_MODEL}_final.pth")
    shutil.copy(best_ckpt_path, final_best_model_path)
    print(f"\nTraining completed.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val_{EARLY_STOP_METRIC}: {best_val_metric:.4f}")
    print(f"Validation acc/kappa: {best_val_acc:.4f}/{best_val_kappa:.4f}")
    print(f"Test acc/kappa (at best epoch): {best_test_acc:.4f}/{best_test_kappa:.4f}")
    print(f"Best checkpoint: {best_ckpt_path}")
    print(f"Final best model copied to: {final_best_model_path}")


if __name__ == "__main__":
    train_model()