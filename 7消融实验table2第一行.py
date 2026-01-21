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

import cv2
from PIL import ImageEnhance, ImageFilter, ImageDraw

def pil_to_cv(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def random_rotate(img):
    angle = random.choice([90, 180, 270, 15, -15, 30, -30])
    return img.rotate(angle, expand=False), "rotate"

def horizontal_flip(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT), "hflip"

def vertical_flip(img):
    return img.transpose(Image.FLIP_TOP_BOTTOM), "vflip"

def adjust_brightness(img):
    factor = random.uniform(0.7, 1.3)
    return ImageEnhance.Brightness(img).enhance(factor), "brightness"

def adjust_contrast(img):
    factor = random.uniform(0.7, 1.3)
    return ImageEnhance.Contrast(img).enhance(factor), "contrast"

def adjust_saturation(img):
    factor = random.uniform(0.8, 1.2)
    return ImageEnhance.Color(img).enhance(factor), "saturation"

def gaussian_blur(img):
    radius = random.uniform(0.5, 1.5)
    return img.filter(ImageFilter.GaussianBlur(radius=radius)), "blur"

def sharpen(img):
    return img.filter(ImageFilter.SHARPEN), "sharpen"

def emboss(img):
    return img.filter(ImageFilter.EMBOSS), "emboss"

def edge_enhance(img):
    return img.filter(ImageFilter.EDGE_ENHANCE), "edge_enhance"

def random_crop_resize(img):
    w, h = img.size
    crop_rate = random.uniform(0.85, 0.98)
    new_w, new_h = int(w * crop_rate), int(h * crop_rate)
    left = random.randint(0, w - new_w)
    top = random.randint(0, h - new_h)
    img_crop = img.crop((left, top, left + new_w, top + new_h))
    return img_crop.resize((w, h), Image.LANCZOS), "crop"

def add_gaussian_noise(img):
    img_array = np.asarray(img).astype(np.float32)
    noise = np.random.normal(0, 8, img_array.shape)
    noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img), "gauss_noise"

def add_salt_pepper_noise(img, amount=0.004, s_vs_p=0.5):
    img_array = np.array(img)
    out = img_array.copy()
    num_salt = np.ceil(amount * img_array.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img_array.shape[:2]]
    out[coords[0], coords[1]] = 255
    num_pepper = np.ceil(amount * img_array.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img_array.shape[:2]]
    out[coords[0], coords[1]] = 0
    return Image.fromarray(out), "s&p_noise"

def add_speckle_noise(img):
    img_array = np.array(img).astype(np.float32)
    noise = np.random.randn(*img_array.shape)
    noisy_img = img_array + img_array * noise * 0.12
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img), "speckle_noise"

def add_poisson_noise(img):
    img_array = np.array(img).astype(np.float32)
    vals = len(np.unique(img_array))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(img_array * vals) / float(vals)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy), "poisson_noise"

def add_multiplicative_gaussian_noise(img):
    img_array = np.array(img).astype(np.float32)
    noise = np.random.normal(loc=1.0, scale=0.13, size=img_array.shape)
    noisy_img = img_array * noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img), "multi_gauss_noise"

def bilateral_noise(img):
    cvimg = pil_to_cv(img)
    diameter = random.choice([7, 9, 11, 13])
    sigColor = random.randint(50, 150)
    sigSpace = random.randint(30, 80)
    cv_bilat = cv2.bilateralFilter(cvimg, diameter, sigColor, sigSpace)
    post_process = random.choice(["gauss", "sp"])
    pil_bilat = cv_to_pil(cv_bilat)
    if post_process == "gauss":
        pil_bilat, _ = add_gaussian_noise(pil_bilat)
        return pil_bilat, "bilateral_gauss"
    else:
        pil_bilat, _ = add_salt_pepper_noise(pil_bilat)
        return pil_bilat, "bilateral_sp"

def random_scale(img):
    w, h = img.size
    scale_factor = random.uniform(0.9, 1.1)
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    scaled_img = img.resize((new_w, new_h), Image.LANCZOS)
    if scale_factor > 1:
        left = (new_w - w) // 2
        top = (new_h - h) // 2
        scaled_img = scaled_img.crop((left, top, left + w, top + h))
    else:
        new_img = Image.new("RGB", (w, h), (0, 0, 0))
        paste_x = (w - new_w) // 2
        paste_y = (h - new_h) // 2
        new_img.paste(scaled_img, (paste_x, paste_y))
        scaled_img = new_img
    return scaled_img, "scale"

def adjust_gamma(img):
    gamma = random.uniform(0.7, 1.3)
    img_array = np.asarray(img).astype(np.float32) / 255.0
    img_array = np.power(img_array, gamma)
    img_array = (img_array * 255).astype(np.uint8)
    return Image.fromarray(img_array), "gamma"

def elastic_transform(img):
    try:
        from scipy.ndimage import gaussian_filter, map_coordinates
        img_array = np.asarray(img)
        h, w = img_array.shape[:2]
        dx = np.random.randn(h, w) * 2
        dy = np.random.randn(h, w) * 2
        dx = gaussian_filter(dx, sigma=3)
        dy = gaussian_filter(dy, sigma=3)
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        if len(img_array.shape) == 3:
            transformed = np.zeros_like(img_array)
            for i in range(img_array.shape[2]):
                transformed[:, :, i] = map_coordinates(
                    img_array[:, :, i], indices, order=1, mode="reflect"
                ).reshape(h, w)
        else:
            transformed = map_coordinates(img_array, indices, order=1, mode="reflect").reshape(h, w)
        return Image.fromarray(transformed.astype(np.uint8)), "elastic"
    except ImportError:
        return img, "elastic_skip"

def affine_transform(img):
    w, h = img.size
    angle = random.uniform(-20, 20)
    tx = random.uniform(-0.1, 0.1) * w
    ty = random.uniform(-0.1, 0.1) * h
    scale = random.uniform(0.9, 1.1)
    mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    mat[0, 2] += tx
    mat[1, 2] += ty
    cvimg = pil_to_cv(img)
    cvimg2 = cv2.warpAffine(cvimg, mat, (w, h), borderMode=cv2.BORDER_REFLECT)
    return cv_to_pil(cvimg2), "affine"

def add_red_curves(img):
    w, h = img.size
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    red_palette = [
        (139, 0, 0), (178, 34, 34), (220, 20, 60), (255, 0, 0),
        (255, 69, 0), (255, 99, 71), (255, 127, 80), (255, 160, 122)
    ]
    num_curves = random.randint(2, 5)
    curve_width = random.randint(10, 30)
    for _ in range(num_curves):
        curve_color = random.choice(red_palette)
        points = [(0, random.randint(0, h))]
        for _ in range(3):
            x = random.randint(0, w)
            y = random.randint(0, h)
            points.append((x, y))
        points.append((w, random.randint(0, h)))
        draw.line(points, fill=curve_color, width=curve_width)
    return img_copy, f"red_curves_{num_curves}"

def add_red_strip(img):
    w, h = img.size
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    is_horizontal = random.choice([True, False])
    strip_w = random.randint(10, 30)
    if is_horizontal:
        y = random.randint(0, h - strip_w)
        draw.rectangle([(0, y), (w, y + strip_w)], fill=(255, 0, 0))
        suffix = f"redstrip_h_{strip_w}px"
    else:
        x = random.randint(0, w - strip_w)
        draw.rectangle([(x, 0), (x + strip_w, h)], fill=(255, 0, 0))
        suffix = f"redstrip_v_{strip_w}px"
    return img_copy, suffix

def add_white_square(img):
    w, h = img.size
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    max_len = min(w, h) // 2
    sq_len = random.randint(int(max_len * 0.2), max(12, int(max_len)))
    x1 = random.randint(0, w - sq_len)
    y1 = random.randint(0, h - sq_len)
    draw.rectangle([x1, y1, x1 + sq_len, y1 + sq_len], fill=(255, 255, 255))
    suffix = f"whitesq_{sq_len}px"
    return img_copy, suffix

augmentation_methods = [
    random_rotate, horizontal_flip, vertical_flip, adjust_brightness,
    adjust_contrast, adjust_saturation, gaussian_blur, sharpen, emboss,
    edge_enhance, random_crop_resize, add_gaussian_noise, add_salt_pepper_noise,
    add_speckle_noise, add_poisson_noise, add_multiplicative_gaussian_noise,
    bilateral_noise, random_scale, adjust_gamma, elastic_transform,
    affine_transform, add_red_strip, add_white_square, add_red_curves,
]

AUG_EXEC_CONFIG = {
    "exec_prob": [0.7, 0.8, 0.9],
    "min_exec_num": 0,
    "max_exec_num": 2,
}

def apply_augmentation(img, methods, num_methods):
    selected_methods = random.sample(methods, min(num_methods, len(methods)))
    augmented_img = img.copy()
    method_names = []
    exec_prob = random.choice(AUG_EXEC_CONFIG["exec_prob"])

    for method in selected_methods:
        if random.random() < exec_prob:
            augmented_img, method_name = method(augmented_img)
            method_names.append(method_name)
        else:
            method_names.append(f"{method.__name__}_skipped")

    actual_exec_num = len([n for n in method_names if not n.endswith("_skipped")])
    if actual_exec_num < AUG_EXEC_CONFIG["min_exec_num"]:
        remaining_methods = [m for m in methods if m not in selected_methods]
        need_add = AUG_EXEC_CONFIG["min_exec_num"] - actual_exec_num
        add_methods = random.sample(remaining_methods, min(need_add, len(remaining_methods)))
        for method in add_methods:
            augmented_img, method_name = method(augmented_img)
            method_names.append(method_name)

    if len(method_names) > AUG_EXEC_CONFIG["max_exec_num"]:
        method_names = method_names[:AUG_EXEC_CONFIG["max_exec_num"]]

    return augmented_img, "_".join(method_names)
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
TRAIN_CSV = os.path.join("data", "traini.csv")
VAL_CSV = "val_image.csv"
TEST_CSV = "test_image.csv"

TRAIN_IMG_DIR = os.path.join("data", "train_table2")  # jpg images are here

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
CKPT_DIR = "CFP_checkpoints_TABLE2"
os.makedirs(CKPT_DIR, exist_ok=True)

# Model
MODEL_OPTIONS = {
    "efficientnet_b3": {"input_size": 300, "description": "EfficientNet-B3 (12.2M params)"},
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
    def __init__(
        self,
        csvfile,
        transform=None,
        img_dir=None,
        path_col="video_name",
        label_col="label",
        exts=(".jpg", ".jpeg", ".png"),
        drop_missing=True,
        # ===== 新增：训练增强配置 =====
        is_train=False,
        class_repeat_map=None,   # 例如 {0:5,1:8,2:12} 表示“每张图额外重复这么多次”
        enable_aug=False,
    ):
        self.df = pd.read_csv(csvfile)
        self.transform = transform
        self.img_dir = img_dir
        self.path_col = path_col
        self.label_col = label_col
        self.exts = exts

        self.is_train = is_train
        self.class_repeat_map = class_repeat_map or {0: 0, 1: 0, 2: 0}
        self.enable_aug = enable_aug

        if self.path_col not in self.df.columns:
            raise ValueError(f"CSV missing path_col='{self.path_col}'. columns={self.df.columns.tolist()}")
        if self.label_col not in self.df.columns:
            raise ValueError(f"CSV missing label_col='{self.label_col}'. columns={self.df.columns.tolist()}")

        self.name2path = None
        if self.img_dir is not None:
            self.name2path = {}
            for root, _, files in os.walk(self.img_dir):
                for fn in files:
                    low = fn.lower()
                    if low.endswith(self.exts):
                        base = os.path.splitext(fn)[0]
                        self.name2path[base] = os.path.join(root, fn)

        resolved_rows = []
        missing = 0

        for _, row in self.df.iterrows():
            raw = str(row[self.path_col])
            base = os.path.splitext(os.path.basename(raw))[0]

            img_path = None
            if self.img_dir is None:
                img_path = raw
                if not os.path.isabs(img_path):
                    img_path = os.path.normpath(img_path)
            else:
                cand = os.path.join(self.img_dir, raw)
                if os.path.exists(cand):
                    img_path = cand
                else:
                    for ext in self.exts:
                        cand2 = os.path.join(self.img_dir, base + ext)
                        if os.path.exists(cand2):
                            img_path = cand2
                            break
                    if img_path is None and self.name2path is not None:
                        img_path = self.name2path.get(base, None)

            if img_path is None or (not os.path.exists(img_path)):
                missing += 1
                if not drop_missing:
                    row = row.copy()
                    row["_resolved_path"] = img_path if img_path is not None else ""
                    resolved_rows.append(row)
                continue

            row = row.copy()
            row["_resolved_path"] = img_path
            resolved_rows.append(row)

        self.df = pd.DataFrame(resolved_rows)

        print(f"[Dataset] loaded csv={csvfile}")
        print(f"[Dataset] total rows in csv: {len(pd.read_csv(csvfile))}")
        print(f"[Dataset] resolved images: {len(self.df)}")
        print(f"[Dataset] missing/unmatched: {missing}")
        if len(self.df) == 0:
            raise RuntimeError("No images matched. Check img_dir and CSV naming.")

        # ===== 新增：构造“重复采样索引” =====
        self.index_map = None
        if self.is_train:
            idxs = []
            for i in range(len(self.df)):
                y = int(self.df.iloc[i][self.label_col])
                extra = int(self.class_repeat_map.get(y, 0))
                repeat = 1 + max(0, extra)  # 原图1次 + 额外extra次
                idxs.extend([i] * repeat)
            self.index_map = idxs
            print(f"[Dataset] train repeat enabled. base={len(self.df)} -> repeated={len(self.index_map)}")

    def __len__(self):
        return len(self.index_map) if self.index_map is not None else len(self.df)

    def __getitem__(self, idx):
        real_idx = self.index_map[idx] if self.index_map is not None else idx
        row = self.df.iloc[real_idx]
        img_path = row["_resolved_path"]
        label = int(row[self.label_col])

        image = Image.open(img_path).convert("RGB")

        # ===== 新增：仅训练集做增强（每次取样随机）=====
        if self.is_train and self.enable_aug:
            num_methods = random.choice([1, 2])
            image, _ = apply_augmentation(image, augmentation_methods, num_methods)

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
    class_repeat_map = {0: 5, 1: 8, 2: 12}

    train_ds = MyDataset(
        TRAIN_CSV,
        transform=basic_transform,
        img_dir=TRAIN_IMG_DIR,
        path_col="video_name",
        label_col="label",
        drop_missing=True,
        is_train=True,
        class_repeat_map=class_repeat_map,
        enable_aug=True,
    )

    val_ds = MyDataset(
        VAL_CSV,
        transform=basic_transform,
        img_dir=None,  # 关键：不拼目录
        path_col="path",
        label_col="class",
    )

    test_ds = MyDataset(
        TEST_CSV,
        transform=basic_transform,
        img_dir=None,
        path_col="path",
        label_col="class",
    )

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
        train_loss, train_acc, train_kappa = train_one_epoch(
            model, train_loader, optimizer, epoch, criterion, scheduler
        )
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