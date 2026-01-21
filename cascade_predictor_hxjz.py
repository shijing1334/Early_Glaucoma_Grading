import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, accuracy_score
import timm
import os
import logging

logger = logging.getLogger(__name__)


def extract_video_name(path):
    if isinstance(path, str):
        return os.path.basename(path).split('.')[0]
    return str(path)


class MyDataset(Dataset):
    def __init__(self, csvfile, transform=None):
        self.df = pd.read_csv(csvfile)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['path']
        label = int(row['class'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, img_path


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, hidden_dim, p=0.3):
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


# 多尺度特征融合（features_only）版
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
            nn.Dropout(0.2),
            ResidualMLPBlock(1024, 2048, p=0.1),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            ResidualMLPBlock(512, 1024, p=0.1),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        feats = self.backbone(x)  # list of (B,C,H,W)
        pooled = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in feats]  # list of (B,C)
        fused = torch.cat(pooled, dim=1)  # (B, sum(C))
        return self.classifier(fused)


class CascadePredictor:
    def __init__(self, model_path, csv_path, selected_model='efficientnet_b3', device='cuda',
                 cascade_thresholds=[0, 0, 0]):
        """
        Args:
            cascade_thresholds: List of 3 thresholds [threshold_non2, threshold_non0, threshold_non1]
        """
        self.model_path = model_path
        self.csv_path = csv_path
        self.selected_model = selected_model
        self.device = device

        # 解析阈值列表
        if isinstance(cascade_thresholds, list) and len(cascade_thresholds) == 3:
            self.threshold_non2, self.threshold_non0, self.threshold_non1 = cascade_thresholds
        elif isinstance(cascade_thresholds, (tuple)) and len(cascade_thresholds) == 3:
            self.threshold_non2, self.threshold_non0, self.threshold_non1 = cascade_thresholds
        else:
            logger.warning(f"Invalid cascade_thresholds format: {cascade_thresholds}. Using defaults.")
            self.threshold_non2, self.threshold_non0, self.threshold_non1 = 0.8, 0.8, 0.8

        logger.info(
            f"Cascade thresholds: non2={self.threshold_non2}, non0={self.threshold_non0}, non1={self.threshold_non1}")

        self.model_options = {
            'efficientnet_b3': {'input_size': 300},
            'efficientnet_b0': {'input_size': 224},
            'efficientnet_b1': {'input_size': 240},
            'efficientnet_b2': {'input_size': 260},
            'efficientnet_b4': {'input_size': 380},
            'resnet50': {'input_size': 224},
            'resnet101': {'input_size': 224},
            'densenet121': {'input_size': 224},
            'vit_base_patch16_224': {'input_size': 224},
            'xcit_small_24_p8_224': {'input_size': 224},
            'xcit_tiny_12_p8_224': {'input_size': 224},
            'tf_efficientnetv2_l':{'input_size': 300},
            'tf_efficientnetv2_m': {'input_size': 336},
            'beit_base_patch16_224': {'input_size': 224,},
            'tf_efficientnetv2_s': {'input_size': 300, 'description': 'EfficientNetV2-S (TF pretrained, 21.5M params)'},
        }
        self.input_size = self.model_options[selected_model]['input_size']
        self.batch_size = 10
        self._initialize_model()
        self._initialize_dataloader()
        logger.info(f"CascadePredictor initialized with model: {selected_model}")

    def _initialize_model(self):
        self.model = AdvancedClassifier(
            model_name=self.selected_model,
            num_classes=3,
            pretrained=False
        )
        try:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Loaded three-class model from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def _initialize_dataloader(self):
        self.transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        try:
            self.dataset = MyDataset(self.csv_path, transform=self.transform)
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0
            )
            logger.info(f"Loaded dataset with {len(self.dataset)} samples")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def cascade_predictions(self, three_class_outputs, binary_predictions, video_name):
        """
        修正字段名，使用新的sigmoid分数字段
        """
        # 找到对应的一阶段预测结果
        binary_pred = None
        for pred in binary_predictions:
            if pred['video_name'] == video_name:
                binary_pred = pred
                break

        if binary_pred is None:
            logger.warning(f"No binary prediction found for {video_name}, using three-class prediction")
            return three_class_outputs.argmax().item()

        # 获取三分类的概率和预测
        three_class_probs = F.softmax(three_class_outputs, dim=0)
        three_class_pred = three_class_outputs.argmax().item()

        # 获取一阶段的预测概率（修改字段名）
        pred_non2_score = binary_pred.get('pred_non2_score', 0.0)  # 正确字段名
        pred_non0_score = binary_pred.get('pred_non0_score', 0.0)  # 正确字段名
        pred_non1_score = binary_pred.get('pred_non1_score', 0.0)  # 正确字段名

        # 记录修正历史
        correction_history = []
        current_pred = three_class_pred

        # 连续应用修正规则
        max_iterations = 3  # 防止无限循环
        for iteration in range(max_iterations):
            original_pred = current_pred

            # 规则1: 如果一阶段认为非2，但二阶段预测为2，改为1
            if pred_non2_score > self.threshold_non2 and current_pred == 2:
                current_pred = 1
                correction_history.append(
                    f"Rule1: {original_pred}->1 (non2_score={pred_non2_score:.3f} > {self.threshold_non2})")

            # 规则2: 如果一阶段认为非0，但二阶段预测为0，改为1
            elif pred_non0_score > self.threshold_non0 and current_pred == 0:
                current_pred = 1
                correction_history.append(
                    f"Rule2: {original_pred}->1 (non0_score={pred_non0_score:.3f} > {self.threshold_non0})")

            # 规则3: 如果一阶段认为非1，但二阶段预测为1，改为第二大概率类别
            elif pred_non1_score > self.threshold_non1 and current_pred == 1:
                sorted_indices = torch.argsort(three_class_probs, descending=True)
                second_best_class = sorted_indices[1].item()
                current_pred = second_best_class
                correction_history.append(
                    f"Rule3: {original_pred}->{second_best_class} (non1_score={pred_non1_score:.3f} > {self.threshold_non1})")

            # 如果没有修正，跳出循环
            if current_pred == original_pred:
                break

        # 输出调试信息（如果有修正）
        if correction_history:
            logger.debug(f"Video {video_name}: {three_class_pred} -> {current_pred} | " +
                         " | ".join(correction_history))

        return current_pred

    def predict(self, binary_predictions):
        logger.info(f"Starting cascade prediction with {len(binary_predictions)} binary predictions")
        logger.info(
            f"Using thresholds: [non2={self.threshold_non2}, non0={self.threshold_non0}, non1={self.threshold_non1}]")

        binary_dict = {pred['video_name']: pred for pred in binary_predictions}
        all_preds_original = []
        all_preds_cascade = []
        all_labels = []
        cascade_changes = 0
        matched_samples = 0

        # 统计不同类型的修正
        correction_stats = {
            'pred2_to_1': 0,  # 预测2改为1
            'pred0_to_1': 0,  # 预测0改为1
            'pred1_to_second': 0,  # 预测1改为第二大概率类别
            'multiple_corrections': 0,  # 多次修正
            'pred2_to_0': 0,  # 预测2最终改为0
            'pred2_to_2': 0,  # 预测2最终还是2（经过中间修正）
        }

        with torch.no_grad():
            for imgs, labels, paths in tqdm(self.dataloader, desc='Cascade Prediction'):
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(imgs)
                preds_original = outputs.argmax(dim=1)

                batch_preds_cascade = []
                for i in range(len(preds_original)):
                    video_name = extract_video_name(paths[i])
                    original_pred = preds_original[i].cpu().item()

                    if video_name in binary_dict:
                        matched_samples += 1
                        cascade_pred = self.cascade_predictions(
                            outputs[i].cpu(), binary_predictions, video_name
                        )

                        if cascade_pred != original_pred:
                            cascade_changes += 1

                            # 统计修正类型（根据最终结果）
                            if original_pred == 2 and cascade_pred == 1:
                                correction_stats['pred2_to_1'] += 1
                            elif original_pred == 2 and cascade_pred == 0:
                                correction_stats['pred2_to_0'] += 1
                            elif original_pred == 0 and cascade_pred == 1:
                                correction_stats['pred0_to_1'] += 1
                            elif original_pred == 1 and cascade_pred != 1:
                                correction_stats['pred1_to_second'] += 1

                            # 检查是否可能经过了多次修正（启发式判断）
                            binary_pred = binary_dict[video_name]
                            non2_high = binary_pred.get('pred_non2_score', 0.0) > self.threshold_non2
                            non0_high = binary_pred.get('pred_non0_score', 0.0) > self.threshold_non0
                            non1_high = binary_pred.get('pred_non1_score', 0.0) > self.threshold_non1

                            active_rules = sum([non2_high, non0_high, non1_high])
                            if active_rules >= 2:
                                correction_stats['multiple_corrections'] += 1

                            logger.debug(
                                f"Cascade correction: {video_name} - Original: {original_pred} -> Cascade: {cascade_pred}")
                    else:
                        logger.warning(f"No binary prediction found for {video_name}")
                        cascade_pred = original_pred

                    batch_preds_cascade.append(cascade_pred)

                all_preds_original.extend(preds_original.cpu().numpy())
                all_preds_cascade.extend(batch_preds_cascade)
                all_labels.extend(labels.cpu().numpy())

        all_preds_original = np.array(all_preds_original)
        all_preds_cascade = np.array(all_preds_cascade)
        all_labels = np.array(all_labels)

        # 计算准确率和kappa值
        original_acc = accuracy_score(all_labels, all_preds_original) * 100
        cascade_acc = accuracy_score(all_labels, all_preds_cascade) * 100
        original_kappa = cohen_kappa_score(all_labels, all_preds_original, weights='quadratic')
        cascade_kappa = cohen_kappa_score(all_labels, all_preds_cascade, weights='quadratic')

        total_samples = len(all_labels)

        # 输出结果统计
        logger.info(f"Cascade Results:")
        logger.info(
            f"  Matched samples: {matched_samples}/{total_samples} ({matched_samples / total_samples * 100:.1f}%)")
        logger.info(f"  Original accuracy: {original_acc:.3f}%")
        logger.info(f"  Cascade accuracy: {cascade_acc:.3f}%")
        logger.info(f"  Original kappa: {original_kappa:.4f}")
        logger.info(f"  Cascade kappa: {cascade_kappa:.4f}")
        logger.info(
            f"  Cascade changes: {cascade_changes}/{total_samples} ({cascade_changes / total_samples * 100:.1f}%)")
        logger.info(f"  Accuracy improvement: {cascade_acc - original_acc:.3f}%")
        logger.info(f"  Kappa improvement: {cascade_kappa - original_kappa:.4f}")

        # 输出修正统计
        logger.info(f"  Correction statistics:")
        logger.info(f"    Pred 2->1: {correction_stats['pred2_to_1']}")
        logger.info(f"    Pred 0->1: {correction_stats['pred0_to_1']}")
        logger.info(f"    Pred 1->2nd: {correction_stats['pred1_to_second']}")
        logger.info(f"    Pred 2->0: {correction_stats['pred2_to_0']}")
        logger.info(f"    Multiple corrections: {correction_stats['multiple_corrections']}")

        # 逐样本输出：用于混淆矩阵/对齐一阶段csv
        per_sample = [
            {
                "video_name": extract_video_name(self.dataset.df.iloc[i]["path"]),
                "true_label": int(all_labels[i]),
                "cascade_prediction": int(all_preds_cascade[i]),
                "three_class_prediction": int(all_preds_original[i]),
            }
            for i in range(len(all_labels))
        ]
        return cascade_acc, cascade_kappa, per_sample