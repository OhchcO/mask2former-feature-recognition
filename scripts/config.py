# -*- coding: utf-8 -*-
"""
统一配置文件 — 修改这里即可适配不同数据集
"""
import os

# ============================================================
# 数据路径（根据平台自动选择）
# ============================================================
HOME = os.path.expanduser("~")

if os.name == "nt":
    # Windows
    DATA_ROOT = r"E:\soft\code\Mask2former_data\test-instance-0623"
    TRAIN_IMAGE_DIR = f"{DATA_ROOT}/train_encoded_views"
    TRAIN_MASK_DIR = f"{DATA_ROOT}/train_masks"
    VAL_IMAGE_DIR = f"{DATA_ROOT}/val_encoded_views"
    VAL_MASK_DIR = f"{DATA_ROOT}/val_masks"
    INSTANCE_CLASS_MAP_PATH = f"{DATA_ROOT}/class_map.json"
    MODEL_DIR = r"E:\soft\code\Mask2former"
    SAVE_DIR = r"E:\soft\code\Mask2former_data\results\models\finetuned_instance_model_v623"
    LOG_DIR = r"E:\soft\code\Mask2former_data\results\tensorboard_logs_ins_v623"
else:
    # Linux
    DATA_ROOT = os.path.join(HOME, "mask2former_data", "data_24")
    TRAIN_IMAGE_DIR = os.path.join(DATA_ROOT, "semantic_views_train")
    TRAIN_MASK_DIR = os.path.join(DATA_ROOT, "masks_train")
    VAL_IMAGE_DIR = os.path.join(DATA_ROOT, "semantic_views_val")
    VAL_MASK_DIR = os.path.join(DATA_ROOT, "masks_val")
    INSTANCE_CLASS_MAP_PATH = os.path.join(DATA_ROOT, "class_map_train.json")
    MODEL_DIR = os.path.join(HOME, "mask2former", "mask2former-feature-recognition")
    SAVE_DIR = os.path.join(HOME, "mask2former_data", "results", "models", "finetuned_instance_model_v61024_y")
    LOG_DIR = os.path.join(HOME, "mask2former_data", "results", "tensorboard_logs_ins_v61024_y")

# ============================================================
# 类别配置（换数据集只改这里）
# ============================================================
NUM_CLASSES = 3  # 不含背景（背景=0 自动处理）

# 类别名称（索引 → 名称，从0开始，不含背景）
CLASS_NAMES = {
    0: "开放型腔",
    1: "封闭型腔",
    2: "复合型腔",
}

# 类别权重（从 class_map.json 自动计算，Median Frequency Balancing）
def _calculate_class_weights(class_map_path, num_classes):
    """从 class_map.json 自动计算类别权重（实例分割专用）"""
    import json
    import numpy as np

    if not os.path.exists(class_map_path):
        print(f"Warning: class_map not found ({class_map_path}), using uniform weights")
        return [1.0] * num_classes

    with open(class_map_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 收集所有出现的类别ID，映射到连续范围
    all_class_ids = set()
    for img_name, instances in data.items():
        all_class_ids.update(instances.values())
    sorted_classes = sorted(all_class_ids)
    class_to_idx = {cid: idx for idx, cid in enumerate(sorted_classes)}

    class_counts = np.zeros(num_classes)
    for img_name, instances in data.items():
        for instance_id, class_id in instances.items():
            if class_id in class_to_idx:
                class_counts[class_to_idx[class_id]] += 1

    total = class_counts.sum()
    if total == 0:
        return [1.0] * num_classes

    class_pct = class_counts / total
    present = class_counts > 0
    median_freq = np.median(class_pct[present])

    weights = np.zeros(num_classes)
    weights[present] = median_freq / class_pct[present]
    weights = weights / weights[present].sum() * present.sum()

    print(f"Class weights (from {len(data)} images, {int(total)} instances): "
          f"{[round(w, 4) for w in weights.tolist()]}")
    return weights.tolist()


CLASS_WEIGHTS = _calculate_class_weights(INSTANCE_CLASS_MAP_PATH, NUM_CLASSES)

# ============================================================
# 训练超参
# ============================================================
BATCH_SIZE = 4
LEARNING_RATE = 5e-5
NUM_EPOCHS = 30
IMG_SIZE = (1024, 1024)
WEIGHT_DECAY = 0.01
PATIENCE = 10  # 早停耐心值：mAP连续不提升则停止
WARMUP_RATIO = 0.1
VAL_SPLIT = 0.2
SEED = 42

# ============================================================
# color_encoder 配置（STP面类型编码，与训练类别独立）
# ============================================================
ENCODER_TYPE_GAP = 51
ENCODER_TYPE_R_BASE = {
    0: 0,     # 平面:    R ∈ [0,  50]
    1: 51,    # 圆柱面:  R ∈ [51, 101]
    2: 102,   # 圆锥面:  R ∈ [102,152]
    3: 153,   # 球面:    R ∈ [153,203]
    4: 204,   # 其他面:  R ∈ [204,254]
}

ENCODER_TYPE_NAMES = {
    0: "平面",
    1: "圆柱面",
    2: "圆锥面",
    3: "球面",
    4: "其他面",
}
