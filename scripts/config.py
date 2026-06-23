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
    DATA_ROOT = "E:/soft/data/分割数据集"
    TRAIN_IMAGE_DIR = f"{DATA_ROOT}/train/images"
    TRAIN_MASK_DIR = f"{DATA_ROOT}/train/masks"
    VAL_IMAGE_DIR = f"{DATA_ROOT}/val/images"
    VAL_MASK_DIR = f"{DATA_ROOT}/val/masks"
    INSTANCE_CLASS_MAP_PATH = f"{DATA_ROOT}/train/class_map.json"
    MODEL_DIR = r"E:\soft\code\Mask2former"
    SAVE_DIR = r"E:\soft\code\Mask2former_data\results\models\finetuned_instance_model_v4"
    LOG_DIR = r"E:\soft\code\Mask2former_data\results\tensorboard_logs_ins"
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
NUM_CLASSES = 4  # 不含背景（背景=0 自动处理）

# 灰度掩码值 → 类别ID 映射
# key: 掩码图中的像素值, value: 模型训练用的类别ID
LABEL_MAPPING = {
    255: 0,   # 背景 → 0
    0: 1,     # 宽体槽 → 1
    1: 2,     # 封闭槽 → 2
    2: 3,     # 开放槽 → 3
    3: 4,     # 孔 → 4
}

# 类别名称（ID → 名称）
CLASS_NAMES = {
    0: "Background",
    1: "宽体槽",
    2: "封闭槽",
    3: "开放槽",
    4: "孔"
}

# 类别权重（从训练集自动计算，Median Frequency Balancing）
def _calculate_class_weights(mask_dir, num_classes, bg_value=255):
    """从掩码文件夹自动计算类别权重"""
    import numpy as np
    from PIL import Image

    if not os.path.exists(mask_dir):
        print(f"Warning: mask_dir not found ({mask_dir}), using uniform weights")
        return [1.0] * num_classes

    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
    if not mask_files:
        print(f"Warning: no masks found in {mask_dir}, using uniform weights")
        return [1.0] * num_classes

    total_pixels = 0
    class_counts = np.zeros(num_classes)

    for mask_file in mask_files:
        mask = np.array(Image.open(os.path.join(mask_dir, mask_file)).convert("L"))
        mapped = mask.copy()
        mapped[mask == bg_value] = 0
        for cls in range(num_classes):
            class_counts[cls] += np.sum(mapped == cls)
        total_pixels += mask.size

    class_pct = class_counts / total_pixels
    present = class_counts > 0
    median_freq = np.median(class_pct[present])

    weights = np.zeros(num_classes)
    weights[present] = median_freq / class_pct[present]
    weights = weights / weights[present].sum() * present.sum()

    print(f"Class weights (from {len(mask_files)} masks): {[round(w, 4) for w in weights.tolist()]}")
    return weights.tolist()


CLASS_WEIGHTS = _calculate_class_weights(TRAIN_MASK_DIR, num_classes=NUM_CLASSES + 1)

# 可视化颜色（类别ID → RGB）
SEMANTIC_COLORS = {
    0: [255, 255, 255],  # 背景 - 白色
    1: [255, 165, 0],    # 宽体槽 - 橙色
    2: [128, 0, 128],    # 封闭槽 - 紫色
    3: [0, 255, 255],    # 开放槽 - 青色
    4: [255, 0, 0],      # 孔 - 红色
}

# ============================================================
# 训练超参
# ============================================================
BATCH_SIZE = 4
LEARNING_RATE = 2e-5
NUM_EPOCHS = 20
IMG_SIZE = (1024, 1024)
WEIGHT_DECAY = 0.01
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
