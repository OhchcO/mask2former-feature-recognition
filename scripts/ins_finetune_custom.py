# 实例分割训练 (Linux版)
# 输入：图片 + 实例掩码（每个实例不同像素值）+ class_map.json
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from config import (  # type: ignore
    CLASS_NAMES, NUM_CLASSES, CLASS_WEIGHTS,
    TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, VAL_IMAGE_DIR, VAL_MASK_DIR,
    INSTANCE_CLASS_MAP_PATH, VAL_CLASS_MAP_PATH, MODEL_DIR, SAVE_DIR, LOG_DIR,
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, PATIENCE, WEIGHT_DECAY, SEED, CLASS_ID_MAP,
)

import torch
import random
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
from PIL import Image
import json
import pickle
import numpy as np
import time
from datetime import datetime
from tqdm import tqdm

# 调试开关：True 时只取前12个样本快速验证，False 使用全量数据
DEBUG_MODE = True
MAX_SAMPLES = 12


class InstanceSegmentationDataset(Dataset):
    """实例分割数据集
    输入：
        image_dir: 原始图片文件夹
        mask_dir: 实例掩码文件夹（0-254=实例, 255=背景）
        class_map_path: class_map.json 路径（实例ID→类别ID映射）
    """
    def __init__(self, image_dir, mask_dir, class_map_path, processor, size=(1024, 1024)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.processor = processor
        self.size = size

        # 加载 class_map.json
        with open(class_map_path, 'r', encoding='utf-8') as f:
            self.class_map = json.load(f)
        print(f"  Loaded class_map.json with {len(self.class_map)} images")

        # 匹配图片和掩码（带缓存，避免每次启动都逐个验证）
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        cache_path = os.path.join(image_dir, "_valid_pairs.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                self.images = pickle.load(f)
            print(f"  Loaded {len(self.images)} valid images from cache (instant)")
        else:
            valid_pairs = []
            empty_count = 0
            print(f"  Validating {len(self.images)} masks (first run, will cache)...")
            for img_name in tqdm(self.images, desc="  Validating masks", unit="img"):
                mask_path = os.path.join(mask_dir, img_name)
                if os.path.exists(mask_path) and img_name in self.class_map:
                    mask_arr = np.array(Image.open(mask_path).convert("L"))
                    if np.any(mask_arr < 255):
                        valid_pairs.append(img_name)
                    else:
                        empty_count += 1
            self.images = valid_pairs
            with open(cache_path, "wb") as f:
                pickle.dump(valid_pairs, f)
            print(f"  Found {len(self.images)} valid images ({empty_count} empty masks skipped), cached to {cache_path}")

        # 使用 config 中定义的显式映射
        self.class_to_idx = CLASS_ID_MAP
        self.idx_to_name = CLASS_NAMES
        print(f"Class ID mapping: {self.class_to_idx}")
        print(f"Index to name: {self.idx_to_name}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.resize(self.size)
        mask = mask.resize(self.size, Image.NEAREST)

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        mask_np = np.array(mask)
        img_class_map = self.class_map[img_name]

        # 获取所有实例ID，只排除背景255（0-254都是合法实例）
        instance_ids = sorted([int(x) for x in np.unique(mask_np) if x < 255])

        mask_labels = []
        class_labels = []

        for inst_id in instance_ids:
            # 从class_map获取该实例的类别ID
            if str(inst_id) not in img_class_map:
                continue
            class_id = int(img_class_map[str(inst_id)])
            # 映射到连续索引（如 5→0, 6→1, 7→2）
            if class_id not in self.class_to_idx:
                continue
            class_id = self.class_to_idx[class_id]

            # 生成该实例的二值掩码
            binary_mask = (mask_np == inst_id).astype(np.float32)
            if binary_mask.sum() > 0:
                mask_labels.append(torch.tensor(binary_mask, dtype=torch.float32))
                class_labels.append(torch.tensor(class_id, dtype=torch.int64))

        return (
            inputs,
            torch.stack(mask_labels) if mask_labels else torch.zeros(1, *mask_np.shape, dtype=torch.float32),
            torch.stack(class_labels) if class_labels else torch.zeros(1, dtype=torch.int64),
            img_name,
        )


def collate_fn(batch):
    # 过滤掉空标签的样本（防止 cost matrix infeasible）
    valid_batch = [s for s in batch if len(s[1]) > 0]
    if len(valid_batch) == 0:
        return None
    batch_inputs = {
        key: torch.stack([sample[0][key] for sample in valid_batch])
        for key in valid_batch[0][0].keys()
    }
    # 每个样本的实例数量可能不同，不能直接stack，需要列表
    batch_mask_labels = [sample[1] for sample in valid_batch]
    batch_class_labels = [sample[2] for sample in valid_batch]
    batch_img_names = [sample[3] for sample in valid_batch]
    return batch_inputs, batch_mask_labels, batch_class_labels, batch_img_names


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.1f} GB")
    else:
        device = torch.device("cpu")
        print("Warning: CUDA not available, using CPU")
    return device


def mask_to_coco_rle(binary_mask):
    """将二值掩码转换为 COCO RLE 格式"""
    from pycocotools import mask as coco_mask
    rle = coco_mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def mask_to_bbox(binary_mask):
    """从二值掩码获取 bbox [x, y, w, h]"""
    ys, xs = np.where(binary_mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]


def evaluate_model(model, processor, dataloader, device, num_classes, iou_threshold=0.5):
    """评估实例分割模型（mAP）
    使用 pycocotools COCOeval 进行标准 COCO 评估
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    model.eval()

    # COCO 格式数据
    coco_images = []
    coco_annotations = []
    coco_results = []

    ann_id = 0
    img_id = 0

    with torch.no_grad():
        for inputs, mask_labels, class_labels, _ in tqdm(dataloader, desc="  Validating"):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            batch_size = inputs["pixel_values"].shape[0]
            h, w = mask_labels[0].shape[1], mask_labels[0].shape[2]
            target_sizes = [(h, w)] * batch_size

            try:
                outputs = model(**inputs)
            except ValueError as e:
                if "infeasible" in str(e):
                    continue
                raise
            pred_results = processor.post_process_instance_segmentation(
                outputs, target_sizes=target_sizes, threshold=0.5
            )

            for i, pred_result in enumerate(pred_results):
                seg_map = pred_result['segmentation'].cpu().numpy()
                segments_info = pred_result['segments_info']

                cur_img_id = img_id + i

                coco_images.append({
                    "id": cur_img_id,
                    "width": w,
                    "height": h
                })

                # 收集预测
                for seg in segments_info:
                    inst_id = seg['id']
                    cls = seg['label_id']
                    score = seg['score']

                    if cls < num_classes:
                        binary_mask = (seg_map == inst_id).astype(np.uint8)
                        area = int(binary_mask.sum())
                        if area == 0:
                            continue
                        rle = mask_to_coco_rle(binary_mask)
                        bbox = mask_to_bbox(binary_mask)

                        coco_results.append({
                            "image_id": cur_img_id,
                            "category_id": cls,
                            "segmentation": rle,
                            "area": area,
                            "bbox": bbox,
                            "score": score
                        })

                # 收集 GT
                gt_mask_labels = mask_labels[i]
                gt_class_labels = class_labels[i]
                for g_idx in range(len(gt_class_labels)):
                    cls = int(gt_class_labels[g_idx])
                    if cls >= num_classes:
                        continue
                    gt_mask = gt_mask_labels[g_idx].numpy() > 0.5
                    area = int(gt_mask.sum())
                    if area == 0:
                        continue
                    rle = mask_to_coco_rle(gt_mask.astype(np.uint8))
                    bbox = mask_to_bbox(gt_mask)

                    coco_annotations.append({
                        "id": ann_id,
                        "image_id": cur_img_id,
                        "category_id": cls,
                        "segmentation": rle,
                        "area": area,
                        "bbox": bbox,
                        "iscrowd": 0
                    })
                    ann_id += 1

            img_id += batch_size

    # 没有预测或没有 GT 则返回 0
    if len(coco_annotations) == 0 or len(coco_results) == 0:
        return 0.0, [0.0] * num_classes

    # 构建 COCO categories
    coco_categories = [
        {"id": cls_id, "name": name}
        for cls_id, name in CLASS_NAMES.items()
    ]

    coco_gt = COCO()
    coco_gt.dataset = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": coco_categories
    }
    coco_gt.createIndex()

    coco_dt = coco_gt.loadRes(coco_results)

    # COCOeval: 使用 mask IoU（ segm 模式）
    coco_eval = COCOeval(coco_gt, coco_dt, "segm")
    coco_eval.params.iouThrs = [iou_threshold]  # 只用 0.5
    coco_eval.params.maxDets = [1, 10, 100]
    import contextlib
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull):
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

    # 获取各类别 AP（过滤无效值-1.0）
    ap_per_class = []
    for cls_id in range(num_classes):
        cls_idx = coco_eval.params.catIds.index(cls_id) if cls_id in coco_eval.params.catIds else -1
        if cls_idx >= 0:
            prec = coco_eval.eval['precision'][0, :, cls_idx, 0, -1]
            prec_valid = prec[prec >= 0]  # 过滤 -1.0（无GT的无效值）
            ap_val = np.mean(prec_valid) if len(prec_valid) > 0 else 0.0
            ap_per_class.append(float(ap_val))
        else:
            ap_per_class.append(0.0)

    valid_aps = [ap for ap in ap_per_class if ap >= 0]
    mAP = np.mean(valid_aps) if valid_aps else 0.0

    return mAP, ap_per_class


def run_sanity_check(model, processor, train_loader, device):
    if len(train_loader.dataset) == 0:
        print("\n[Sanity Check] Skipped: no training samples")
        return

    print("\n" + "=" * 60)
    print("[Sanity Check] Inspecting one training batch")
    print("=" * 60)

    model.eval()
    train_batch = next(iter(train_loader))
    inputs, mask_labels, class_labels, img_names = train_batch
    inputs = {k: v.to(device) for k, v in inputs.items()}

    idx_to_name = train_loader.dataset.idx_to_name
    print(f"Train batch size: {len(mask_labels)}")
    for i in range(len(mask_labels)):
        cls_ids = class_labels[i].cpu().numpy().tolist()
        cls_names = [idx_to_name.get(c, f"Unknown({c})") for c in cls_ids]
        num_instances = mask_labels[i].shape[0]
        print(f"  Sample {i}: {img_names[i]}, {num_instances} instances, classes={cls_names}")

    with torch.no_grad():
        # 转为列表格式传给模型
        mask_labels_device = [m.to(device) for m in mask_labels]
        class_labels_device = [c.to(device) for c in class_labels]
        outputs = model(
            pixel_values=inputs["pixel_values"],
            mask_labels=mask_labels_device,
            class_labels=class_labels_device
        )
        print(f"Train batch forward loss: {outputs.loss.item():.4f}")

    print("=" * 60)


class Tee:
    """同时输出到终端和文件"""
    def __init__(self, filepath):
        self.file = open(filepath, 'w', encoding='utf-8')
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()
        sys.stdout = self.stdout
        sys.stderr = self.stderr


def finetune():
    # 设置随机种子
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    model_dir = MODEL_DIR
    train_image_dir = TRAIN_IMAGE_DIR
    train_mask_dir = TRAIN_MASK_DIR
    train_class_map = INSTANCE_CLASS_MAP_PATH
    val_image_dir = VAL_IMAGE_DIR
    val_mask_dir = VAL_MASK_DIR
    val_class_map = VAL_CLASS_MAP_PATH
    save_dir = SAVE_DIR
    log_dir = LOG_DIR

    # 创建时间戳子目录，每次训练独立存放
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(save_dir, run_name)
    log_dir = os.path.join(log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)

    # 同时输出到终端和日志文件
    tee = Tee(os.path.join(log_dir, "train.log"))
    sys.stdout = tee
    sys.stderr = tee


    print("=" * 60)
    print("Mask2Former Instance Segmentation Fine-tuning")
    print(f"Log file: {os.path.join(log_dir, 'train.log')}")
    print("=" * 60)

    # Step 1: 加载模型
    print("\n[Step 1/6] Loading pretrained model...")
    processor = Mask2FormerImageProcessor.from_pretrained(model_dir)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        model_dir,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    )
    print(f"Set num_labels to {NUM_CLASSES}")

    device = get_device()
    model = model.to(device)

    # 设置类别权重（传给 nn.CrossEntropyLoss 的 weight 参数）
    # empty_weight 形状: [num_classes + 1]，最后一个是 no-object 权重
    if CLASS_WEIGHTS is not None:
        class_weights_tensor = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32)
        eos_coef = model.criterion.eos_coef
        full_weights = torch.cat([class_weights_tensor, torch.tensor([eos_coef])])
        model.criterion.empty_weight = full_weights.to(device)
        # print(f"Set per-class weights: {CLASS_WEIGHTS}")
        print(f"Full criterion weights (with no-object): {model.criterion.empty_weight.tolist()}")

    # Step 2: 加载数据
    print("\n[Step 2/6] Preparing data...")
    if not os.path.exists(train_image_dir):
        print(f"Error: Training image directory not found: {train_image_dir}")
        return
    if not os.path.exists(train_mask_dir):
        print(f"Error: Training mask directory not found: {train_mask_dir}")
        return
    if not os.path.exists(train_class_map):
        print(f"Error: Training class_map not found: {train_class_map}")
        return

    train_dataset = InstanceSegmentationDataset(
        train_image_dir, train_mask_dir, train_class_map, processor
    )
    # 快速验证：只取前MAX_SAMPLES个样本
    if DEBUG_MODE and len(train_dataset) > MAX_SAMPLES:
        train_dataset.images = train_dataset.images[:MAX_SAMPLES]
        print(f"[DEBUG] Truncated train dataset to {len(train_dataset)} samples")

    val_dataset = None
    if (os.path.exists(val_image_dir) and os.path.exists(val_mask_dir)
            and os.path.exists(val_class_map)):
        val_dataset = InstanceSegmentationDataset(
            val_image_dir, val_mask_dir, val_class_map, processor
        )
        if DEBUG_MODE and len(val_dataset) > MAX_SAMPLES:
            val_dataset.images = val_dataset.images[:MAX_SAMPLES]
            print(f"[DEBUG] Truncated val dataset to {len(val_dataset)} samples")
    else:
        print("Warning: Validation dataset not found, skipping validation")

    if len(train_dataset) == 0:
        print("Error: No valid training samples found!")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False,
        collate_fn=collate_fn,
    )

    val_loader = None
    if val_dataset and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True if device.type == "cuda" else False,
            collate_fn=collate_fn,
        )

    # Step 3: 配置训练参数
    print("\n[Step 3/6] Configuring training parameters...")
    # 全量微调：不冻结任何参数（encoder + decoder + 分类头全部参与训练）
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )

    print(f"\nTraining Configuration:")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Image size: 1024x1024")
    print(f"  Number of classes: {NUM_CLASSES}")
    print(f"  Device: {device}")
    print(f"  FP32 Full Precision")
    print(f"  Training samples: {len(train_dataset)}")
    if val_dataset:
        print(f"  Validation samples: {len(val_dataset)}")

    print("\nClass Legend:")
    for class_id, class_name in CLASS_NAMES.items():
        print(f"  {class_id}: {class_name}")

    run_sanity_check(model, processor, train_loader, device)

    # Step 4: 开始训练
    print("\n[Step 4/6] Starting fine-tuning...")
    print("=" * 60)

    # 初始化TensorBoard（每次运行创建带时间戳的子文件夹）
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs: {log_dir}")

    best_loss = float('inf')
    best_mAP = 0.0
    no_improve_epochs = 0
    training_history = []
    start_time = time.time()
    global_step = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        epoch_start_time = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]", leave=False)
        for batch_idx, batch in enumerate(pbar):
            if batch is None:
                continue
            inputs, mask_labels, class_labels, _ = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            mask_labels_device = [m.to(device) for m in mask_labels]
            class_labels_device = [c.to(device) for c in class_labels]

            try:
                outputs = model(
                    pixel_values=inputs["pixel_values"],
                    mask_labels=mask_labels_device,
                    class_labels=class_labels_device
                )
                loss = outputs.loss
            except ValueError as e:
                if "infeasible" in str(e):
                    print(f"  [Skip] Cost matrix infeasible at batch {batch_idx}, skipping...")
                    optimizer.zero_grad()
                    continue
                raise

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            global_step += 1
            # 每 100 步记录一次 train loss 到 TensorBoard
            if global_step % 100 == 0:
                writer.add_scalar('Loss/train_iter', loss.item(), global_step)
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        scheduler.step()
        avg_train_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        current_lr = scheduler.get_last_lr()[0]

        val_loss = None
        val_mAP = None
        val_ap_per_class = None

        if val_loader:
            print(f"\n  Running validation...")
            # 计算验证损失（每 epoch）
            model.eval()
            val_total_loss = 0
            val_batch_count = 0
            with torch.no_grad():
                for v_idx, (v_inputs, v_mask_labels, v_class_labels, _) in enumerate(val_loader):
                    v_inputs = {k: v.to(device) for k, v in v_inputs.items()}
                    v_mask_device = [m.to(device) for m in v_mask_labels]
                    v_class_device = [c.to(device) for c in v_class_labels]
                    try:
                        v_outputs = model(
                            pixel_values=v_inputs["pixel_values"],
                            mask_labels=v_mask_device,
                            class_labels=v_class_device
                        )
                    except ValueError as e:
                        if "infeasible" in str(e):
                            print(f"  [Skip Val] Cost matrix infeasible at batch {v_idx}, skipping...")
                            continue
                        raise
                    val_total_loss += v_outputs.loss.item()
                    val_batch_count += 1
            val_loss = val_total_loss / max(val_batch_count, 1)
            model.train()

            # 完整 mAP 评估（每 5 epoch）
            if (epoch + 1) % 5 == 0:
                val_mAP, val_ap_per_class = evaluate_model(
                    model, processor, val_loader, device, NUM_CLASSES
                )
                print(f"  Validation - Loss: {val_loss:.4f}, mAP@0.5: {val_mAP:.4f}")
            else:
                print(f"  Validation - Loss: {val_loss:.4f} (mAP next epoch {(5 - (epoch + 1) % 5)} epoch(s))")

        training_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'val_mAP': val_mAP,
            'val_ap_per_class': val_ap_per_class,
            'lr': current_lr,
            'time': epoch_time
        })

        # TensorBoard记录（全部按 iteration 记录）
        writer.add_scalar('Loss/train_epoch', avg_train_loss, global_step)
        if val_loss is not None:
            writer.add_scalar('Loss/val', val_loss, global_step)
        writer.add_scalar('LR', current_lr, global_step)
        if val_mAP is not None:
            writer.add_scalar('mAP/val', val_mAP, global_step)
        if val_ap_per_class is not None:
            for idx, cls_id in enumerate(range(NUM_CLASSES)):
                cls_name = CLASS_NAMES[cls_id]
                if not np.isnan(val_ap_per_class[idx]):
                    writer.add_scalar(f'AP_per_class/{cls_name}', val_ap_per_class[idx], global_step)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] completed, Train Loss: {avg_train_loss:.4f}, LR: {current_lr:.2e}, Time: {epoch_time:.1f}s")

        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            print(f"  -> New best loss: {best_loss:.4f}")

        if val_mAP is not None and val_mAP > best_mAP:
            best_mAP = val_mAP
            no_improve_epochs = 0
            # 保存最佳模型
            best_save_dir = os.path.join(save_dir, "best_model")
            os.makedirs(best_save_dir, exist_ok=True)
            model.save_pretrained(best_save_dir)
            processor.save_pretrained(best_save_dir)
            print(f"  -> New best mAP@0.5: {best_mAP:.4f}, best model saved")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= PATIENCE:
                print(f"\n  Early stopping: mAP not improved for {PATIENCE} epochs")
                break

        # 每个 epoch 结束保存最新权重（防止中途崩溃丢失进度）
        latest_save_dir = os.path.join(save_dir, "latest_model")
        os.makedirs(latest_save_dir, exist_ok=True)
        model.save_pretrained(latest_save_dir)
        processor.save_pretrained(latest_save_dir)

        # 每 30 轮保存一个 checkpoint
        if (epoch + 1) % 30 == 0:
            ckpt_save_dir = os.path.join(save_dir, f"checkpoint_epoch{epoch + 1}")
            os.makedirs(ckpt_save_dir, exist_ok=True)
            model.save_pretrained(ckpt_save_dir)
            processor.save_pretrained(ckpt_save_dir)
            print(f"  -> Checkpoint saved: epoch {epoch + 1}")

    total_time = time.time() - start_time

    # Step 5: 保存模型
    print("\n" + "=" * 60)
    print("[Step 5/6] Saving model...")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)

    # Step 6: 训练总结
    print("\n" + "=" * 60)
    print("[Step 6/6] Training Summary")
    print("=" * 60)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Best mAP@0.5: {best_mAP:.4f}")
    print(f"Best model: {os.path.join(save_dir, 'best_model')}")
    print(f"Latest model: {os.path.join(save_dir, 'latest_model')}")

    print("\n" + "=" * 60)
    print("Training History")
    print("=" * 60)
    header = f"{'Epoch':<8}{'TrainLoss':<12}{'ValLoss':<12}{'mAP@0.5':<10}{'LR':<12}{'Time':<8}"
    print(header)
    print("-" * 60)

    for record in training_history:
        if record['epoch'] % 5 == 0 or record['epoch'] == 1:
            loss_str = f"{record['train_loss']:.4f}"
            val_loss_str = f"{record['val_loss']:.4f}" if record['val_loss'] is not None else "N/A"
            mAP_str = f"{record['val_mAP']:.4f}" if record['val_mAP'] is not None else "N/A"
            lr_str = f"{record['lr']:.2e}"
            time_str = f"{record['time']:.1f}s"
            print(f"{record['epoch']:<8}{loss_str:<12}{val_loss_str:<12}{mAP_str:<10}{lr_str:<12}{time_str:<8}")

    if val_loader and training_history[-1]['val_ap_per_class'] is not None:
        print("\n" + "=" * 60)
        print("Final Validation - AP per Class (IoU=0.5)")
        print("=" * 60)
        final_ap = training_history[-1]['val_ap_per_class']
        for idx, cls_id in enumerate(range(NUM_CLASSES)):
            cls_name = CLASS_NAMES[cls_id]
            ap_val = final_ap[idx]
            if not np.isnan(ap_val):
                print(f"  {cls_name}: {ap_val:.4f}")
            else:
                print(f"  {cls_name}: N/A")

    # 关闭TensorBoard
    writer.close()

    print("\n" + "=" * 60)
    print("View training curves with TensorBoard:")
    print(f"  tensorboard --logdir={log_dir}")
    print("=" * 60)

    print("\nUsage:")
    best_path = os.path.join(save_dir, "best_model")
    latest_path = os.path.join(save_dir, "latest_model")
    print(f'  # 加载最佳模型:')
    print(f'  processor = Mask2FormerImageProcessor.from_pretrained("{best_path}")')
    print(f'  model = Mask2FormerForUniversalSegmentation.from_pretrained("{best_path}")')
    print(f'  # 或加载最新模型:')
    print(f'  processor = Mask2FormerImageProcessor.from_pretrained("{latest_path}")')
    print(f'  model = Mask2FormerForUniversalSegmentation.from_pretrained("{latest_path}")')

    tee.close()
    print(f"\nTraining log saved to: {log_dir}/train.log")


if __name__ == "__main__":
    finetune()
