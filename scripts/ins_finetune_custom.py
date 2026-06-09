# 实例分割训练
# 输入：图片 + 实例掩码（每个实例不同像素值）+ class_map.json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
from PIL import Image
import os
import json
import numpy as np
import time
from tqdm import tqdm

CLASS_NAMES = {
    0: "Background",
    1: "宽体槽",
    2: "封闭槽",
    3: "开放槽",
    4: "孔"
}

NUM_CLASSES = len(CLASS_NAMES)

CLASS_WEIGHTS = [0.0048, 0.7365, 0.9892, 2.5213, 0.7483]

# 调试开关：True 时只取前12个样本快速验证，False 使用全量数据
DEBUG_MODE = False
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
        print(f"Loaded class_map.json with {len(self.class_map)} images")

        # 匹配图片和掩码
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        valid_pairs = []
        for img_name in self.images:
            mask_path = os.path.join(mask_dir, img_name)
            if os.path.exists(mask_path) and img_name in self.class_map:
                valid_pairs.append(img_name)
        self.images = valid_pairs
        print(f"Found {len(self.images)} valid image-mask pairs with class_map entries")

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

        # 获取所有实例ID（0-254），排除背景255
        instance_ids = sorted([int(x) for x in np.unique(mask_np) if x < 255])

        mask_labels = []
        class_labels = []

        for inst_id in instance_ids:
            # 从class_map获取该实例的类别ID
            if str(inst_id) not in img_class_map:
                continue
            class_id = int(img_class_map[str(inst_id)])

            # 生成该实例的二值掩码
            binary_mask = (mask_np == inst_id).astype(np.float32)
            if binary_mask.sum() > 0:
                mask_labels.append(torch.tensor(binary_mask, dtype=torch.float32))
                class_labels.append(torch.tensor(class_id, dtype=torch.int64))

        return (
            inputs,
            torch.stack(mask_labels) if mask_labels else torch.zeros(1, *mask_np.shape, dtype=torch.float32),
            torch.stack(class_labels) if class_labels else torch.zeros(1, dtype=torch.int64),
        )


def collate_fn(batch):
    batch_inputs = {
        key: torch.stack([sample[0][key] for sample in batch])
        for key in batch[0][0].keys()
    }
    # 每个样本的实例数量可能不同，不能直接stack，需要列表
    batch_mask_labels = [sample[1] for sample in batch]
    batch_class_labels = [sample[2] for sample in batch]
    return batch_inputs, batch_mask_labels, batch_class_labels


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
        for inputs, mask_labels, class_labels in tqdm(dataloader, desc="  Validating"):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            batch_size = inputs["pixel_values"].shape[0]
            h, w = mask_labels[0].shape[1], mask_labels[0].shape[2]
            target_sizes = [(h, w)] * batch_size

            outputs = model(**inputs)
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
                    if cls == 0 or cls >= num_classes:
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
        return 0.0, [0.0] * (num_classes - 1)

    # 构建 COCO categories
    coco_categories = [
        {"id": cls_id, "name": name}
        for cls_id, name in CLASS_NAMES.items()
        if cls_id != 0
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
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # 获取各类别 AP（跳过背景，过滤无效值-1.0）
    ap_per_class = []
    for cls_id in range(1, num_classes):
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
    print("\n" + "=" * 60)
    print("[Sanity Check] Inspecting one training batch")
    print("=" * 60)

    model.eval()
    train_batch = next(iter(train_loader))
    inputs, mask_labels, class_labels = train_batch
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print(f"Train batch size: {len(mask_labels)}")
    for i in range(len(mask_labels)):
        cls_unique = class_labels[i].cpu().numpy().tolist()
        num_instances = mask_labels[i].shape[0]
        print(f"  Sample {i}: {num_instances} instances, classes={cls_unique}")

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


def finetune():
    model_dir = r"E:\soft\code\Mask2former"
    train_image_dir = r"E:\soft\code\Mask2former_data\data\semantic_views_train"
    train_mask_dir = r"E:\soft\code\Mask2former_data\data\ins_masks_train"
    train_class_map = r"E:\soft\code\Mask2former_data\data\class_map_train.json"
    val_image_dir = r"E:\soft\code\Mask2former_data\data\semantic_views_val"
    val_mask_dir = r"E:\soft\code\Mask2former_data\data\ins_masks_val"
    val_class_map = r"E:\soft\code\Mask2former_data\data\class_map_val.json"
    save_dir = r"E:\soft\code\Mask2former_data\results\models\finetuned_instance_model_v4"
    log_dir = r"E:\soft\code\Mask2former_data\results\tensorboard_logs_ins"

    print("=" * 60)
    print("Mask2Former Instance Segmentation Fine-tuning")
    print("=" * 60)

    # Step 1: 加载模型
    print("\n[Step 1/6] Loading pretrained model...")
    global processor
    processor = Mask2FormerImageProcessor.from_pretrained(model_dir)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        model_dir,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    )
    print(f"Set num_labels to {NUM_CLASSES}")

    if hasattr(model, 'config') and CLASS_WEIGHTS is not None:
        model.config.class_weight = CLASS_WEIGHTS
        print(f"Set class weights: {CLASS_WEIGHTS}")

    device = get_device()
    model = model.to(device)

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
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
        collate_fn=collate_fn,
    )

    val_loader = None
    if val_dataset and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
            pin_memory=True if device.type == "cuda" else False,
            collate_fn=collate_fn,
        )

    # Step 3: 配置训练参数
    print("\n[Step 3/6] Configuring training parameters...")
    for param in model.model.pixel_level_module.encoder.parameters():
        param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-5,
        weight_decay=0.01
    )

    num_epochs = 30
    patience = 10  # 早停耐心值：mAP连续不提升则停止
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    print(f"\nTraining Configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: 4")
    print(f"  Learning rate: 5e-5")
    print(f"  Image size: 1024x1024")
    print(f"  Number of classes: {NUM_CLASSES}")
    print(f"  Device: {device}")
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
    from datetime import datetime
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=os.path.join(log_dir, run_name))
    print(f"TensorBoard logs: {log_dir}/{run_name}")

    best_loss = float('inf')
    best_mAP = 0.0
    no_improve_epochs = 0
    training_history = []
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        epoch_start_time = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)
        for batch_idx, (inputs, mask_labels, class_labels) in enumerate(pbar):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            mask_labels_device = [m.to(device) for m in mask_labels]
            class_labels_device = [c.to(device) for c in class_labels]

            outputs = model(
                pixel_values=inputs["pixel_values"],
                mask_labels=mask_labels_device,
                class_labels=class_labels_device
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        scheduler.step()
        avg_train_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        current_lr = scheduler.get_last_lr()[0]

        val_mAP = None
        val_ap_per_class = None

        if val_loader:
            print(f"\n  Running validation...")
            val_mAP, val_ap_per_class = evaluate_model(
                model, processor, val_loader, device, NUM_CLASSES
            )
            print(f"  Validation - mAP@0.5: {val_mAP:.4f}")

        training_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_mAP': val_mAP,
            'val_ap_per_class': val_ap_per_class,
            'lr': current_lr,
            'time': epoch_time
        })

        # TensorBoard记录
        writer.add_scalar('Loss/train', avg_train_loss, epoch + 1)
        writer.add_scalar('LR', current_lr, epoch + 1)
        if val_mAP is not None:
            writer.add_scalar('mAP/val', val_mAP, epoch + 1)
        if val_ap_per_class is not None:
            for idx, cls_id in enumerate(range(1, NUM_CLASSES)):
                cls_name = CLASS_NAMES[cls_id]
                if not np.isnan(val_ap_per_class[idx]):
                    writer.add_scalar(f'AP_per_class/{cls_name}', val_ap_per_class[idx], epoch + 1)

        print(f"Epoch [{epoch+1}/{num_epochs}] completed, Train Loss: {avg_train_loss:.4f}, LR: {current_lr:.2e}, Time: {epoch_time:.1f}s")

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
            print(f"  -> New best mAP@0.5: {best_mAP:.4f}, model saved")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"\n  Early stopping: mAP not improved for {patience} epochs")
                break

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
    print(f"Model saved to: {save_dir}")

    print("\n" + "=" * 60)
    print("Training History")
    print("=" * 60)
    header = f"{'Epoch':<8}{'Loss':<12}{'mAP@0.5':<10}{'LR':<12}{'Time':<8}"
    print(header)
    print("-" * 60)

    for record in training_history:
        if record['epoch'] % 5 == 0 or record['epoch'] == 1:
            loss_str = f"{record['train_loss']:.4f}"
            mAP_str = f"{record['val_mAP']:.4f}" if record['val_mAP'] is not None else "N/A"
            lr_str = f"{record['lr']:.2e}"
            time_str = f"{record['time']:.1f}s"
            print(f"{record['epoch']:<8}{loss_str:<12}{mAP_str:<10}{lr_str:<12}{time_str:<8}")

    if val_loader and training_history[-1]['val_ap_per_class'] is not None:
        print("\n" + "=" * 60)
        print("Final Validation - AP per Class (IoU=0.5)")
        print("=" * 60)
        final_ap = training_history[-1]['val_ap_per_class']
        for idx, cls_id in enumerate(range(1, NUM_CLASSES)):
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
    print(f'  processor = Mask2FormerImageProcessor.from_pretrained("{save_dir}")')
    print(f'  model = Mask2FormerForUniversalSegmentation.from_pretrained("{save_dir}")')


if __name__ == "__main__":
    finetune()
