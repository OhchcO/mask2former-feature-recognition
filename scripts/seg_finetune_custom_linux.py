# 语义分割训练 (Linux版)
# 输入：图片 + 语义掩码（0=背景, 1-4=类别）
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
from PIL import Image
import os
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

LABEL_MAPPING = {
    255: 0,
    0: 1,
    1: 2,
    2: 3,
    3: 4
}

CLASS_WEIGHTS = [0.0048, 0.7365, 0.9892, 2.5213, 0.7483]

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, processor, size=(1024, 1024)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.processor = processor
        self.size = size
        
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        
        valid_pairs = []
        for img_name in self.images:
            mask_path = os.path.join(mask_dir, img_name)
            if os.path.exists(mask_path):
                valid_pairs.append(img_name)
        
        self.images = valid_pairs
        print(f"Found {len(self.images)} valid image-mask pairs")
    
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
        
        mapped_mask = np.zeros_like(mask_np)
        for src_val, dst_val in LABEL_MAPPING.items():
            mapped_mask[mask_np == src_val] = dst_val
        
        unique_labels = np.unique(mapped_mask)
        mask_labels = []
        class_labels = []
        
        for label_id in unique_labels:
            binary_mask = (mapped_mask == label_id).astype(np.float32)
            if binary_mask.sum() > 0:
                mask_labels.append(torch.tensor(binary_mask, dtype=torch.float32))
                class_labels.append(torch.tensor(int(label_id), dtype=torch.int64))

        return (
            inputs,
            torch.stack(mask_labels),
            torch.stack(class_labels),
            torch.tensor(mapped_mask, dtype=torch.long),
        )


def collate_fn(batch):
    batch_inputs = {
        key: torch.stack([sample[0][key] for sample in batch])
        for key in batch[0][0].keys()
    }
    batch_mask_labels = [sample[1] for sample in batch]
    batch_class_labels = [sample[2] for sample in batch]
    batch_gt_masks = torch.stack([sample[3] for sample in batch])
    return batch_inputs, batch_mask_labels, batch_class_labels, batch_gt_masks

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

def calculate_metrics(pred_mask, gt_mask, num_classes):
    pred_mask = pred_mask.cpu().numpy()
    gt_mask = gt_mask.cpu().numpy()
    
    correct = np.sum(pred_mask == gt_mask)
    total = pred_mask.size
    accuracy = correct / total
    
    iou_per_class = []
    for cls in range(num_classes):
        pred_cls = (pred_mask == cls)
        gt_cls = (gt_mask == cls)
        
        intersection = np.sum(pred_cls & gt_cls)
        union = np.sum(pred_cls | gt_cls)
        
        if union > 0:
            iou = intersection / union
        else:
            iou = float('nan')
        
        iou_per_class.append(iou)
    
    valid_ious = [iou for iou in iou_per_class if not np.isnan(iou)]
    miou = np.mean(valid_ious) if valid_ious else 0.0
    
    return accuracy, miou, iou_per_class

def evaluate_model(model, dataloader, device, num_classes):
    model.eval()
    
    total_correct = 0
    total_pixels = 0
    iou_sums = np.zeros(num_classes)
    iou_counts = np.zeros(num_classes)
    
    with torch.no_grad():
        for inputs, mask_labels, class_labels, gt_masks in tqdm(dataloader, desc="  Validating"):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            
            batch_size = gt_masks.shape[0]
            h, w = gt_masks.shape[1], gt_masks.shape[2]
            target_sizes = [(h, w)] * batch_size
            pred_results = processor.post_process_semantic_segmentation(
                outputs, target_sizes=target_sizes
            )
            
            for i, pred_seg in enumerate(pred_results):
                if isinstance(pred_seg, torch.Tensor):
                    pred_seg = pred_seg.cpu().numpy()
                
                gt_mask = gt_masks[i].numpy()
                
                correct = np.sum(pred_seg == gt_mask)
                total = pred_seg.size
                total_correct += correct
                total_pixels += total
                
                for cls in range(num_classes):
                    pred_cls = (pred_seg == cls)
                    gt_cls = (gt_mask == cls)
                    
                    intersection = np.sum(pred_cls & gt_cls)
                    union = np.sum(pred_cls | gt_cls)
                    
                    if union > 0:
                        iou_sums[cls] += intersection / union
                        iou_counts[cls] += 1
    
    accuracy = total_correct / total_pixels if total_pixels > 0 else 0
    
    iou_per_class = []
    for cls in range(num_classes):
        if iou_counts[cls] > 0:
            iou_per_class.append(iou_sums[cls] / iou_counts[cls])
        else:
            iou_per_class.append(float('nan'))
    
    valid_ious = [iou for iou in iou_per_class if not np.isnan(iou)]
    miou = np.mean(valid_ious) if valid_ious else 0.0
    
    return accuracy, miou, iou_per_class


def run_sanity_check(model, train_loader, val_loader, device):
    print("\n" + "=" * 60)
    print("[Sanity Check] Inspecting one training batch")
    print("=" * 60)

    model.eval()

    train_batch = next(iter(train_loader))
    inputs, mask_labels, class_labels, gt_masks = train_batch
    inputs = {k: v.to(device) for k, v in inputs.items()}
    mask_labels = [m.to(device) for m in mask_labels]
    class_labels = [c.to(device) for c in class_labels]

    print(f"Train batch size: {gt_masks.shape[0]}")
    for i in range(gt_masks.shape[0]):
        gt_unique = torch.unique(gt_masks[i]).cpu().numpy().tolist()
        cls_unique = torch.unique(class_labels[i]).cpu().numpy().tolist()
        print(f"  Train sample {i}: gt classes={gt_unique}, supervised classes={cls_unique}, mask_count={mask_labels[i].shape[0]}")

    with torch.no_grad():
        outputs = model(
            pixel_values=inputs["pixel_values"],
            mask_labels=mask_labels,
            class_labels=class_labels
        )
        print(f"Train batch forward loss: {outputs.loss.item():.4f}")

        target_sizes = [(gt_masks.shape[1], gt_masks.shape[2])] * gt_masks.shape[0]
        pred_results = processor.post_process_semantic_segmentation(
            outputs, target_sizes=target_sizes
        )
        for i, pred_seg in enumerate(pred_results):
            pred_unique = torch.unique(pred_seg).cpu().numpy().tolist()
            print(f"  Train sample {i}: pred classes={pred_unique}")

    if val_loader is None:
        print("\n[Sanity Check] No validation loader, skip validation batch check")
        return

    print("\n" + "=" * 60)
    print("[Sanity Check] Inspecting one validation batch")
    print("=" * 60)

    val_batch = next(iter(val_loader))
    inputs, _, _, gt_masks = val_batch
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print(f"Validation batch size: {gt_masks.shape[0]}")
    for i in range(gt_masks.shape[0]):
        gt_unique = torch.unique(gt_masks[i]).cpu().numpy().tolist()
        print(f"  Val sample {i}: gt classes={gt_unique}")

    with torch.no_grad():
        outputs = model(**inputs)
        target_sizes = [(gt_masks.shape[1], gt_masks.shape[2])] * gt_masks.shape[0]
        pred_results = processor.post_process_semantic_segmentation(
            outputs, target_sizes=target_sizes
        )
        for i, pred_seg in enumerate(pred_results):
            pred_unique = torch.unique(pred_seg).cpu().numpy().tolist()
            print(f"  Val sample {i}: pred classes={pred_unique}")

    print("=" * 60)

def finetune():
    HOME = os.path.expanduser("~")
    model_dir = os.path.join(HOME, "Mask2former")
    train_image_dir = os.path.join(HOME, "Mask2former_data", "data", "semantic_views_train")
    train_mask_dir = os.path.join(HOME, "Mask2former_data", "data", "seg_masks_train")
    val_image_dir = os.path.join(HOME, "Mask2former_data", "data", "semantic_views_val")
    val_mask_dir = os.path.join(HOME, "Mask2former_data", "data", "seg_masks_val")
    save_dir = os.path.join(HOME, "Mask2former_data", "results", "models", "finetuned_model_segv4")
    log_dir = os.path.join(HOME, "Mask2former_data", "results", "tensorboard_logs_seg")

    print("=" * 60)
    print("Mask2Former Semantic Segmentation Fine-tuning (Linux)")
    print("=" * 60)
    
    print("\n[Step 1/6] Loading pretrained model...")
    global processor
    processor = Mask2FormerImageProcessor.from_pretrained(model_dir)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        model_dir,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    )
    print(f"Set num_labels to {NUM_CLASSES} (was 133 for COCO)")
    
    if hasattr(model, 'config') and CLASS_WEIGHTS is not None:
        model.config.class_weight = CLASS_WEIGHTS
        print(f"Set class weights: {CLASS_WEIGHTS}")
    
    device = get_device()
    model = model.to(device)
    
    print("\n[Step 2/6] Preparing data...")
    if not os.path.exists(train_image_dir):
        print(f"Error: Training image directory not found: {train_image_dir}")
        return
    
    if not os.path.exists(train_mask_dir):
        print(f"Error: Training mask directory not found: {train_mask_dir}")
        return
    
    print("Loading training dataset...")
    train_dataset = SegmentationDataset(train_image_dir, train_mask_dir, processor)
    
    val_dataset = None
    if os.path.exists(val_image_dir) and os.path.exists(val_mask_dir):
        print("Loading validation dataset...")
        val_dataset = SegmentationDataset(val_image_dir, val_mask_dir, processor)
    else:
        print("Warning: Validation dataset not found, skipping validation")
    
    if len(train_dataset) == 0:
        print("Error: No valid training image-mask pairs found!")
        return
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=6, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False,
        collate_fn=collate_fn,
    )
    
    val_loader = None
    if val_dataset and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset, 
            batch_size=6, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True if device.type == "cuda" else False,
            collate_fn=collate_fn,
        )
    
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
    patience = 10  # 早停耐心值：mIoU连续不提升则停止

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs,
        eta_min=1e-6
    )
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: 6")
    print(f"  Learning rate: 5e-5")
    print(f"  Image size: 1024x1024")
    print(f"  Number of classes: {NUM_CLASSES}")
    print(f"  Device: {device}")
    print(f"  Training samples: {len(train_dataset)}")
    if val_dataset:
        print(f"  Validation samples: {len(val_dataset)}")
    
    print("\nClass Legend:")
    for class_id, class_name in CLASS_NAMES.items():
        print(f"  {class_id}: {class_name} (weight: {CLASS_WEIGHTS[class_id]:.4f})")

    run_sanity_check(model, train_loader, val_loader, device)
    
    print("\n[Step 4/6] Starting fine-tuning...")
    print("=" * 60)

    # 初始化TensorBoard
    from datetime import datetime
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=os.path.join(log_dir, run_name))
    print(f"TensorBoard logs: {log_dir}/{run_name}")

    best_loss = float('inf')
    best_mIoU = 0.0
    no_improve_epochs = 0
    training_history = []

    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        epoch_start_time = time.time()
        
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)
        for batch_idx, (inputs, mask_labels, class_labels, _) in enumerate(pbar):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            mask_labels = [m.to(device) for m in mask_labels]
            class_labels = [c.to(device) for c in class_labels]

            outputs = model(
                pixel_values=inputs["pixel_values"],
                mask_labels=mask_labels,
                class_labels=class_labels
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
        
        val_accuracy = None
        val_miou = None
        val_iou_per_class = None
        
        if val_loader:
            print(f"\n  Running validation...")
            val_accuracy, val_miou, val_iou_per_class = evaluate_model(
                model, val_loader, device, NUM_CLASSES
            )
            print(f"  Validation - Acc: {val_accuracy:.4f}, mIoU: {val_miou:.4f}")
        
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_accuracy': val_accuracy,
            'val_miou': val_miou,
            'val_iou_per_class': val_iou_per_class,
            'lr': current_lr,
            'time': epoch_time
        })

        # TensorBoard记录
        writer.add_scalar('Loss/train', avg_train_loss, epoch + 1)
        writer.add_scalar('LR', current_lr, epoch + 1)
        if val_accuracy is not None:
            writer.add_scalar('Accuracy/val', val_accuracy, epoch + 1)
        if val_miou is not None:
            writer.add_scalar('mIoU/val', val_miou, epoch + 1)
        if val_iou_per_class is not None:
            for cls_id, cls_name in CLASS_NAMES.items():
                if not np.isnan(val_iou_per_class[cls_id]):
                    writer.add_scalar(f'IoU_per_class/{cls_name}', val_iou_per_class[cls_id], epoch + 1)

        print(f"Epoch [{epoch+1}/{num_epochs}] completed, Train Loss: {avg_train_loss:.4f}, LR: {current_lr:.2e}, Time: {epoch_time:.1f}s")

        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            print(f"  -> New best loss: {best_loss:.4f}")

        if val_miou is not None and val_miou > best_mIoU:
            best_mIoU = val_miou
            no_improve_epochs = 0
            # 保存最佳模型
            best_save_dir = os.path.join(save_dir, "best_model")
            os.makedirs(best_save_dir, exist_ok=True)
            model.save_pretrained(best_save_dir)
            processor.save_pretrained(best_save_dir)
            print(f"  -> New best mIoU: {best_mIoU:.4f}, model saved")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"\n  Early stopping: mIoU not improved for {patience} epochs")
                break
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("[Step 5/6] Saving model...")
    
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    
    print("\n" + "=" * 60)
    print("[Step 6/6] Training Summary")
    print("=" * 60)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Best mIoU: {best_mIoU:.4f}")
    print(f"Final loss: {training_history[-1]['train_loss']:.4f}")
    print(f"Model saved to: {save_dir}")
    
    print("\n" + "=" * 60)
    print("Training History")
    print("=" * 60)
    
    header = f"{'Epoch':<8}{'Loss':<12}{'Acc':<10}{'mIoU':<10}{'LR':<12}{'Time':<8}"
    print(header)
    print("-" * 60)
    
    for record in training_history:
        if record['epoch'] % 5 == 0 or record['epoch'] == 1:
            loss_str = f"{record['train_loss']:.4f}"
            acc_str = f"{record['val_accuracy']:.4f}" if record['val_accuracy'] else "N/A"
            miou_str = f"{record['val_miou']:.4f}" if record['val_miou'] else "N/A"
            lr_str = f"{record['lr']:.2e}"
            time_str = f"{record['time']:.1f}s"
            
            print(f"{record['epoch']:<8}{loss_str:<12}{acc_str:<10}{miou_str:<10}{lr_str:<12}{time_str:<8}")
    
    if val_loader and training_history[-1]['val_iou_per_class'] is not None:
        print("\n" + "=" * 60)
        print("Final Validation - IoU per Class")
        print("=" * 60)
        
        final_iou = training_history[-1]['val_iou_per_class']
        for cls_id, cls_name in CLASS_NAMES.items():
            iou_val = final_iou[cls_id]
            if not np.isnan(iou_val):
                print(f"  {cls_name}: {iou_val:.4f}")
            else:
                print(f"  {cls_name}: N/A")
    
    # 关闭TensorBoard
    writer.close()

    print("\n" + "=" * 60)
    print("View training curves with TensorBoard:")
    print(f"  tensorboard --logdir={log_dir}")
    print("=" * 60)

    print("\nUsage:")
    print(f'  from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor')
    print(f'  processor = Mask2FormerImageProcessor.from_pretrained("{save_dir}")')
    print(f'  model = Mask2FormerForUniversalSegmentation.from_pretrained("{save_dir}")')

if __name__ == "__main__":
    finetune()
