# 根据类型计算类权重
import os
import numpy as np
from PIL import Image

CLASS_NAMES = {
    0: "背景",
    1: "宽体槽",
    2: "封闭槽",
    3: "开放槽",
    4: "孔"
}

BACKGROUND_VALUE = 255

def calculate_class_weights(mask_dir, num_classes=5):
    print("=" * 50)
    print("Calculating Class Weights")
    print("=" * 50)
    print(f"Background mapping: {BACKGROUND_VALUE} -> 0")

    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
    print(f"\nFound {len(mask_files)} mask files")
    
    total_pixels = 0
    class_counts = np.zeros(num_classes)
    
    for mask_file in mask_files:
        mask_path = os.path.join(mask_dir, mask_file)
        mask = np.array(Image.open(mask_path).convert("L"))
        mapped_mask = mask.copy()
        mapped_mask[mask == BACKGROUND_VALUE] = 0

        for cls in range(num_classes):
            class_counts[cls] += np.sum(mapped_mask == cls)
        
        total_pixels += mask.size
    
    print("\nClass Distribution:")
    print(f"{'Class':<20}{'Pixels':<15}{'Percentage':<15}")
    print("-" * 50)
    
    for cls in range(num_classes):
        percentage = class_counts[cls] / total_pixels * 100
        print(f"{CLASS_NAMES[cls]:<20}{int(class_counts[cls]):<15}{percentage:.2f}%")
    
    class_percentages = class_counts / total_pixels
    present_classes = class_counts > 0

    median_freq = np.median(class_percentages[present_classes])
    weights = np.zeros(num_classes, dtype=np.float64)
    weights[present_classes] = median_freq / class_percentages[present_classes]
    weights = weights / weights[present_classes].sum() * present_classes.sum()
    
    print("\nCalculated Weights (Median Frequency Balancing):")
    print(f"{'Class':<20}{'Weight':<15}")
    print("-" * 35)
    
    for cls in range(num_classes):
        print(f"{CLASS_NAMES[cls]:<20}{weights[cls]:.4f}")
    
    print("\nInverse Frequency Weights:")
    inv_weights = np.zeros(num_classes, dtype=np.float64)
    inv_weights[present_classes] = 1.0 / class_percentages[present_classes]
    inv_weights = inv_weights / inv_weights[present_classes].sum() * present_classes.sum()
    
    print(f"{'Class':<20}{'Weight':<15}")
    print("-" * 35)
    
    for cls in range(num_classes):
        print(f"{CLASS_NAMES[cls]:<20}{inv_weights[cls]:.4f}")
    
    print("\nRecommended weights for training:")
    print(f"CLASS_WEIGHTS = {weights.tolist()}")
    
    return weights

if __name__ == "__main__":
    train_mask_dir = r"E:\soft\code\Mask2former_data\data\seg_masks_train"

    if os.path.exists(train_mask_dir):
        weights = calculate_class_weights(train_mask_dir)
    else:
        print(f"Error: Mask directory not found: {train_mask_dir}")