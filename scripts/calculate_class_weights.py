# 根据类型计算类权重
import os
import numpy as np
from PIL import Image

LABEL_MAPPING = {
    255: 0,
    0: 1,
    1: 2,
    2: 3,
    3: 4
}

CLASS_NAMES = {
    0: "Background",
    1: "Class 0",
    2: "Class 1",
    3: "Class 2",
    4: "Class 3"
}

def calculate_class_weights(mask_dir, num_classes=5):
    print("=" * 50)
    print("Calculating Class Weights (with Label Mapping)")
    print("=" * 50)
    
    print("\nLabel Mapping:")
    for src, dst in LABEL_MAPPING.items():
        print(f"  {src} -> {dst} ({CLASS_NAMES[dst]})")
    
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
    print(f"\nFound {len(mask_files)} mask files")
    
    total_pixels = 0
    class_counts = np.zeros(num_classes)
    
    for mask_file in mask_files:
        mask_path = os.path.join(mask_dir, mask_file)
        mask = np.array(Image.open(mask_path).convert("L"))
        
        mapped_mask = np.zeros_like(mask)
        for src_val, dst_val in LABEL_MAPPING.items():
            mapped_mask[mask == src_val] = dst_val
        
        for cls in range(num_classes):
            class_counts[cls] += np.sum(mapped_mask == cls)
        
        total_pixels += mask.size
    
    print("\nClass Distribution (after mapping):")
    print(f"{'Class':<20}{'Pixels':<15}{'Percentage':<15}")
    print("-" * 50)
    
    for cls in range(num_classes):
        percentage = class_counts[cls] / total_pixels * 100
        print(f"{CLASS_NAMES[cls]:<20}{int(class_counts[cls]):<15}{percentage:.2f}%")
    
    class_percentages = class_counts / total_pixels
    
    median_freq = np.median(class_percentages)
    weights = median_freq / class_percentages
    
    weights = weights / weights.sum() * num_classes
    
    print("\nCalculated Weights (Median Frequency Balancing):")
    print(f"{'Class':<20}{'Weight':<15}")
    print("-" * 35)
    
    for cls in range(num_classes):
        print(f"{CLASS_NAMES[cls]:<20}{weights[cls]:.4f}")
    
    print("\nInverse Frequency Weights:")
    inv_weights = 1.0 / (class_percentages + 1e-8)
    inv_weights = inv_weights / inv_weights.sum() * num_classes
    
    print(f"{'Class':<20}{'Weight':<15}")
    print("-" * 35)
    
    for cls in range(num_classes):
        print(f"{CLASS_NAMES[cls]:<20}{inv_weights[cls]:.4f}")
    
    print("\nRecommended weights for training:")
    print(f"CLASS_WEIGHTS = {weights.tolist()}")
    
    return weights

if __name__ == "__main__":
    train_mask_dir = r"E:\soft\code\Mask2former\train\masks_png"
    
    if os.path.exists(train_mask_dir):
        weights = calculate_class_weights(train_mask_dir)
    else:
        print(f"Error: Mask directory not found: {train_mask_dir}")