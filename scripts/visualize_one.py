# 可视化单个样本的标签掩码结果
import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

ORIGINAL_COLORS = {
    0: [255, 0, 0],
    1: [0, 255, 0],
    2: [0, 0, 255],
    3: [255, 255, 0],
    255: [0, 0, 0]
}

MAPPED_COLORS = {
    0: [0, 0, 0],
    1: [255, 0, 0],
    2: [0, 255, 0],
    3: [0, 0, 255],
    4: [255, 255, 0]
}

def visualize_one_sample(image_dir, mask_dir, img_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    img_path = os.path.join(image_dir, img_file)
    mask_path = os.path.join(mask_dir, img_file)
    
    if not os.path.exists(img_path):
        print(f"Error: Image not found: {img_path}")
        return
    
    if not os.path.exists(mask_path):
        print(f"Error: Mask not found: {mask_path}")
        return
    
    image = np.array(Image.open(img_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L"))
    
    mapped_mask = np.zeros_like(mask)
    for src_val, dst_val in LABEL_MAPPING.items():
        mapped_mask[mask == src_val] = dst_val
    
    original_values = np.unique(mask)
    mapped_values = np.unique(mapped_mask)
    
    print(f"File: {img_file}")
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Original pixel values: {original_values}")
    print(f"Mapped pixel values: {mapped_values}")
    
    print("\nOriginal value counts:")
    for val in original_values:
        count = np.sum(mask == val)
        percentage = count / mask.size * 100
        print(f"  {val}: {count} pixels ({percentage:.2f}%)")
    
    print("\nMapped value counts:")
    for val in mapped_values:
        count = np.sum(mapped_mask == val)
        percentage = count / mapped_mask.size * 100
        print(f"  {val} ({CLASS_NAMES[val]}): {count} pixels ({percentage:.2f}%)")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    colored_original = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for val, color in ORIGINAL_COLORS.items():
        colored_original[mask == val] = color
    axes[1].imshow(colored_original)
    axes[1].set_title("Original Mask")
    axes[1].axis("off")
    
    legend_orig = []
    for val in original_values:
        if val in ORIGINAL_COLORS:
            color_norm = [c/255 for c in ORIGINAL_COLORS[val]]
            if val == 255:
                legend_orig.append(mpatches.Patch(color=color_norm, label=f"{val} (Background)"))
            else:
                legend_orig.append(mpatches.Patch(color=color_norm, label=f"{val}"))
    if legend_orig:
        axes[1].legend(handles=legend_orig, loc='upper right', fontsize=8)
    
    colored_mapped = np.zeros((*mapped_mask.shape, 3), dtype=np.uint8)
    for val, color in MAPPED_COLORS.items():
        colored_mapped[mapped_mask == val] = color
    axes[2].imshow(colored_mapped)
    axes[2].set_title("Mapped Mask")
    axes[2].axis("off")
    
    legend_mapped = []
    for val in mapped_values:
        if val in MAPPED_COLORS and val > 0:
            color_norm = [c/255 for c in MAPPED_COLORS[val]]
            legend_mapped.append(mpatches.Patch(color=color_norm, label=f"{val}: {CLASS_NAMES[val]}"))
    if legend_mapped:
        axes[2].legend(handles=legend_mapped, loc='upper right', fontsize=8)
    
    plt.suptitle(f"Sample: {img_file}\nOriginal: {original_values} -> Mapped: {mapped_values}", fontsize=12)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"sample_{img_file}.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"\nSaved to: {output_path}")

if __name__ == "__main__":
    image_dir = r"E:\soft\code\Mask2former\train\images_png"
    mask_dir = r"E:\soft\code\Mask2former\train\masks_png"
    output_dir = r"E:\soft\code\Mask2former\results\visualizations\mapping_check"
    
    img_file = "000066.png"
    visualize_one_sample(image_dir, mask_dir, img_file, output_dir)