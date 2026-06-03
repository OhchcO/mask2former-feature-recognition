# 可视化标签掩码结果，验证标签的正确性
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

def visualize_samples(image_dir, mask_dir, output_dir, num_samples=20):
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])[:num_samples]
    
    print(f"Visualizing {len(image_files)} samples...")
    
    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, img_file)
        
        if not os.path.exists(mask_path):
            print(f"Skipping {img_file}: mask not found")
            continue
        
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        
        mapped_mask = np.zeros_like(mask)
        for src_val, dst_val in LABEL_MAPPING.items():
            mapped_mask[mask == src_val] = dst_val
        
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
        
        colored_mapped = np.zeros((*mapped_mask.shape, 3), dtype=np.uint8)
        for val, color in MAPPED_COLORS.items():
            colored_mapped[mapped_mask == val] = color
        axes[2].imshow(colored_mapped)
        axes[2].set_title("Mapped Mask")
        axes[2].axis("off")
        
        legend_patches = []
        for val, color in MAPPED_COLORS.items():
            if val > 0:
                color_norm = [c/255 for c in color]
                legend_patches.append(mpatches.Patch(color=color_norm, label=f"{val}: {CLASS_NAMES[val]}"))
        if legend_patches:
            axes[2].legend(handles=legend_patches, loc='upper right', fontsize=8)
        
        original_values = np.unique(mask)
        mapped_values = np.unique(mapped_mask)
        
        plt.suptitle(f"{idx+1}/{len(image_files)}: {img_file}\nOriginal: {original_values} -> Mapped: {mapped_values}", fontsize=10)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f"mapping_{idx+1:03d}_{img_file}.png")
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close()
        
        print(f"[{idx+1}/{len(image_files)}] {img_file}: {original_values} -> {mapped_values}")
    
    print(f"\nDone! Saved to {output_dir}")

if __name__ == "__main__":
    image_dir = r"E:\soft\code\Mask2former\train\images_png"
    mask_dir = r"E:\soft\code\Mask2former\train\masks_png"
    output_dir = r"E:\soft\code\Mask2former\results\visualizations\mapping_check"
    
    visualize_samples(image_dir, mask_dir, output_dir, num_samples=20)