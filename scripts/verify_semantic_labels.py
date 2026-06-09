# -*- coding: utf-8 -*-
"""
语义标签验证
输入：语义灰度掩码文件夹
输出：染色可视化图片
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 颜色定义（与verify_instance_labels.py一致）
SEMANTIC_COLORS = {
    0: [255, 255, 255],  # 背景 - 白色
    1: [255, 165, 0],    # 宽体槽 - 橙色
    2: [128, 0, 128],    # 封闭槽 - 紫色
    3: [0, 255, 255],    # 开放槽 - 青色
    4: [255, 0, 0],      # 孔 - 红色
}

SEMANTIC_NAMES = {
    0: "背景",
    1: "宽体槽",
    2: "封闭槽",
    3: "开放槽",
    4: "孔",
}


def colorize_semantic_mask(semantic_mask):
    """将灰度语义掩码转为RGB染色图"""
    h, w = semantic_mask.shape
    color_img = np.full((h, w, 3), 255, dtype=np.uint8)  # 默认白色背景
    for class_id, color in SEMANTIC_COLORS.items():
        color_img[semantic_mask == class_id] = color
    return color_img


def verify_semantic_labels(input_dir, output_dir):
    """批量验证语义标签"""
    os.makedirs(output_dir, exist_ok=True)

    mask_files = sorted([f for f in os.listdir(input_dir)
                         if f.endswith('.png') and os.path.isfile(os.path.join(input_dir, f))])
    print(f"Found {len(mask_files)} semantic masks in {input_dir}")

    for mask_file in mask_files:
        mask_path = os.path.join(input_dir, mask_file)
        semantic_mask = np.array(Image.open(mask_path).convert("L"))

        # 染色
        color_img = colorize_semantic_mask(semantic_mask)

        # 统计类别
        unique_ids = np.unique(semantic_mask)

        # 绘图：左原图右染色
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        ax1.imshow(semantic_mask, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax1.set_title("Original Semantic Mask", fontsize=12, fontweight='bold')
        ax1.axis("off")

        ax2.imshow(color_img, interpolation="nearest")
        ax2.set_title("Colorized", fontsize=12, fontweight='bold')
        ax2.axis("off")

        # 图例
        legend_handles = []
        for cid in sorted(unique_ids):
            cid = int(cid)
            if cid in SEMANTIC_NAMES:
                color_norm = [c / 255 for c in SEMANTIC_COLORS[cid]]
                count = np.sum(semantic_mask == cid)
                legend_handles.append(
                    mpatches.Patch(color=color_norm, label=f"{SEMANTIC_NAMES[cid]} (id={cid}, {count}px)")
                )
        ax2.legend(handles=legend_handles, loc='lower right', fontsize=9,
                   title="Semantic Class", title_fontsize=10)

        fig.suptitle(f"Semantic Label Verification: {mask_file}", fontsize=13, y=0.98)
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"verify_{mask_file}")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_path}")

    print(f"\nDone! Verified {len(mask_files)} masks -> {output_dir}")


if __name__ == "__main__":
    # 设置输入文件夹（包含语义掩码PNG）和输出文件夹
    input_dir = r"E:\soft\code\Mask2former_data\temp"
    output_dir = r"E:\soft\code\Mask2former_data\results\verify_seg_labels"

    verify_semantic_labels(input_dir, output_dir)
