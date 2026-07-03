"""
批量实例分割推理
# 基本用法(结果保存在 input_dir/inference_results/)
python scripts/ins_batch_inference.py --input_dir D:\Code\AAGNet\test_datasets\balanced_dataset\train\encoded_views

# 指定模型和阈值
python scripts/ins_batch_inference.py --input_dir /path/to/images --model_dir /path/to/model --threshold 0.3

# 指定输出目录
python scripts/ins_batch_inference.py --input_dir /path/to/images --output_dir /path/to/results
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

sys.path.insert(0, os.path.dirname(__file__))
from config import (  # type: ignore
    CLASS_NAMES, NUM_CLASSES, MODEL_DIR,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# 类别颜色
COLOR_PALETTE = [
    np.array([255, 0, 0]), np.array([0, 255, 0]), np.array([0, 0, 255]),
    np.array([255, 255, 0]), np.array([255, 0, 255]), np.array([0, 255, 255]),
    np.array([128, 0, 255]), np.array([255, 128, 0]),
]
CLASS_COLORS = {cls_id: COLOR_PALETTE[i % len(COLOR_PALETTE)]
                for i, cls_id in enumerate(CLASS_NAMES.keys())}


def mask_to_bbox(binary_mask):
    ys, xs = np.where(binary_mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]


def mask_to_coco_rle(binary_mask):
    from pycocotools import mask as coco_mask
    rle = coco_mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def colorize_class_mask(class_mask):
    colored = np.full((*class_mask.shape, 3), 255, dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        colored[class_mask == class_id] = color
    return colored


def colorize_instance_mask(instance_mask):
    colored = np.zeros((*instance_mask.shape, 3), dtype=np.uint8)
    instance_ids = [int(x) for x in np.unique(instance_mask) if x != 0]
    for instance_id in instance_ids:
        rng = np.random.default_rng(instance_id)
        colored[instance_mask == instance_id] = rng.integers(30, 256, size=3, dtype=np.uint8)
    return colored


def process_single_image(image_path, model, processor, device, threshold=0.5, mask_threshold=0.5):
    """处理单张图片，返回结果字典"""
    image_pil = Image.open(image_path).convert("RGB")

    inputs = processor(images=image_pil, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    result = processor.post_process_instance_segmentation(
        outputs,
        target_sizes=[image_pil.size[::-1]],
        threshold=threshold,
        mask_threshold=mask_threshold,
    )[0]

    raw_segmentation = result["segmentation"]
    if isinstance(raw_segmentation, torch.Tensor):
        raw_segmentation = raw_segmentation.cpu().numpy()
    segments_info = result["segments_info"]

    # 构建结果
    instance_mask = np.zeros_like(raw_segmentation, dtype=np.uint16)
    class_mask = np.full_like(raw_segmentation, 255, dtype=np.uint8)
    class_map = {}

    new_id = 1
    for segment in sorted(segments_info, key=lambda x: x["id"]):
        raw_id = int(segment["id"])
        class_id = int(segment["label_id"])
        score = float(segment.get("score", 0.0))
        if score < threshold:
            continue
        bin_mask = raw_segmentation == raw_id
        area = int(bin_mask.sum())
        if area == 0:
            continue
        instance_mask[bin_mask] = new_id
        class_mask[bin_mask] = class_id
        class_map[str(new_id)] = {
            "class_id": class_id,
            "class_name": CLASS_NAMES.get(class_id, f"class_{class_id}"),
            "score": round(score, 4),
            "area": area,
            "bbox": mask_to_bbox(bin_mask),
        }
        new_id += 1

    return {
        "image_pil": image_pil,
        "instance_mask": instance_mask,
        "class_mask": class_mask,
        "class_map": class_map,
    }


def save_results(image_name, result, output_dir):
    """保存单张图片的推理结果"""
    prefix = os.path.splitext(image_name)[0]

    # 保存 mask
    Image.fromarray(result["instance_mask"]).save(
        os.path.join(output_dir, f"{prefix}_instance_mask.png"))
    Image.fromarray(result["class_mask"]).save(
        os.path.join(output_dir, f"{prefix}_class_mask.png"))

    # 保存 class_map.json
    with open(os.path.join(output_dir, f"{prefix}_class_map.json"), "w", encoding="utf-8") as f:
        json.dump(result["class_map"], f, ensure_ascii=False, indent=2)

    # 保存可视化
    instance_color = colorize_instance_mask(result["instance_mask"])
    class_color = colorize_class_mask(result["class_mask"])

    legend_patches = []
    for cls_id, cls_name in CLASS_NAMES.items():
        color = CLASS_COLORS[cls_id] / 255.0
        legend_patches.append(mpatches.Patch(color=color, label=f"{cls_id}={cls_name}"))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].imshow(result["image_pil"])
    axes[0].set_title("Input")
    axes[0].axis("off")
    axes[1].imshow(instance_color)
    axes[1].set_title("Instance Mask")
    axes[1].axis("off")
    axes[2].imshow(class_color)
    axes[2].set_title("Class Mask")
    axes[2].axis("off")
    axes[2].legend(handles=legend_patches, loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_visualization.png"), dpi=120, bbox_inches="tight")
    plt.close()


def batch_inference(input_dir, model_dir=None, output_dir=None,
                    threshold=0.5, mask_threshold=0.5, device_name="auto"):
    """批量推理：遍历 input_dir 下所有 .png 图片"""
    if model_dir is None:
        model_dir = MODEL_DIR
    if output_dir is None:
        output_dir = os.path.join(input_dir, "inference_results")

    # 收集图片
    image_files = sorted([f for f in os.listdir(input_dir)
                          if f.endswith('.png') and os.path.isfile(os.path.join(input_dir, f))])
    if not image_files:
        print(f"未找到 .png 图片: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 加载模型
    print(f"加载模型: {model_dir}")
    processor = Mask2FormerImageProcessor.from_pretrained(model_dir)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_dir)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() and device_name != "cpu" else "cpu")
    model = model.to(device)
    print(f"设备: {device}")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"图片数量: {len(image_files)}")
    print(f"阈值: threshold={threshold}, mask_threshold={mask_threshold}")

    # 汇总统计
    total_instances = 0
    class_counts = defaultdict(int)

    for img_name in tqdm(image_files, desc="推理中"):
        img_path = os.path.join(input_dir, img_name)
        result = process_single_image(img_path, model, processor, device, threshold, mask_threshold)
        save_results(img_name, result, output_dir)

        n_inst = len(result["class_map"])
        total_instances += n_inst
        for info in result["class_map"].values():
            class_counts[info["class_id"]] += 1

    # 打印汇总
    print(f"\n{'=' * 50}")
    print(f"批量推理完成")
    print(f"{'=' * 50}")
    print(f"处理图片: {len(image_files)} 张")
    print(f"检测实例: {total_instances} 个")
    print(f"类别统计:")
    for cls_id in sorted(CLASS_NAMES.keys()):
        print(f"  {cls_id} ({CLASS_NAMES[cls_id]}): {class_counts[cls_id]} 个")
    print(f"结果保存在: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mask2Former 批量实例分割推理")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="输入图片文件夹")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="模型目录（默认使用 config 中的 MODEL_DIR）")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录（默认: input_dir/inference_results）")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="实例置信度阈值")
    parser.add_argument("--mask_threshold", type=float, default=0.5,
                        help="mask二值化阈值")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])

    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"目录不存在: {args.input_dir}")
    else:
        batch_inference(
            input_dir=args.input_dir,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            threshold=args.threshold,
            mask_threshold=args.mask_threshold,
            device_name=args.device,
        )
