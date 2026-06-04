"""
推理时，输入2张图片，一张是原始图片，一张是面ID图，输出是原始图片的分割图

用法：
python scripts/inference_mask_utils.py --image E:\soft\code\Mask2former\train\images_png\000001.png --image E:\soft\code\Mask2former\train\images_png\000001.png --face_id_image E:\soft\code\Mask2former\temp\000001.png
"""
import torch
import numpy as np
from PIL import Image
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from face_mask_utils import (
    process_image_with_annotations,
    load_color_json,
    create_face_masks_from_color_json,
    extract_face_masks_from_color_image,
    voting_postprocess,
    clip_segmentation_to_masks
)


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

CLASS_NAMES = {
    1: "宽体槽",
    2: "封闭槽",
    3: "开放槽",
    4: "孔",
}

CLASS_COLORS = {
    1: np.array([255, 0, 0], dtype=np.uint8),
    2: np.array([0, 255, 0], dtype=np.uint8),
    3: np.array([0, 0, 255], dtype=np.uint8),
    4: np.array([255, 255, 0], dtype=np.uint8),
}

# 投票后处理配置
ENABLE_VOTING = True  # 是否启用投票后处理
ANNOTATION_JSON = r"E:\soft\code\Mask2former\temp\000001.json"  # 标注文件路径
IMAGE_ID = 1  # 当前图像对应的ID（需要根据实际情况修改）
VOTING_MIN_RATIO = 0.5  # 投票阈值


def colorize_segmentation(segmentation):
    colored = np.zeros((*segmentation.shape, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        colored[segmentation == class_id] = color
    return colored


def build_legend_patches():
    patches = []
    for class_id, class_name in CLASS_NAMES.items():
        color = CLASS_COLORS[class_id] / 255.0
        patches.append(mpatches.Patch(color=color, label=f"{class_id}={class_name}"))
    return patches

def run_inference(image_path, unc_image_path, enable_voting=None, annotation_json=None, image_id=None, min_ratio=None, face_id_image_path=None):
    # model_dir = r"E:\soft\code\Mask2former\results\models\finetuned_model_v3"
    model_dir = r"E:\soft\code\Mask2former\linux_models\finetuned_model_v4"
    # 使用全局配置或传入参数
    if enable_voting is None:
        enable_voting = ENABLE_VOTING
    if annotation_json is None:
        annotation_json = ANNOTATION_JSON
    if image_id is None:
        image_id = IMAGE_ID
    if min_ratio is None:
        min_ratio = VOTING_MIN_RATIO

    print("加载微调后的模型...")
    processor = Mask2FormerImageProcessor.from_pretrained(model_dir)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_dir)
    model.eval()

    device = torch.device("cuda")
    model = model.to(device)
    print(f"设备: {device}")

    print(f"加载图像: {image_path}")
    print(f"加载无色图像: {unc_image_path}")
    image = Image.open(image_path).convert("RGB")
    unc_image = Image.open(unc_image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print("运行推理...")
    with torch.no_grad():
        outputs = model(**inputs)

    semantic_seg = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )[0]

    if isinstance(semantic_seg, torch.Tensor):
        semantic_seg = semantic_seg.cpu().numpy()

    # 投票后处理
    processed_seg = semantic_seg.copy()
    if enable_voting:
        import os
        import json

        # 优先使用面ID图（直接从图像提取面信息）
        if face_id_image_path and os.path.exists(face_id_image_path):
            print(f"使用面ID图: {face_id_image_path}")
            face_masks = extract_face_masks_from_color_image(face_id_image_path)
            processed_seg = voting_postprocess(semantic_seg, face_masks, min_ratio)
            processed_seg = clip_segmentation_to_masks(processed_seg, face_masks)
        elif os.path.exists(annotation_json):
            # 检测JSON格式
            with open(annotation_json, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            if 'faces' in json_data:
                # 颜色JSON格式（从color_to_json.py生成）
                print("检测到颜色JSON格式")
                faces, json_image_size = load_color_json(annotation_json)
                image_size = semantic_seg.shape[:2]
                face_masks = create_face_masks_from_color_json(faces, image_size)
                processed_seg = voting_postprocess(semantic_seg, face_masks, min_ratio)
                processed_seg = clip_segmentation_to_masks(processed_seg, face_masks)
            else:
                # COCO格式
                print("检测到COCO JSON格式")
                processed_seg = process_image_with_annotations(
                    semantic_seg, annotation_json, image_id, min_ratio
                )
        else:
            print(f"警告: 未找到面ID图或标注文件, 跳过投票后处理")

    # 可视化对比
    colored_seg = colorize_segmentation(semantic_seg)
    colored_processed = colorize_segmentation(processed_seg)

    overlay_seg = colored_seg.astype(np.float32) / 255.0
    overlay_seg[semantic_seg == 0] = np.nan

    overlay_processed = colored_processed.astype(np.float32) / 255.0
    overlay_processed[processed_seg == 0] = np.nan

    legend_patches = build_legend_patches()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 第一行：原始分割结果
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(colored_seg)
    axes[0, 1].set_title("Raw Segmentation")
    axes[0, 1].axis("off")
    axes[0, 1].legend(handles=legend_patches, loc="upper right", fontsize=9)

    axes[0, 2].imshow(unc_image)
    axes[0, 2].imshow(overlay_seg, alpha=1)
    axes[0, 2].set_title("Raw Overlay")
    axes[0, 2].axis("off")
    axes[0, 2].legend(handles=legend_patches, loc="upper right", fontsize=9)

    # 第二行：投票后处理结果
    axes[1, 0].imshow(image)
    axes[1, 0].set_title("Original Image")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(colored_processed)
    axes[1, 1].set_title("Processed Segmentation (Voting)")
    axes[1, 1].axis("off")
    axes[1, 1].legend(handles=legend_patches, loc="upper right", fontsize=9)

    axes[1, 2].imshow(unc_image)
    axes[1, 2].imshow(overlay_processed, alpha=1)
    axes[1, 2].set_title("Processed Overlay (Voting)")
    axes[1, 2].axis("off")
    axes[1, 2].legend(handles=legend_patches, loc="upper right", fontsize=9)

    plt.tight_layout()
    output_path = r"E:\soft\code\Mask2former\results\visualizations\inference_result.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    # 打印分析结果
    print("\n" + "="*50)
    print("像素值分析 (原始)")
    print("="*50)
    print_stats(semantic_seg)

    if enable_voting:
        print("\n" + "="*50)
        print("像素值分析 (投票后处理)")
        print("="*50)
        print_stats(processed_seg)

    print(f"\n结果已保存到: {output_path}")

    return processed_seg if enable_voting else semantic_seg


def print_stats(segmentation):
    unique_values = np.unique(segmentation)
    print(f"检测到的类别ID: {unique_values}")

    total_pixels = segmentation.size
    print(f"总像素数: {total_pixels}")

    for value in unique_values:
        count = np.sum(segmentation == value)
        percentage = (count / total_pixels) * 100
        class_name = CLASS_NAMES.get(int(value), "背景")
        print(f"  类别 {value} ({class_name}): {count} 像素 ({percentage:.2f}%)")

if __name__ == "__main__":
    import argparse, os

    parser = argparse.ArgumentParser(description="Mask2Former Inference with Voting Postprocess")
    parser.add_argument("--unc_image", type=str, default=None,
                        help="输入图像路径（未处理图像）")
    parser.add_argument("--image", type=str, default=r"E:\soft\code\Mask2former\train\images_png\000001.png",
                        help="输入图像路径（原始图像）")
    parser.add_argument("--face_id_image", type=str, default=None,
                        help="面ID图路径（不同颜色表示不同面，优先使用）")
    parser.add_argument("--json", type=str, default=ANNOTATION_JSON,
                        help="标注JSON文件路径（备选）")
    parser.add_argument("--image_id", type=int, default=IMAGE_ID,
                        help="图像ID（对应JSON中的image_id）")
    parser.add_argument("--min_ratio", type=float, default=VOTING_MIN_RATIO,
                        help="投票阈值")
    parser.add_argument("--no_voting", action="store_true",
                        help="禁用投票后处理")

    args = parser.parse_args()

    if os.path.exists(args.image):
        result = run_inference(
            args.image,
            args.unc_image,
            enable_voting=not args.no_voting,
            annotation_json=args.json,
            image_id=args.image_id,
            min_ratio=args.min_ratio,
            face_id_image_path=args.face_id_image
        )
    else:
        print(f"图像不存在: {args.image}")
        print("请修改 --image 参数指向你的测试图像")
