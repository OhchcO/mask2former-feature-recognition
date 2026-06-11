r"""
实例分割推理脚本。

用法示例：
python scripts/ins_inference_mask_utils.py --image E:\soft\code\Mask2former_data\data\semantic_views_val\000001.png --unc_image E:\soft\code\Mask2former_data\data\semantic_views_val\000001.png --face_id_image E:\soft\code\Mask2former\temp\000001.png

输出：
1. instance_mask.png：实例ID图，0为背景，每个实例一个不同ID
2. class_mask.png：类别ID图，0为背景，1-4为目标类别
3. class_map.json：实例ID到类别ID、类别名称、置信度、面积、bbox的映射
4. visualization.png：可视化结果
"""
import argparse
import json
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from face_mask_utils import extract_face_masks_from_color_image


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

CLASS_NAMES = {
    0: "Background",
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

DEFAULT_MODEL_DIR = r"E:\soft\code\Mask2former_data\results\models\finetuned_instance_model_v4"
DEFAULT_OUTPUT_DIR = r"E:\soft\code\Mask2former\results\visualizations\instance_inference"


def get_device(device_name="auto"):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def build_legend_patches():
    patches = []
    for class_id, class_name in CLASS_NAMES.items():
        if class_id == 0:
            continue
        color = CLASS_COLORS[class_id] / 255.0
        patches.append(mpatches.Patch(color=color, label=f"{class_id}={class_name}"))
    return patches


def colorize_class_mask(class_mask):
    colored = np.zeros((*class_mask.shape, 3), dtype=np.uint8)
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


def mask_to_bbox(binary_mask):
    ys, xs = np.where(binary_mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]


def build_masks_and_class_map(segmentation, segments_info, skip_background=True):
    segmentation = segmentation.astype(np.int32)
    instance_mask = np.zeros_like(segmentation, dtype=np.uint16)
    class_mask = np.zeros_like(segmentation, dtype=np.uint8)
    class_map = {}

    new_instance_id = 1
    for segment in sorted(segments_info, key=lambda x: x["id"]):
        raw_instance_id = int(segment["id"])
        class_id = int(segment["label_id"])
        score = float(segment.get("score", 0.0))

        if skip_background and class_id == 0:
            continue

        binary_mask = segmentation == raw_instance_id
        area = int(binary_mask.sum())
        if area == 0:
            continue

        instance_mask[binary_mask] = new_instance_id
        class_mask[binary_mask] = class_id
        class_map[str(new_instance_id)] = {
            "class_id": class_id,
            "class_name": CLASS_NAMES.get(class_id, f"class_{class_id}"),
            "score": score,
            "area": area,
            "bbox": mask_to_bbox(binary_mask),
            "raw_instance_id": raw_instance_id,
        }
        new_instance_id += 1

    return instance_mask, class_mask, class_map


def postprocess_instances_with_faces(raw_segmentation, segments_info, face_masks, min_ratio=0.5):
    info_by_raw_id = {int(segment["id"]): segment for segment in segments_info}
    instance_mask = np.zeros_like(raw_segmentation, dtype=np.uint16)
    class_mask = np.zeros_like(raw_segmentation, dtype=np.uint8)
    class_map = {}

    new_instance_id = 1
    for face_id, face_info in face_masks.items():
        face_mask = face_info["mask"].astype(bool)
        face_pixels = raw_segmentation[face_mask]
        face_pixels = face_pixels[face_pixels != 0]
        if len(face_pixels) == 0:
            continue

        unique_ids, counts = np.unique(face_pixels, return_counts=True)
        max_idx = int(np.argmax(counts))
        raw_instance_id = int(unique_ids[max_idx])
        ratio = float(counts[max_idx] / face_mask.sum())

        if ratio < min_ratio:
            print(f"  面 {face_id}: 主实例占比 {ratio:.2%} 低于阈值，仍使用占比最高实例")

        segment = info_by_raw_id.get(raw_instance_id)
        if segment is None:
            continue

        class_id = int(segment["label_id"])
        if class_id == 0:
            continue

        instance_mask[face_mask] = new_instance_id
        class_mask[face_mask] = class_id
        class_map[str(new_instance_id)] = {
            "class_id": class_id,
            "class_name": CLASS_NAMES.get(class_id, f"class_{class_id}"),
            "score": float(segment.get("score", 0.0)),
            "area": int(face_mask.sum()),
            "bbox": mask_to_bbox(face_mask),
            "raw_instance_id": raw_instance_id,
            "face_id": int(face_id),
            "vote_ratio": ratio,
        }
        print(f"  面 {face_id}: 实例 {new_instance_id}, 类别 {class_id}({CLASS_NAMES.get(class_id)}), 占比 {ratio:.2%}")
        new_instance_id += 1

    return instance_mask, class_mask, class_map


def save_outputs(instance_mask, class_mask, class_map, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)

    instance_mask_path = os.path.join(output_dir, f"{prefix}_instance_mask.png")
    class_mask_path = os.path.join(output_dir, f"{prefix}_class_mask.png")
    class_map_path = os.path.join(output_dir, f"{prefix}_class_map.json")

    Image.fromarray(instance_mask).save(instance_mask_path)
    Image.fromarray(class_mask).save(class_mask_path)
    with open(class_map_path, "w", encoding="utf-8") as f:
        json.dump(class_map, f, ensure_ascii=False, indent=2)

    return instance_mask_path, class_mask_path, class_map_path


def visualize(image, unc_image, raw_instance_mask, raw_class_mask, processed_instance_mask,
              processed_class_mask, output_dir, prefix, title_suffix=""):
    os.makedirs(output_dir, exist_ok=True)

    raw_instance_color = colorize_instance_mask(raw_instance_mask)
    raw_class_color = colorize_class_mask(raw_class_mask)
    legend_patches = build_legend_patches()

    raw_overlay = raw_class_color.astype(np.float32) / 255.0
    raw_overlay[raw_class_mask == 0] = np.nan

    if processed_instance_mask is None or processed_class_mask is None:
        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(raw_instance_color)
        axes[1].set_title("Instance Mask")
        axes[1].axis("off")

        axes[2].imshow(raw_class_color)
        axes[2].set_title("Class Mask")
        axes[2].axis("off")
        axes[2].legend(handles=legend_patches, loc="upper right", fontsize=9)

        axes[3].imshow(unc_image)
        axes[3].imshow(raw_overlay, alpha=0.7)
        axes[3].set_title("Overlay")
        axes[3].axis("off")
        axes[3].legend(handles=legend_patches, loc="upper right", fontsize=9)
    else:
        processed_instance_color = colorize_instance_mask(processed_instance_mask)
        processed_class_color = colorize_class_mask(processed_class_mask)
        processed_overlay = processed_class_color.astype(np.float32) / 255.0
        processed_overlay[processed_class_mask == 0] = np.nan

        fig, axes = plt.subplots(2, 4, figsize=(22, 10))
        axes[0, 0].imshow(image)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(raw_instance_color)
        axes[0, 1].set_title("Raw Instance Mask")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(raw_class_color)
        axes[0, 2].set_title("Raw Class Mask")
        axes[0, 2].axis("off")
        axes[0, 2].legend(handles=legend_patches, loc="upper right", fontsize=9)

        axes[0, 3].imshow(unc_image)
        axes[0, 3].imshow(raw_overlay, alpha=0.7)
        axes[0, 3].set_title("Raw Overlay")
        axes[0, 3].axis("off")
        axes[0, 3].legend(handles=legend_patches, loc="upper right", fontsize=9)

        axes[1, 0].imshow(image)
        axes[1, 0].set_title("Original Image")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(processed_instance_color)
        axes[1, 1].set_title("Processed Instance Mask")
        axes[1, 1].axis("off")

        axes[1, 2].imshow(processed_class_color)
        axes[1, 2].set_title("Processed Class Mask")
        axes[1, 2].axis("off")
        axes[1, 2].legend(handles=legend_patches, loc="upper right", fontsize=9)

        axes[1, 3].imshow(unc_image)
        axes[1, 3].imshow(processed_overlay, alpha=0.7)
        axes[1, 3].set_title("Processed Overlay")
        axes[1, 3].axis("off")
        axes[1, 3].legend(handles=legend_patches, loc="upper right", fontsize=9)

    if title_suffix:
        fig.suptitle(title_suffix)
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{prefix}_visualization.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def print_instance_stats(class_map, title):
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)
    print(f"检测到实例数: {len(class_map)}")

    class_counts = {}
    for instance_id, info in class_map.items():
        class_id = int(info["class_id"])
        class_counts[class_id] = class_counts.get(class_id, 0) + 1
        print(
            f"  实例 {instance_id}: 类别 {class_id}({info['class_name']}), "
            f"score={info['score']:.4f}, area={info['area']}, bbox={info['bbox']}"
        )

    print("类别实例统计:")
    for class_id, count in sorted(class_counts.items()):
        print(f"  类别 {class_id}({CLASS_NAMES.get(class_id, '未知')}): {count} 个实例")


def run_inference(image_path, unc_image_path=None, face_id_image_path=None, model_dir=DEFAULT_MODEL_DIR,
                  output_dir=DEFAULT_OUTPUT_DIR, threshold=0.5, mask_threshold=0.5,
                  min_ratio=0.5, device_name="auto"):

    print("加载实例分割模型...")
    processor = Mask2FormerImageProcessor.from_pretrained(model_dir)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_dir)
    model.eval()

    device = get_device(device_name)
    model = model.to(device)
    print(f"设备: {device}")

    print(f"加载图像: {image_path}")
    image = Image.open(image_path).convert("RGB")

    # 叠加底图：如果用户没传 unc_image，自动将原图转灰度
    if unc_image_path and os.path.exists(unc_image_path):
        unc_image = Image.open(unc_image_path).convert("RGB")
    else:
        unc_image = image.convert("L").convert("RGB")

    inputs = processor(images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print("运行实例分割推理...")
    with torch.no_grad():
        outputs = model(**inputs)

    result = processor.post_process_instance_segmentation(
        outputs,
        target_sizes=[image.size[::-1]],
        threshold=threshold,
        mask_threshold=mask_threshold,
    )[0]

    raw_segmentation = result["segmentation"]
    if isinstance(raw_segmentation, torch.Tensor):
        raw_segmentation = raw_segmentation.cpu().numpy()
    segments_info = result["segments_info"]

    raw_instance_mask, raw_class_mask, raw_class_map = build_masks_and_class_map(
        raw_segmentation, segments_info
    )

    prefix = os.path.splitext(os.path.basename(image_path))[0]
    raw_output_dir = os.path.join(output_dir, "raw")
    raw_paths = save_outputs(raw_instance_mask, raw_class_mask, raw_class_map, raw_output_dir, prefix)

    processed_instance_mask = None
    processed_class_mask = None
    processed_class_map = None
    processed_paths = None

    if face_id_image_path:
        if os.path.exists(face_id_image_path):
            print("\n开始面ID图后处理...")
            face_masks = extract_face_masks_from_color_image(face_id_image_path)
            processed_instance_mask, processed_class_mask, processed_class_map = postprocess_instances_with_faces(
                raw_segmentation, segments_info, face_masks, min_ratio=min_ratio
            )
            processed_output_dir = os.path.join(output_dir, "processed")
            processed_paths = save_outputs(
                processed_instance_mask, processed_class_mask, processed_class_map, processed_output_dir, prefix
            )
        else:
            print(f"警告: 面ID图不存在，跳过后处理: {face_id_image_path}")

    visualization_path = visualize(
        image=image,
        unc_image=unc_image,
        raw_instance_mask=raw_instance_mask,
        raw_class_mask=raw_class_mask,
        processed_instance_mask=processed_instance_mask,
        processed_class_mask=processed_class_mask,
        output_dir=output_dir,
        prefix=prefix,
    )

    print_instance_stats(raw_class_map, "原始实例分割结果")
    if processed_class_map is not None:
        print_instance_stats(processed_class_map, "面ID图后处理实例分割结果")

    print("\n结果已保存:")
    print(f"  原始实例ID图: {raw_paths[0]}")
    print(f"  原始类别ID图: {raw_paths[1]}")
    print(f"  原始class_map: {raw_paths[2]}")
    if processed_paths:
        print(f"  后处理实例ID图: {processed_paths[0]}")
        print(f"  后处理类别ID图: {processed_paths[1]}")
        print(f"  后处理class_map: {processed_paths[2]}")
    print(f"  可视化图: {visualization_path}")

    return {
        "raw_instance_mask": raw_instance_mask,
        "raw_class_mask": raw_class_mask,
        "raw_class_map": raw_class_map,
        "processed_instance_mask": processed_instance_mask,
        "processed_class_mask": processed_class_mask,
        "processed_class_map": processed_class_map,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mask2Former Instance Segmentation Inference")
    parser.add_argument("--image", type=str, required=True, help="输入图像路径")
    parser.add_argument("--unc_image", type=str, default=None, help="用于叠加显示的无色/原始图像路径，默认等于 --image")
    parser.add_argument("--face_id_image", type=str, default=None, help="面ID图路径，不同颜色表示不同面；传入后会按面做实例后处理")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR, help="微调后的实例分割模型目录")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="输出目录")
    parser.add_argument("--threshold", type=float, default=0.5, help="实例置信度阈值")
    parser.add_argument("--mask_threshold", type=float, default=0.5, help="mask二值化阈值")
    parser.add_argument("--min_ratio", type=float, default=0.5, help="面ID图后处理时主实例投票阈值")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="推理设备")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"图像不存在: {args.image}")
    elif not os.path.exists(args.model_dir):
        print(f"模型目录不存在: {args.model_dir}")
    else:
        run_inference(
            image_path=args.image,
            unc_image_path=args.unc_image,
            face_id_image_path=args.face_id_image,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            threshold=args.threshold,
            mask_threshold=args.mask_threshold,
            min_ratio=args.min_ratio,
            device_name=args.device,
        )
