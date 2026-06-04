# 单纯推理，不使用投票后处理
import torch
import numpy as np
from PIL import Image
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

def run_inference(image_path):
    model_dir = r"E:\soft\code\Mask2former\results\models\finetuned_model_v3"
    
    print("加载微调后的模型...")
    processor = Mask2FormerImageProcessor.from_pretrained(model_dir)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_dir)
    model.eval()
    
    device = torch.device("cuda")
    model = model.to(device)
    print(f"设备: {device}")
    
    print(f"加载图像: {image_path}")    
    device = torch.device("cuda")
    image = Image.open(image_path).convert("RGB")
    
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

    colored_seg = colorize_segmentation(semantic_seg)
    overlay_seg = colored_seg.astype(np.float32) / 255.0
    overlay_seg[semantic_seg == 0] = np.nan
    legend_patches = build_legend_patches()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    axes[1].imshow(colored_seg)
    axes[1].set_title("Segmentation (Foreground Only)")
    axes[1].axis("off")
    axes[1].legend(handles=legend_patches, loc="upper right", fontsize=9)
    
    axes[2].imshow(image)
    axes[2].imshow(overlay_seg, alpha=0.5)
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    axes[2].legend(handles=legend_patches, loc="upper right", fontsize=9)
    
    plt.tight_layout()
    output_path = r"E:\soft\code\Mask2former\results\visualizations\inference_result.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print("\n" + "="*50)
    print("像素值分析")
    print("="*50)
    
    unique_values = np.unique(semantic_seg)
    print(f"检测到的类别ID: {unique_values}")
    
    total_pixels = semantic_seg.size
    print(f"总像素数: {total_pixels}")
    
    for value in unique_values:
        count = np.sum(semantic_seg == value)
        percentage = (count / total_pixels) * 100
        class_name = CLASS_NAMES.get(int(value), "背景")
        print(f"  类别 {value} ({class_name}): {count} 像素 ({percentage:.2f}%)")
    
    print(f"\n结果已保存到: {output_path}")
    
    return semantic_seg

if __name__ == "__main__":
    test_image = r"E:\soft\code\Mask2former\train\images_png\000001.png"
    # test_image = r"E:\soft\code\Mask2former\test\2.png"
    import os
    if os.path.exists(test_image):
        result = run_inference(test_image)
    else:
        print(f"图像不存在: {test_image}")
        print("请修改 test_image 变量指向你的测试图像")
