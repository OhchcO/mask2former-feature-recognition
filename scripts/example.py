# 例子，确保mask2former模型加载成功
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt

model_dir = r"E:\soft\code\Mask2former"

print("正在加载模型...")
processor = Mask2FormerImageProcessor.from_pretrained(model_dir)
model = Mask2FormerForUniversalSegmentation.from_pretrained(model_dir)

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"模型已加载到: {device}")

def panoptic_segmentation(image_path):
    image = Image.open(image_path).convert("RGB")
    
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    result = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    
    return image, result

def visualize_result(image, result):
    panoptic_seg = result["segmentation"]
    segments_info = result["segments_info"]
    
    if isinstance(panoptic_seg, torch.Tensor):
        panoptic_seg = panoptic_seg.cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    axes[1].imshow(panoptic_seg)
    axes[1].set_title("Panoptic Segmentation")
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.savefig("segmentation_result.png", dpi=150, bbox_inches="tight")
    plt.show()
    
    print("\n分割结果详情:")
    for segment in segments_info:
        print(f"  类别ID: {segment['label_id']}, 置信度: {segment['score']:.3f}")

def download_sample_image():
    import urllib.request
    import os
    
    sample_url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    sample_path = "sample_image.jpg"
    
    if not os.path.exists(sample_path):
        print("正在下载示例图像...")
        try:
            urllib.request.urlretrieve(sample_url, sample_path)
            print(f"示例图像已保存到: {os.path.abspath(sample_path)}")
        except Exception as e:
            print(f"下载失败: {e}")
            print("请手动将图像放到当前目录")
            return None
    
    return sample_path

if __name__ == "__main__":
    import os
    
    test_image_path = "test_image.jpg"
    
    if not os.path.exists(test_image_path):
        print("未找到 test_image.jpg，将使用示例图像...")
        test_image_path = download_sample_image()
    
    if test_image_path and os.path.exists(test_image_path):
        print(f"\n使用图像: {os.path.abspath(test_image_path)}")
        image, result = panoptic_segmentation(test_image_path)
        visualize_result(image, result)
    else:
        print("\n请将测试图像放到当前目录，或修改代码中的图像路径")
        print("示例用法:")
        print("  image, result = panoptic_segmentation('your_image.jpg')")
        print("  visualize_result(image, result)")