# 下载mask2former模型
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
import os

save_dir = r"E:\soft\code\Mask2former"
os.makedirs(save_dir, exist_ok=True)

print("正在下载模型...")
processor = Mask2FormerImageProcessor.from_pretrained(
    "facebook/mask2former-swin-tiny-coco-panoptic",
    cache_dir=save_dir
)
model = Mask2FormerForUniversalSegmentation.from_pretrained(
    "facebook/mask2former-swin-tiny-coco-panoptic",
    cache_dir=save_dir
)

print("正在保存到指定目录...")
processor.save_pretrained(save_dir)
model.save_pretrained(save_dir)

print(f"✅ 下载完成！保存位置: {save_dir}")