"""
数据增强脚本

对图像和掩码同时进行相同的几何变换，保持一一对应。
颜色变换只应用于图像，不应用于掩码。

用法：
    python augment_data.py --image_dir path/to/temp --mask_dir path/to/temp_mask --output_image_dir path/to/aug_images --output_mask_dir path/to/aug_masks --num_augment 6
"""

import os
import cv2
import numpy as np
import argparse
import random


def random_rotate(image, mask, max_angle=30):
    """随机旋转"""
    angle = random.uniform(-max_angle, max_angle)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    image_rot = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    mask_rot = cv2.warpAffine(mask, M, (w, h), cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
    return image_rot, mask_rot


def random_flip(image, mask):
    """随机翻转（水平/垂直/both）"""
    flip_type = random.choice([0, 1, -1])
    image_flip = cv2.flip(image, flip_type)
    mask_flip = cv2.flip(mask, flip_type)
    return image_flip, mask_flip


def random_scale(image, mask, scale_range=(0.8, 1.2)):
    """随机缩放"""
    scale = random.uniform(*scale_range)
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    
    image_scaled = cv2.resize(image, (new_w, new_h))
    mask_scaled = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    # 裁剪或填充到原尺寸
    if scale >= 1.0:
        # 裁剪中心区域
        start_h = (new_h - h) // 2
        start_w = (new_w - w) // 2
        image_scaled = image_scaled[start_h:start_h + h, start_w:start_w + w]
        mask_scaled = mask_scaled[start_h:start_h + h, start_w:start_w + w]
    else:
        # 填充到原尺寸
        pad_h = (h - new_h) // 2
        pad_w = (w - new_w) // 2
        image_scaled = cv2.copyMakeBorder(image_scaled, pad_h, h - new_h - pad_h, 
                                           pad_w, w - new_w - pad_w, cv2.BORDER_REFLECT)
        mask_scaled = cv2.copyMakeBorder(mask_scaled, pad_h, h - new_h - pad_h, 
                                          pad_w, w - new_w - pad_w, cv2.BORDER_REFLECT)
    
    return image_scaled, mask_scaled


def random_brightness_contrast(image, brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2)):
    """随机亮度和对比度调整（只对图像）"""
    brightness = random.uniform(*brightness_range)
    contrast = random.uniform(*contrast_range)
    
    image = image.astype(np.float32)
    image = image * contrast + (brightness - 1) * 128
    image = np.clip(image, 0, 255).astype(np.uint8)
    
    return image


def random_noise(image, noise_level=10):
    """随机噪声（只对图像）"""
    noise = np.random.normal(0, noise_level, image.shape).astype(np.float32)
    image = image.astype(np.float32) + noise
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def augment_pair(image, mask, augment_type):
    """
    对图像和掩码进行指定类型的增强
    
    Args:
        image: 原始图像
        mask: 原始掩码
        augment_type: 增强类型编号 (1-6)
    
    Returns:
        augmented_image, augmented_mask
    """
    if augment_type == 1:
        # 水平翻转
        return cv2.flip(image, 1), cv2.flip(mask, 1)
    
    elif augment_type == 2:
        # 垂直翻转
        return cv2.flip(image, 0), cv2.flip(mask, 0)
    
    elif augment_type == 3:
        # 随机旋转
        return random_rotate(image, mask, max_angle=20)
    
    elif augment_type == 4:
        # 随机缩放 + 亮度调整
        img_s, mask_s = random_scale(image, mask, scale_range=(0.9, 1.1))
        img_s = random_brightness_contrast(img_s)
        return img_s, mask_s
    
    elif augment_type == 5:
        # 旋转 + 翻转
        img_r, mask_r = random_rotate(image, mask, max_angle=15)
        return cv2.flip(img_r, 1), cv2.flip(mask_r, 1)
    
    elif augment_type == 6:
        # 缩放 + 噪声
        img_s, mask_s = random_scale(image, mask, scale_range=(0.85, 1.15))
        img_s = random_noise(img_s, noise_level=8)
        return img_s, mask_s
    
    else:
        return image.copy(), mask.copy()


def augment_dataset(image_dir, mask_dir, output_image_dir, output_mask_dir, num_augment=6):
    """
    对整个数据集进行增强
    
    Args:
        image_dir: 原始图像目录
        mask_dir: 原始掩码目录
        output_image_dir: 增强后图像输出目录
        output_mask_dir: 增强后掩码输出目录
        num_augment: 每张图像的增强次数
    """
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    
    # 获取图像列表
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"找到 {len(image_files)} 张图像")
    print(f"每张图像生成 {num_augment} 个增强样本")
    print(f"预计生成 {len(image_files) * num_augment} 个样本")
    
    count = 0
    for img_name in image_files:
        # 加载图像和掩码
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name)
        
        if not os.path.exists(mask_path):
            print(f"警告: 掩码不存在 {mask_path}, 跳过")
            continue
        
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            print(f"警告: 无法加载 {img_name}, 跳过")
            continue
        
        # 保存原始图像
        base_name = os.path.splitext(img_name)[0]
        cv2.imwrite(os.path.join(output_image_dir, f"{base_name}_0.png"), image)
        cv2.imwrite(os.path.join(output_mask_dir, f"{base_name}_0.png"), mask)
        count += 1
        
        # 生成增强样本
        for aug_id in range(1, num_augment + 1):
            aug_image, aug_mask = augment_pair(image, mask, aug_id)
            
            aug_name = f"{base_name}_{aug_id}.png"
            cv2.imwrite(os.path.join(output_image_dir, aug_name), aug_image)
            cv2.imwrite(os.path.join(output_mask_dir, aug_name), aug_mask)
            count += 1
        
        print(f"  处理: {img_name} -> {num_augment + 1} 个样本")
    
    print(f"\n完成! 共生成 {count} 个样本")


def main():
    parser = argparse.ArgumentParser(description="数据增强脚本")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="原始图像目录")
    parser.add_argument("--mask_dir", type=str, required=True,
                        help="原始掩码目录")
    parser.add_argument("--output_image_dir", type=str, required=True,
                        help="增强后图像输出目录")
    parser.add_argument("--output_mask_dir", type=str, required=True,
                        help="增强后掩码输出目录")
    parser.add_argument("--num_augment", type=int, default=6,
                        help="每张图像的增强次数（默认：6）")
    
    args = parser.parse_args()
    
    # 检查目录是否存在
    if not os.path.exists(args.image_dir):
        print(f"错误: 图像目录不存在: {args.image_dir}")
        return
    
    if not os.path.exists(args.mask_dir):
        print(f"错误: 掩码目录不存在: {args.mask_dir}")
        return
    
    print("=" * 50)
    print("数据增强工具")
    print("=" * 50)
    print(f"图像目录: {args.image_dir}")
    print(f"掩码目录: {args.mask_dir}")
    print(f"输出图像目录: {args.output_image_dir}")
    print(f"输出掩码目录: {args.output_mask_dir}")
    print(f"增强次数: {args.num_augment}")
    print("=" * 50)
    
    augment_dataset(
        args.image_dir,
        args.mask_dir,
        args.output_image_dir,
        args.output_mask_dir,
        args.num_augment
    )


if __name__ == "__main__":
    main()
