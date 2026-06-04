"""
按零件划分训练集和验证集

每12个图片为一个零件，同一个零件的所有图片放在同一集合。

用法：
    python split_dataset.py --image_dir train/new_images_png --mask_dir train/new_masks_png --output_dir train --val_ratio 0.2 --images_per_part 12
"""

import os
import shutil
import argparse
import random


def split_by_part(image_dir, mask_dir, output_dir, val_ratio=0.2, images_per_part=12, seed=42):
    """
    按零件划分训练集和验证集
    
    Args:
        image_dir: 图像目录
        mask_dir: 掩码目录
        output_dir: 输出目录
        val_ratio: 验证集比例
        images_per_part: 每个零件的图片数（增强前）
        seed: 随机种子
    """
    random.seed(seed)
    
    # 输出目录
    train_image_dir = os.path.join(output_dir, "train_images_png")
    train_mask_dir = os.path.join(output_dir, "train_masks_png")
    val_image_dir = os.path.join(output_dir, "val_images_png")
    val_mask_dir = os.path.join(output_dir, "val_masks_png")
    
    for d in [train_image_dir, train_mask_dir, val_image_dir, val_mask_dir]:
        os.makedirs(d, exist_ok=True)
    
    # 获取所有图片
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    
    # 按零件分组
    # 增强后的文件名格式: 000001_0.png, 000001_1.png, ...
    # 原始编号 000001-000012 是第一个零件
    parts = {}
    for img_name in image_files:
        # 提取原始编号（增强前的编号）
        base_name = img_name.split('_')[0]  # 000001
        
        # 计算属于哪个零件
        part_id = (int(base_name) - 1) // images_per_part
        
        if part_id not in parts:
            parts[part_id] = []
        parts[part_id].append(img_name)
    
    print(f"找到 {len(image_files)} 张图片")
    print(f"共 {len(parts)} 个零件")
    print(f"每个零件约 {images_per_part} 张原始图片")
    
    # 划分零件
    part_ids = list(parts.keys())
    random.shuffle(part_ids)
    
    val_count = max(1, int(len(part_ids) * val_ratio))
    val_parts = part_ids[:val_count]
    train_parts = part_ids[val_count:]
    
    print(f"\n划分结果:")
    print(f"  训练集零件: {len(train_parts)} 个")
    print(f"  验证集零件: {len(val_parts)} 个")
    
    # 统计
    train_count = 0
    val_count = 0
    
    # 复制训练集
    for part_id in train_parts:
        for img_name in parts[part_id]:
            # 复制图像
            src_img = os.path.join(image_dir, img_name)
            dst_img = os.path.join(train_image_dir, img_name)
            shutil.copy2(src_img, dst_img)
            
            # 复制掩码
            src_mask = os.path.join(mask_dir, img_name)
            dst_mask = os.path.join(train_mask_dir, img_name)
            if os.path.exists(src_mask):
                shutil.copy2(src_mask, dst_mask)
            
            train_count += 1
    
    # 复制验证集
    for part_id in val_parts:
        for img_name in parts[part_id]:
            # 复制图像
            src_img = os.path.join(image_dir, img_name)
            dst_img = os.path.join(val_image_dir, img_name)
            shutil.copy2(src_img, dst_img)
            
            # 复制掩码
            src_mask = os.path.join(mask_dir, img_name)
            dst_mask = os.path.join(val_mask_dir, img_name)
            if os.path.exists(src_mask):
                shutil.copy2(src_mask, dst_mask)
            
            val_count += 1
    
    print(f"\n文件统计:")
    print(f"  训练集: {train_count} 张图片")
    print(f"  验证集: {val_count} 张图片")
    
    print(f"\n输出目录:")
    print(f"  训练图像: {train_image_dir}")
    print(f"  训练掩码: {train_mask_dir}")
    print(f"  验证图像: {val_image_dir}")
    print(f"  验证掩码: {val_mask_dir}")
    
    # 打印零件分配详情
    print(f"\n零件分配详情:")
    for part_id in sorted(parts.keys()):
        part_images = parts[part_id]
        first_img = part_images[0].split('_')[0]
        last_img = part_images[-1].split('_')[0]
        set_type = "验证" if part_id in val_parts else "训练"
        print(f"  零件 {part_id + 1}: {first_img}-{last_img} ({len(part_images)} 张) -> {set_type}集")


def main():
    parser = argparse.ArgumentParser(description="按零件划分训练集和验证集")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="图像目录")
    parser.add_argument("--mask_dir", type=str, required=True,
                        help="掩码目录")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="验证集比例（默认：0.2）")
    parser.add_argument("--images_per_part", type=int, default=12,
                        help="每个零件的原始图片数（默认：12）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（默认：42）")
    
    args = parser.parse_args()
    
    # 检查目录
    if not os.path.exists(args.image_dir):
        print(f"错误: 图像目录不存在: {args.image_dir}")
        return
    
    if not os.path.exists(args.mask_dir):
        print(f"错误: 掩码目录不存在: {args.mask_dir}")
        return
    
    print("=" * 50)
    print("按零件划分训练集和验证集")
    print("=" * 50)
    print(f"图像目录: {args.image_dir}")
    print(f"掩码目录: {args.mask_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"验证集比例: {args.val_ratio}")
    print(f"每零件图片数: {args.images_per_part}")
    print(f"随机种子: {args.seed}")
    print("=" * 50)
    
    split_by_part(
        args.image_dir,
        args.mask_dir,
        args.output_dir,
        args.val_ratio,
        args.images_per_part,
        args.seed
    )
    
    print("\n完成!")


if __name__ == "__main__":
    main()
