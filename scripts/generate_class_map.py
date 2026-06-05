# -*- coding: utf-8 -*-
"""
随机生成 class_map.json
"""
import os
import json
import random

# 类别定义
CLASSES = {
    0: "宽体槽",
    1: "封闭槽",
    2: "开放槽",
    3: "孔"
}

# 实例ID列表（用户提供的）
INSTANCE_IDS = list(range(1, 25)) + [255]  # 1-24, 255

# 实例掩码目录
INSTANCE_MASK_DIR = r"E:\soft\code\Mask2former\temp"

def generate_random_class_map():
    """随机生成class_map"""
    class_map = {}

    # 获取所有实例掩码文件
    mask_files = sorted([f for f in os.listdir(INSTANCE_MASK_DIR) if f.endswith('.png')])

    for mask_file in mask_files:
        # 为每个文件随机生成类别映射
        file_mapping = {}
        for inst_id in INSTANCE_IDS:
            # 随机分配类别（0-3）
            class_id = random.randint(0, 3)
            file_mapping[str(inst_id)] = class_id

        class_map[mask_file] = file_mapping
        print(f"{mask_file}: {len(file_mapping)} instances")

    return class_map


if __name__ == "__main__":
    # 生成随机class_map
    class_map = generate_random_class_map()

    # 保存到文件
    output_path = r"E:\soft\code\Mask2former\train\class_map.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(class_map, f, indent=2, ensure_ascii=False)

    print(f"\nGenerated class_map.json with {len(class_map)} files")
    print(f"Saved to: {output_path}")
