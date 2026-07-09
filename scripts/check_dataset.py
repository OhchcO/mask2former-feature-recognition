"""
数据集质量检查脚本
检查项：
1. 图片和掩码是否一一对应
2. 掩码是否为空（全 255）
3. 掩码里的 instance_id 是否在 class_map 中有对应类别
4. class_map 中的类别是否在 CLASS_ID_MAP 中
5. 掩码像素值是否合法（0-254）
"""
import os
import json
import numpy as np
from PIL import Image
from collections import Counter, defaultdict

# ============================================================
# 配置（和 config.py 保持一致）
# ============================================================
import sys
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, INSTANCE_CLASS_MAP_PATH,
    NUM_CLASSES, CLASS_NAMES, CLASS_ID_MAP
)

# ============================================================
# 加载 class_map
# ============================================================
print("=" * 60)
print("数据集质量检查")
print("=" * 60)

with open(INSTANCE_CLASS_MAP_PATH, 'r') as f:
    class_map = json.load(f)

print(f"\n[配置信息]")
print(f"  训练图片目录: {TRAIN_IMAGE_DIR}")
print(f"  训练掩码目录: {TRAIN_MASK_DIR}")
print(f"  class_map: {INSTANCE_CLASS_MAP_PATH}")
print(f"  类别数: {NUM_CLASSES}")
print(f"  类别名: {CLASS_NAMES}")
print(f"  CLASS_ID_MAP: {CLASS_ID_MAP}")
print(f"  class_map 包含图片数: {len(class_map)}")

# ============================================================
# 扫描
# ============================================================
image_files = sorted([f for f in os.listdir(TRAIN_IMAGE_DIR) if f.endswith('.png')])
mask_files = sorted([f for f in os.listdir(TRAIN_MASK_DIR) if f.endswith('.png')])

print(f"\n[文件统计]")
print(f"  图片文件数: {len(image_files)}")
print(f"  掩码文件数: {len(mask_files)}")

# 统计容器
issues = defaultdict(list)  # issue_type -> [filename, ...]
empty_masks = []
no_mask_images = []
class_map_missing = []      # class_map 里没有的图片
instance_id_not_in_map = []  # 掩码里的 instance_id 在 class_map 里找不到
class_id_not_in_map = []     # class_id 在 CLASS_ID_MAP 里找不到
valid_images = []
class_counter = Counter()
instance_counter = Counter()

print(f"\n[开始逐个检查...]")
for i, img_name in enumerate(image_files):
    if (i + 1) % 1000 == 0:
        print(f"  已检查 {i + 1}/{len(image_files)}...")

    # 1. 掩码是否存在
    if img_name not in set(mask_files):
        no_mask_images.append(img_name)
        continue

    mask_path = os.path.join(TRAIN_MASK_DIR, img_name)

    # 2. class_map 是否包含该图片
    if img_name not in class_map:
        class_map_missing.append(img_name)
        continue

    img_class_map = class_map[img_name]

    # 3. 掩码是否为空
    mask_arr = np.array(Image.open(mask_path).convert("L"))
    instance_ids = sorted([int(x) for x in np.unique(mask_arr) if x < 255])

    if len(instance_ids) == 0:
        empty_masks.append(img_name)
        continue

    # 4. 检查每个 instance_id
    has_valid_instance = False
    for inst_id in instance_ids:
        instance_counter[inst_id] += 1

        # instance_id 在 class_map 里？
        if str(inst_id) not in img_class_map:
            instance_id_not_in_map.append((img_name, inst_id))
            continue

        class_id = int(img_class_map[str(inst_id)])

        # class_id 在 CLASS_ID_MAP 里？
        if class_id not in CLASS_ID_MAP:
            class_id_not_in_map.append((img_name, inst_id, class_id))
            continue

        class_counter[class_id] += 1
        has_valid_instance = True

    if has_valid_instance:
        valid_images.append(img_name)

# ============================================================
# 输出报告
# ============================================================
print("\n" + "=" * 60)
print("检查报告")
print("=" * 60)

print(f"\n[结果汇总]")
print(f"  总图片数: {len(image_files)}")
print(f"  有效图片（可训练）: {len(valid_images)}")
print(f"  空掩码（无特征）: {len(empty_masks)}")
print(f"  无掩码文件: {len(no_mask_images)}")
print(f"  class_map 缺失: {len(class_map_missing)}")
print(f"  instance_id 不在 class_map: {len(instance_id_not_in_map)}")
print(f"  class_id 不在 CLASS_ID_MAP: {len(class_id_not_in_map)}")

if class_counter:
    print(f"\n[各类别实例统计]")
    for cls_id in sorted(class_counter.keys()):
        cls_name = CLASS_NAMES.get(cls_id, f"unknown_{cls_id}")
        print(f"  {cls_name} (id={cls_id}): {class_counter[cls_id]} 个实例")

# 详细列出问题
if empty_masks:
    print(f"\n[空掩码图片] (前 20 个)")
    for name in empty_masks[:20]:
        print(f"  {name}")
    if len(empty_masks) > 20:
        print(f"  ... 还有 {len(empty_masks) - 20} 个")

if no_mask_images:
    print(f"\n[无掩码文件的图片] (前 20 个)")
    for name in no_mask_images[:20]:
        print(f"  {name}")

if class_map_missing:
    print(f"\n[class_map 缺失的图片] (前 20 个)")
    for name in class_map_missing[:20]:
        print(f"  {name}")

if instance_id_not_in_map:
    print(f"\n[instance_id 在 class_map 中找不到] (前 20 个)")
    for img_name, inst_id in instance_id_not_in_map[:20]:
        print(f"  {img_name}: instance_id={inst_id}")

if class_id_not_in_map:
    print(f"\n[class_id 不在 CLASS_ID_MAP 中] (前 20 个)")
    for img_name, inst_id, class_id in class_id_not_in_map[:20]:
        print(f"  {img_name}: instance_id={inst_id}, class_id={class_id}")

# 诊断结论
print("\n" + "=" * 60)
print("诊断结论")
print("=" * 60)

total_issues = (len(empty_masks) + len(no_mask_images) + len(class_map_missing) +
                len(instance_id_not_in_map) + len(class_id_not_in_map))

if total_issues == 0:
    print("  数据集完全正常，无任何问题！")
else:
    print(f"  共发现 {total_issues} 个问题")
    if empty_masks:
        print(f"  - {len(empty_masks)} 张空掩码图（训练时自动跳过，不影响训练）")
    if no_mask_images:
        print(f"  - {len(no_mask_images)} 张图没有掩码文件（需补充或删除）")
    if class_map_missing:
        print(f"  - {len(class_map_missing)} 张图不在 class_map 中（需补充）")
    if instance_id_not_in_map:
        print(f"  - {len(instance_id_not_in_map)} 个 instance_id 在 class_map 中找不到（可能导致 cost matrix infeasible）")
    if class_id_not_in_map:
        print(f"  - {len(class_id_not_in_map)} 个 class_id 不在 CLASS_ID_MAP 中（类别不匹配）")

print("\n" + "=" * 60)
