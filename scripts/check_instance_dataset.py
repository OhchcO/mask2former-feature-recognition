# -*- coding: utf-8 -*-
r"""
检查实例分割数据集一致性：
1. 统计 train/val 每类实例数量与面积
2. 检查实例 mask 中的实例 ID 是否都能在 class_map.json 中找到
3. 检查 class_map.json 中的实例 ID 是否都在实例 mask 中出现
4. 检查类别 ID 是否合法
5. 单张图详细诊断（--image）
"""
import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np
from PIL import Image


CLASS_NAMES = {
    0: "背景",
    1: "宽体槽",
    2: "封闭槽",
    3: "开放槽",
    4: "孔",
}

VALID_CLASS_IDS = set(CLASS_NAMES.keys()) - {0}
BACKGROUND_VALUE = 255


def load_class_map(class_map_path):
    with open(class_map_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_mask_instance_ids(mask):
    return sorted(int(x) for x in np.unique(mask) if int(x) != BACKGROUND_VALUE)


def format_class_counts(class_counts):
    lines = []
    for class_id in sorted(VALID_CLASS_IDS):
        lines.append(f"  {class_id} {CLASS_NAMES[class_id]}: {class_counts[class_id]} 个实例")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 单张图详细诊断
# ---------------------------------------------------------------------------
def inspect_single_image(mask_path, class_map_path, detail_image_name):
    print("\n" + "=" * 70)
    print(f"单图详细诊断: {detail_image_name}")
    print("=" * 70)
    print(f"mask_path: {mask_path}")
    print(f"class_map_path: {class_map_path}")

    if not os.path.exists(mask_path):
        print(f"[ERROR] mask 不存在: {mask_path}")
        return
    if not os.path.exists(class_map_path):
        print(f"[ERROR] class_map 不存在: {class_map_path}")
        return

    class_map = load_class_map(class_map_path)

    if detail_image_name not in class_map:
        print(f"[ERROR] class_map 中没有 {detail_image_name} 的条目")
        print(f"  class_map 中前10个key: {list(class_map.keys())[:10]}")
        return

    img_class_map = class_map[detail_image_name]
    mask = np.array(Image.open(mask_path).convert("L"))

    # 全图灰度值统计（含背景）
    all_unique, all_counts = np.unique(mask, return_counts=True)
    total_pixels = mask.size
    print(f"\n全图灰度值统计 (共 {len(all_unique)} 种，尺寸 {mask.shape[1]}x{mask.shape[0]})：")
    print(f"  {'Gray':<6}{'Name':<10}{'Pixels':<12}{'Percentage':<12}")
    print(f"  {'-' * 45}")
    for val, cnt in zip(all_unique, all_counts):
        val = int(val)
        name = "背景" if val == BACKGROUND_VALUE else (
            CLASS_NAMES.get(int(img_class_map.get(str(val), -1)), "未映射")
        )
        pct = cnt / total_pixels * 100
        print(f"  {val:<6}{name:<10}{cnt:<12}{pct:.4f}%")

    # mask 实例 ID 集合
    mask_instance_ids = get_mask_instance_ids(mask)
    mask_id_set = set(mask_instance_ids)
    map_id_set = set(int(x) for x in img_class_map.keys())

    print(f"\nmask 实例ID数量: {len(mask_id_set)}")
    print(f"class_map 实例ID数量: {len(map_id_set)}")

    # mask 有、class_map 没有
    missing_in_map = sorted(mask_id_set - map_id_set)
    # class_map 有、mask 没有
    missing_in_mask = sorted(map_id_set - mask_id_set)
    # 共同的
    common_ids = sorted(mask_id_set & map_id_set)

    if missing_in_map:
        print(f"\n[缺失映射] mask 中有 {len(missing_in_map)} 个实例ID在 class_map 中缺失:")
        for inst_id in missing_in_map:
            area = int(np.sum(mask == inst_id))
            print(f"  ID {inst_id}: {area} px")
    else:
        print("\n[OK] mask 中所有实例ID都有 class_map 映射")

    if missing_in_mask:
        print(f"\n[缺失像素] class_map 中有 {len(missing_in_mask)} 个实例ID在 mask 中没有像素:")
        for inst_id in missing_in_mask:
            class_id = int(img_class_map[str(inst_id)])
            print(f"  ID {inst_id} -> 类别 {class_id} ({CLASS_NAMES.get(class_id, '未知')})")
    else:
        print("\n[OK] class_map 中所有实例ID在 mask 中都有像素")

    # 共同的实例详情
    if common_ids:
        print(f"\n共同实例详情 ({len(common_ids)} 个)：")
        print(f"  {'InstID':<8}{'Class':<10}{'Area(px)':<12}{'MaskRange':<16}")
        print(f"  {'-' * 46}")
        for inst_id in common_ids:
            class_id = int(img_class_map[str(inst_id)])
            area = int(np.sum(mask == inst_id))
            ys, xs = np.where(mask == inst_id)
            mask_range = f"[{int(xs.min())},{int(ys.min())}]-[{int(xs.max())},{int(ys.max())}]" if len(xs) else "none"
            print(f"  {inst_id:<8}{CLASS_NAMES.get(class_id, '未知'):<10}{area:<12}{mask_range:<16}")

    # 按类别统计
    print("\n类别统计:")
    class_dist = defaultdict(int)
    for inst_id_str, class_id_raw in img_class_map.items():
        inst_id = int(inst_id_str)
        class_id = int(class_id_raw)
        if inst_id in mask_id_set:
            class_dist[class_id] += 1
    for class_id in sorted(VALID_CLASS_IDS):
        print(f"  {class_id} {CLASS_NAMES[class_id]}: {class_dist[class_id]} 个实例")

    # 诊断建议
    print("\n诊断建议:")
    if len(missing_in_map) > 0:
        print(f"  - mask 中有 {len(missing_in_map)} 个ID无类别映射，检查class_map生成逻辑")
        max_missing = min(missing_in_map)
        if max_missing < 5 and len(mask_instance_ids) > 5:
            print(f"    疑似：前{max_missing+1}个实例被当做background? 检查生成脚本的ID起始值")
    if len(missing_in_mask) > 0:
        print(f"  - class_map 中有 {len(missing_in_mask)} 个ID无像素")
        print(f"    可能：class_map 为其他图像生成，或mask被重采样/裁剪过")
    if BACKGROUND_VALUE in all_unique and BACKGROUND_VALUE == 255:
        bg_pct = int(all_counts[all_unique == BACKGROUND_VALUE][0]) / total_pixels * 100
        print(f"  - 背景={BACKGROUND_VALUE}，占 {bg_pct:.1f}%")
    if 0 in all_unique and np.sum(mask == 0) > 0:
        id0_area = int(np.sum(mask == 0))
        print(f"  - mask 中灰度值 0 有 {id0_area} px，这是实例还是背景需确认")

    # 检查是否有不在0-254或255范围的值
    out_of_range = [v for v in all_unique if not (0 <= v <= 254 or v == BACKGROUND_VALUE)]
    if out_of_range:
        print(f"  - 出现异常灰度值: {out_of_range}")


# ---------------------------------------------------------------------------
# 数据集批量检查
# ---------------------------------------------------------------------------
def check_split(split_name, image_dir, mask_dir, class_map_path, detail_limit=30):
    print("\n" + "=" * 70)
    print(f"检查 {split_name}")
    print("=" * 70)
    print(f"image_dir: {image_dir}")
    print(f"mask_dir: {mask_dir}")
    print(f"class_map: {class_map_path}")

    errors = []
    warnings = []
    class_counts = defaultdict(int)
    class_area = defaultdict(int)
    image_class_counts = defaultdict(int)
    missing_in_class_map_total = 0
    missing_in_mask_total = 0
    invalid_class_total = 0
    total_instances_in_mask = 0
    total_instances_in_class_map = 0

    # 统计"缺失"ID的分布：哪些ID最常缺失
    missing_in_map_counter = Counter()
    missing_in_mask_counter = Counter()
    # 统计灰度值中出现了多少非实例/非背景的值
    out_of_range_total = 0
    images_with_zero = 0   # mask中出现0的图像数
    images_zero_as_bg = 0  # 0被当做背景（即class_map中有"0"=0）的图像数

    if not os.path.exists(image_dir):
        errors.append(f"图像目录不存在: {image_dir}")
    if not os.path.exists(mask_dir):
        errors.append(f"实例 mask 目录不存在: {mask_dir}")
    if not os.path.exists(class_map_path):
        errors.append(f"class_map 不存在: {class_map_path}")

    if errors:
        for error in errors:
            print(f"[ERROR] {error}")
        return

    class_map = load_class_map(class_map_path)
    image_files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(".png"))
    mask_files = sorted(f for f in os.listdir(mask_dir) if f.lower().endswith(".png"))
    mask_file_set = set(mask_files)

    print(f"图像数量: {len(image_files)}")
    print(f"mask数量: {len(mask_files)}")
    print(f"class_map图像条目数: {len(class_map)}")

    detail_printed = 0

    for image_name in image_files:
        mask_path = os.path.join(mask_dir, image_name)
        if image_name not in mask_file_set:
            warnings.append(f"缺少 mask: {image_name}")
            continue

        if image_name not in class_map:
            warnings.append(f"class_map 缺少图像条目: {image_name}")
            img_class_map = {}
        else:
            img_class_map = class_map[image_name]

        mask = np.array(Image.open(mask_path).convert("L"))
        mask_instance_ids = get_mask_instance_ids(mask)
        mask_id_set = set(mask_instance_ids)
        map_id_set = set(int(x) for x in img_class_map.keys())

        total_instances_in_mask += len(mask_id_set)
        total_instances_in_class_map += len(map_id_set)

        missing_in_class_map = sorted(mask_id_set - map_id_set)
        missing_in_mask = sorted(map_id_set - mask_id_set)

        # 统计缺失ID频次
        for mid in missing_in_class_map:
            missing_in_map_counter[mid] += 1
        for mid in missing_in_mask:
            missing_in_mask_counter[mid] += 1

        if missing_in_class_map:
            missing_in_class_map_total += len(missing_in_class_map)
            if detail_printed < detail_limit:
                ids_str = ", ".join(str(x) for x in missing_in_class_map[:15])
                more = "" if len(missing_in_class_map) <= 15 else f" ...(+{len(missing_in_class_map)-15})"
                print(f"[WARN] {image_name}: mask实例ID不在class_map中 [{len(missing_in_class_map)}个]: {ids_str}{more}")
                detail_printed += 1

        if missing_in_mask:
            missing_in_mask_total += len(missing_in_mask)
            if detail_printed < detail_limit:
                ids_str = ", ".join(str(x) for x in missing_in_mask[:15])
                more = "" if len(missing_in_mask) <= 15 else f" ...(+{len(missing_in_mask)-15})"
                print(f"[WARN] {image_name}: class_map实例ID不在mask中 [{len(missing_in_mask)}个]: {ids_str}{more}")
                detail_printed += 1

        # 统计异常灰度值
        all_unique = np.unique(mask)
        out_of_range_vals = [v for v in all_unique if not (0 <= v <= 254 or v == BACKGROUND_VALUE)]
        if out_of_range_vals:
            out_of_range_total += 1
            if detail_printed < detail_limit:
                print(f"[WARN] {image_name}: 异常灰度值 {out_of_range_vals}")
                detail_printed += 1

        # 统计0的出现情况
        if 0 in all_unique:
            images_with_zero += 1
            if "0" in img_class_map and int(img_class_map["0"]) == 0:
                images_zero_as_bg += 1

        image_classes = set()
        for inst_id_str, class_id_raw in img_class_map.items():
            inst_id = int(inst_id_str)
            class_id = int(class_id_raw)

            if class_id not in VALID_CLASS_IDS:
                invalid_class_total += 1
                if detail_printed < detail_limit:
                    print(f"[WARN] {image_name}: 实例 {inst_id} 类别ID非法: {class_id}")
                    detail_printed += 1
                continue

            if inst_id not in mask_id_set:
                continue

            area = int(np.sum(mask == inst_id))
            class_counts[class_id] += 1
            class_area[class_id] += area
            image_classes.add(class_id)

        for class_id in image_classes:
            image_class_counts[class_id] += 1

    extra_masks = sorted(mask_file_set - set(image_files))
    for mask_name in extra_masks[:detail_limit]:
        warnings.append(f"mask没有对应图像: {mask_name}")

    print("\n类别实例数量:")
    print(format_class_counts(class_counts))

    print("\n类别出现图像数量:")
    for class_id in sorted(VALID_CLASS_IDS):
        print(f"  {class_id} {CLASS_NAMES[class_id]}: {image_class_counts[class_id]} 张图")

    print("\n类别实例面积:")
    for class_id in sorted(VALID_CLASS_IDS):
        avg_area = class_area[class_id] / class_counts[class_id] if class_counts[class_id] > 0 else 0
        print(
            f"  {class_id} {CLASS_NAMES[class_id]}: total={class_area[class_id]} px, "
            f"avg={avg_area:.1f} px"
        )

    print("\n一致性汇总:")
    print(f"  mask中实例总数: {total_instances_in_mask}")
    print(f"  class_map中实例总数: {total_instances_in_class_map}")
    print(f"  mask实例ID缺class_map映射: {missing_in_class_map_total}")
    print(f"  class_map实例ID缺mask像素: {missing_in_mask_total}")
    print(f"  非法类别ID数量: {invalid_class_total}")
    print(f"  额外mask文件数量: {len(extra_masks)}")
    print(f"  其他警告数量: {len(warnings)}")
    print(f"  异常灰度值图像数: {out_of_range_total}")
    print(f"  mask中出现灰度0的图像数: {images_with_zero}")

    zero_classes = [class_id for class_id in sorted(VALID_CLASS_IDS) if class_counts[class_id] == 0]
    if zero_classes:
        print("\n[重点] 以下类别没有任何有效实例:")
        for class_id in zero_classes:
            print(f"  {class_id} {CLASS_NAMES[class_id]}")

    # 最常缺失的ID
    if missing_in_map_counter:
        print(f"\n最常缺失class_map映射的实例ID (top 15):")
        for inst_id, freq in missing_in_map_counter.most_common(15):
            print(f"  ID {inst_id}: 缺失 {freq} 次")
    if missing_in_mask_counter:
        print(f"\n最常缺失mask像素的实例ID (top 15):")
        for inst_id, freq in missing_in_mask_counter.most_common(15):
            print(f"  ID {inst_id}: 缺失 {freq} 次")

    if warnings:
        print("\n部分其他警告:")
        for warning in warnings[:detail_limit]:
            print(f"  [WARN] {warning}")

    # class_map中有，但训练集/图像中未出现的图像条目
    map_image_set = set(class_map.keys())
    actual_image_set = set(image_files)
    extra_map_images = map_image_set - actual_image_set
    missing_map_images = actual_image_set - map_image_set
    if extra_map_images:
        print(f"\nclass_map中有但图像目录中没有的条目: {len(extra_map_images)} 个")
        for img_name in sorted(extra_map_images)[:20]:
            print(f"  {img_name}")
    if missing_map_images:
        print(f"\n图像目录中有但class_map中没有的条目: {len(missing_map_images)} 个")
        for img_name in sorted(missing_map_images)[:20]:
            print(f"  {img_name}")



# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="检查 Mask2Former 实例分割数据集")
    parser.add_argument("--image", type=str, default=None,
                        help="单张图片名（如 000036-0.png），指定后不做全量检查，只诊断这一张")
    parser.add_argument("--image_dir", default=None, help="单张图诊断时的图像目录")
    parser.add_argument("--mask_dir", default=None, help="单张图诊断时的mask目录")
    parser.add_argument("--class_map", default=None, help="单张图诊断时的class_map路径")
    parser.add_argument("--train_image_dir", default=r"E:\soft\code\Mask2former_data\data\semantic_views_train")
    parser.add_argument("--train_mask_dir", default=r"E:\soft\code\Mask2former_data\data\ins_masks_train")
    parser.add_argument("--train_class_map", default=r"E:\soft\code\Mask2former_data\data\class_map_train.json")
    parser.add_argument("--val_image_dir", default=r"E:\soft\code\Mask2former_data\data\semantic_views_val")
    parser.add_argument("--val_mask_dir", default=r"E:\soft\code\Mask2former_data\data\ins_masks_val")
    parser.add_argument("--val_class_map", default=r"E:\soft\code\Mask2former_data\data\class_map_val.json")
    parser.add_argument("--detail_limit", type=int, default=30, help="最多打印多少条详细异常")
    args = parser.parse_args()

    if args.image:
        mask_dir = args.mask_dir or args.train_mask_dir
        class_map_path = args.class_map or args.train_class_map
        inspect_single_image(
            mask_path=os.path.join(mask_dir, args.image),
            class_map_path=class_map_path,
            detail_image_name=args.image,
        )
    else:
        check_split(
            "train",
            args.train_image_dir,
            args.train_mask_dir,
            args.train_class_map,
            detail_limit=args.detail_limit,
        )
        check_split(
            "val",
            args.val_image_dir,
            args.val_mask_dir,
            args.val_class_map,
            detail_limit=args.detail_limit,
        )


if __name__ == "__main__":
    main()
