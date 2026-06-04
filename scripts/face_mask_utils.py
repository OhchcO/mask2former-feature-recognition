# 将COCO格式的标注文件转换为二值面掩码图像，并推理时可以调用用来做面统计
import numpy as np
import cv2
import json
import os


def extract_face_masks_from_color_image(image_path, background_color=None, min_area=10):
    """
    直接从面ID图提取面掩码（不需要中间JSON文件）

    Args:
        image_path: 面ID图路径（不同颜色表示不同面）
        background_color: 背景颜色 (R, G, B)，如果为None则自动检测
        min_area: 最小面面积（像素数），小于此值的区域将被忽略

    Returns:
        face_masks: dict, {face_id: mask_info}
    """
    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法加载图像: {image_path}")

    # BGR转RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 获取图像尺寸
    height, width = image_rgb.shape[:2]
    print(f"面ID图尺寸: {width} x {height}")

    # 找出所有独特颜色
    pixels = image_rgb.reshape(-1, 3)
    unique_colors = np.unique(pixels, axis=0)

    print(f"找到 {len(unique_colors)} 种独特颜色")

    # 自动检测背景颜色（通常是出现次数最多的颜色）
    if background_color is None:
        # 统计每种颜色的像素数
        color_counts = {}
        for color in unique_colors:
            mask = np.all(image_rgb == color, axis=2)
            count = np.sum(mask)
            color_counts[tuple(color)] = count

        # 找出出现次数最多的颜色作为背景
        background_color = max(color_counts, key=color_counts.get)
        print(f"自动检测背景颜色: RGB{background_color}")

    # 提取每个面
    face_masks = {}
    face_id = 1

    for color in unique_colors:
        color_tuple = tuple(color)

        # 跳过背景颜色
        if color_tuple == tuple(background_color):
            continue

        # 创建该颜色的掩码
        mask = np.all(image_rgb == color, axis=2).astype(np.uint8)

        # 计算像素数
        pixel_count = np.sum(mask)

        # 跳过太小的区域（可能是噪点）
        if pixel_count < min_area:
            continue

        face_masks[face_id] = {
            'mask': mask,
            'category_id': None,  # 没有类别信息
            'instance_name': f'face_{face_id}',
            'color_rgb': list(color_tuple),
            'area': int(pixel_count),
            'pixel_count': int(pixel_count)
        }

        face_id += 1

    print(f"共提取 {len(face_masks)} 个面")

    return face_masks


def load_color_json(json_path):
    """
    加载颜色转JSON脚本生成的标注文件

    Args:
        json_path: JSON标注文件路径

    Returns:
        faces: 面信息列表
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    faces = data.get('faces', [])
    image_size = data.get('image_size', None)

    print(f"加载颜色JSON文件: {json_path}")
    print(f"  面数量: {len(faces)}")
    if image_size:
        print(f"  图像尺寸: {image_size['width']} x {image_size['height']}")

    return faces, image_size


def create_face_masks_from_color_json(faces, image_size):
    """
    从颜色JSON创建面掩码

    Args:
        faces: load_color_json返回的面信息列表
        image_size: (height, width) 图像尺寸

    Returns:
        face_masks: dict, {face_id: mask_info}
    """
    height, width = image_size
    face_masks = {}

    for face in faces:
        face_id = face["id"]
        color_rgb = face.get("color_rgb", [0, 0, 0])
        pixels = face.get("pixels", [])
        area = face.get("area", 0)

        # 创建掩码
        mask = np.zeros((height, width), dtype=np.uint8)
        for x, y in pixels:
            if 0 <= x < width and 0 <= y < height:
                mask[y, x] = 1

        face_masks[face_id] = {
            'mask': mask,
            'category_id': None,  # 颜色JSON中没有类别信息
            'instance_name': f'face_{face_id}',
            'color_rgb': color_rgb,
            'area': area,
            'pixel_count': face.get('pixel_count', len(pixels))
        }

    return face_masks


def load_annotations(json_path):
    """
    加载COCO格式的标注文件

    Args:
        json_path: JSON标注文件路径

    Returns:
        annotations: 标注列表
        images: 图像信息列表
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    annotations = data.get('annotations', [])
    images = data.get('images', [])

    print(f"加载标注文件: {json_path}")
    print(f"  图像数量: {len(images)}")
    print(f"  标注数量: {len(annotations)}")

    return annotations, images


def get_annotations_by_image_id(annotations, image_id):
    """
    根据image_id获取该图像的所有标注

    Args:
        annotations: 标注列表
        image_id: 图像ID

    Returns:
        该图像的标注列表
    """
    return [ann for ann in annotations if ann['image_id'] == image_id]


def create_face_mask(polygons, image_size):
    """
    从多边形顶点创建面的填充掩码（支持多个多边形）

    Args:
        polygons: 多边形坐标列表，可以是：
                  - 单个多边形: [x1, y1, x2, y2, ...]
                  - 多个多边形: [[x1, y1, ...], [x1, y1, ...], ...]
        image_size: (height, width) 图像尺寸

    Returns:
        mask: 二值掩码，面内为1，面外为0
    """
    height, width = image_size
    mask = np.zeros((height, width), dtype=np.uint8)

    # 判断是单个多边形还是多个多边形
    if len(polygons) > 0 and isinstance(polygons[0], (list, np.ndarray)):
        # 多个多边形
        for polygon in polygons:
            coords = np.array(polygon).reshape(-1, 2)
            pts = coords.astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
    else:
        # 单个多边形
        coords = np.array(polygons).reshape(-1, 2)
        pts = coords.astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)

    return mask


def create_all_face_masks(annotations, image_size):
    """
    为所有标注创建面掩码

    Args:
        annotations: 标注列表
        image_size: (height, width) 图像尺寸

    Returns:
        face_masks: dict, {instance_id: mask_info}
    """
    face_masks = {}

    for ann in annotations:
        instance_id = ann['id']
        segmentation = ann['segmentation']
        category_id = ann['category_id']
        instance_name = ann.get('instance_name', f'instance_{instance_id}')

        # 支持多个多边形
        mask = create_face_mask(segmentation, image_size)

        face_masks[instance_id] = {
            'mask': mask,
            'category_id': category_id,
            'instance_name': instance_name,
            'segmentation': segmentation,
            'area': ann.get('area', 0)
        }

    return face_masks


def voting_postprocess(segmentation_map, face_masks, min_ratio=0.5):
    """
    使用面掩码进行投票后处理

    Args:
        segmentation_map: 语义分割结果 (H, W)
        face_masks: create_all_face_masks返回的字典
        min_ratio: 投票阈值，默认0.5

    Returns:
        processed_map: 处理后的分割结果
    """
    result = segmentation_map.copy()

    for instance_id, face_info in face_masks.items():
        mask = face_info['mask']
        category_id = face_info.get('category_id')  # 可能为None
        instance_name = face_info['instance_name']

        face_pixels = segmentation_map[mask == 1]

        if len(face_pixels) == 0:
            continue

        unique, counts = np.unique(face_pixels, return_counts=True)
        max_idx = np.argmax(counts)
        predicted_class = unique[max_idx]
        predicted_ratio = counts[max_idx] / len(face_pixels)

        if predicted_ratio >= min_ratio:
            # 主类别占比超过阈值，使用预测的类别
            result[mask == 1] = predicted_class
            decision = f"预测类别 {predicted_class}"
        elif category_id is not None:
            # 有标注类别，使用标注类别
            result[mask == 1] = category_id
            decision = f"标注类别 {category_id}"
        else:
            # 没有标注类别，仍然使用预测的类别（即使占比不高）
            result[mask == 1] = predicted_class
            decision = f"预测类别 {predicted_class} (低置信度)"

        print(f"  面 {instance_name}: {decision} (占比: {predicted_ratio:.2%})")

    return result


def clip_segmentation_to_masks(segmentation_map, face_masks):
    """
    将分割结果裁剪到面边界内，面外的像素设为背景

    Args:
        segmentation_map: 语义分割结果 (H, W)
        face_masks: create_all_face_masks返回的字典

    Returns:
        clipped_map: 裁剪后的分割结果
    """
    result = np.zeros_like(segmentation_map)  # 初始化为背景(0)

    # 合并所有面的掩码
    all_faces_mask = np.zeros(segmentation_map.shape[:2], dtype=np.uint8)
    for instance_id, face_info in face_masks.items():
        all_faces_mask = np.logical_or(all_faces_mask, face_info['mask']).astype(np.uint8)

    # 只保留面内的像素
    result[all_faces_mask == 1] = segmentation_map[all_faces_mask == 1]

    # 统计裁剪掉的像素数
    clipped_pixels = np.sum((segmentation_map != 0) & (all_faces_mask == 0))
    print(f"  裁剪掉面外像素: {clipped_pixels} 个")

    return result


def process_image_with_annotations(segmentation_map, json_path, image_id, min_ratio=0.5, clip_to_boundary=True):
    """
    一步完成：加载标注并进行投票后处理

    Args:
        segmentation_map: Mask2Former的语义分割结果 (H, W)
        json_path: JSON标注文件路径
        image_id: 当前图像的ID
        min_ratio: 投票阈值
        clip_to_boundary: 是否裁剪到面边界内

    Returns:
        processed_map: 处理后的分割结果
    """
    print(f"\n{'='*50}")
    print("开始投票后处理")
    print(f"{'='*50}")

    annotations, images = load_annotations(json_path)

    image_annotations = get_annotations_by_image_id(annotations, image_id)

    if len(image_annotations) == 0:
        print(f"警告: 未找到 image_id={image_id} 的标注")
        return segmentation_map

    print(f"找到 {len(image_annotations)} 个面标注")

    image_size = segmentation_map.shape[:2]
    face_masks = create_all_face_masks(image_annotations, image_size)

    # 投票后处理
    processed_map = voting_postprocess(segmentation_map, face_masks, min_ratio)

    # 裁剪到面边界内
    if clip_to_boundary:
        print("\n裁剪到面边界内...")
        processed_map = clip_segmentation_to_masks(processed_map, face_masks)

    print(f"{'='*50}")
    print("投票后处理完成")
    print(f"{'='*50}\n")

    return processed_map


def visualize_masks(image_path, json_path, image_id, save_path=None):
    """
    可视化面的掩码，用于调试

    Args:
        image_path: 图像路径
        json_path: JSON标注文件路径
        image_id: 图像ID
        save_path: 保存路径（可选）
    """
    from PIL import Image
    import matplotlib.pyplot as plt

    # 加载图像
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # 加载标注
    annotations, _ = load_annotations(json_path)
    image_annotations = get_annotations_by_image_id(annotations, image_id)

    if len(image_annotations) == 0:
        print(f"未找到 image_id={image_id} 的标注")
        return

    # 创建掩码
    image_size = image_np.shape[:2]
    face_masks = create_all_face_masks(image_annotations, image_size)

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image_np)
    axes[0].set_title("原图")

    # 显示所有面的掩码
    all_masks = np.zeros(image_size, dtype=np.uint8)
    for instance_id, face_info in face_masks.items():
        all_masks[face_info['mask'] == 1] = instance_id

    axes[1].imshow(all_masks, cmap='tab10')
    axes[1].set_title(f"面掩码 (共{len(face_masks)}个)")

    # 叠加显示
    overlay = image_np.copy()
    for instance_id, face_info in face_masks.items():
        mask = face_info['mask']
        # 红色高亮面区域
        overlay[mask == 1, 0] = 255
        overlay[mask == 1, 1] = overlay[mask == 1, 1] // 2
        overlay[mask == 1, 2] = overlay[mask == 1, 2] // 2

    axes[2].imshow(overlay)
    axes[2].set_title("叠加显示")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"可视化结果已保存: {save_path}")
    else:
        plt.show()

    # 打印掩码统计
    print(f"\n掩码统计:")
    for instance_id, face_info in face_masks.items():
        mask_area = np.sum(face_info['mask'])
        print(f"  {face_info['instance_name']}: 类别={face_info['category_id']}, 面积={mask_area}像素")


if __name__ == "__main__":
    import sys

    # 默认参数
    json_path = r"E:\soft\code\Mask2former\新建文件夹\coco_dataset.json"
    image_path = r"E:\soft\code\Mask2former\test\1.png"
    image_id = 1

    # 支持命令行参数
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    if len(sys.argv) > 2:
        json_path = sys.argv[2]
    if len(sys.argv) > 3:
        image_id = int(sys.argv[3])

    if os.path.exists(json_path) and os.path.exists(image_path):
        save_path = r"E:\soft\code\Mask2former\results\visualizations\mask_debug.png"
        visualize_masks(image_path, json_path, image_id, save_path)
    else:
        print(f"文件不存在:")
        print(f"  图像: {image_path} ({'存在' if os.path.exists(image_path) else '不存在'})")
        print(f"  标注: {json_path} ({'存在' if os.path.exists(json_path) else '不存在'})")
