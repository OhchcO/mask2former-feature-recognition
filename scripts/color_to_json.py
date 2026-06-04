"""
面ID图转JSON脚本

输入：不同颜色表示不同面的二维图
输出：JSON文件，记录每个面的像素坐标

用法：
    python color_to_json.py --image path/to/face_id_image.png --output path/to/output.json
"""

import numpy as np
import cv2
import json
import argparse
import os


def extract_faces_from_color_image(image_path, background_color=None, min_area=10):
    """
    从彩色面ID图中提取每个面的像素信息

    Args:
        image_path: 面ID图路径
        background_color: 背景颜色 (R, G, B)，如果为None则自动检测
        min_area: 最小面面积（像素数），小于此值的区域将被忽略

    Returns:
        faces: 面信息列表
    """
    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法加载图像: {image_path}")

    # BGR转RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 获取图像尺寸
    height, width = image_rgb.shape[:2]
    print(f"图像尺寸: {width} x {height}")

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
        print(f"自动检测背景颜色: RGB{background_color} (像素数: {color_counts[background_color]})")

    # 提取每个面
    faces = []
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

        # 提取像素坐标 (y, x)
        y_coords, x_coords = np.where(mask == 1)
        pixels_list = [[int(x), int(y)] for x, y in zip(x_coords, y_coords)]

        # 计算边界框
        bbox = [int(np.min(x_coords)), int(np.min(y_coords)),
                int(np.max(x_coords)), int(np.max(y_coords))]

        # 计算面积
        area = pixel_count

        face_info = {
            "id": int(face_id),
            "color_rgb": [int(c) for c in color_tuple],
            "pixel_count": int(pixel_count),
            "bbox": [int(b) for b in bbox],
            "area": int(area),
            "pixels": [[int(x), int(y)] for x, y in pixels_list]
        }

        faces.append(face_info)
        face_id += 1

        print(f"  面 {face_id - 1}: RGB{color_tuple}, 像素数={pixel_count}, 边界框={bbox}")

    print(f"\n共提取 {len(faces)} 个面")

    return faces


def save_faces_to_json(faces, output_path, image_path=None):
    """
    将面信息保存为JSON文件

    Args:
        faces: 面信息列表
        output_path: 输出JSON文件路径
        image_path: 原始图像路径（可选，用于记录）
    """
    data = {
        "source_image": image_path,
        "image_size": None,
        "num_faces": len(faces),
        "faces": faces
    }

    # 如果有图像，记录图像尺寸
    if image_path and os.path.exists(image_path):
        image = cv2.imread(image_path)
        if image is not None:
            h, w = image.shape[:2]
            data["image_size"] = {"width": w, "height": h}

    # 保存JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"已保存到: {output_path}")


def create_face_mask_image(faces, image_size, output_path=None):
    """
    根据面信息创建面ID图（用于验证）

    Args:
        faces: 面信息列表
        image_size: (height, width) 图像尺寸
        output_path: 保存路径（可选）

    Returns:
        face_id_image: 面ID图
    """
    height, width = image_size
    face_id_image = np.zeros((height, width), dtype=np.uint8)

    for face in faces:
        face_id = face["id"]
        for x, y in face["pixels"]:
            if 0 <= x < width and 0 <= y < height:
                face_id_image[y, x] = face_id

    if output_path:
        # 使用颜色映射可视化
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))
        plt.imshow(face_id_image, cmap='tab20')
        plt.title(f"Face ID Image ({len(faces)} faces)")
        plt.colorbar(label='Face ID')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"面ID图已保存: {output_path}")

    return face_id_image


def main():
    parser = argparse.ArgumentParser(description="面ID图转JSON脚本")
    parser.add_argument("--image", type=str, required=True,
                        help="面ID图路径（不同颜色表示不同面）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出JSON文件路径（默认：与图像同名.json）")
    parser.add_argument("--background", type=int, nargs=3, default=None,
                        help="背景颜色 RGB（默认：自动检测）")
    parser.add_argument("--min_area", type=int, default=10,
                        help="最小面面积（像素数，默认：10）")
    parser.add_argument("--visualize", action="store_true",
                        help="是否生成可视化面ID图")

    args = parser.parse_args()

    # 检查图像是否存在
    if not os.path.exists(args.image):
        print(f"错误: 图像不存在: {args.image}")
        return

    # 设置输出路径
    if args.output is None:
        base_name = os.path.splitext(args.image)[0]
        args.output = base_name + ".json"

    # 设置背景颜色
    background_color = tuple(args.background) if args.background else None

    print("=" * 50)
    print("面ID图转JSON工具")
    print("=" * 50)
    print(f"输入图像: {args.image}")
    print(f"输出JSON: {args.output}")
    print(f"背景颜色: {background_color if background_color else '自动检测'}")
    print(f"最小面积: {args.min_area} 像素")
    print("=" * 50)

    # 提取面信息
    faces = extract_faces_from_color_image(
        args.image,
        background_color=background_color,
        min_area=args.min_area
    )

    if len(faces) == 0:
        print("警告: 未找到任何面！请检查背景颜色设置。")
        return

    # 保存JSON
    save_faces_to_json(faces, args.output, args.image)

    # 可视化（可选）
    if args.visualize:
        image = cv2.imread(args.image)
        if image is not None:
            h, w = image.shape[:2]
            vis_path = os.path.splitext(args.output)[0] + "_visualization.png"
            create_face_mask_image(faces, (h, w), vis_path)

    print("\n完成！")


if __name__ == "__main__":
    main()
