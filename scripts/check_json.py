# 检查JSON文件是否符合COCO格式
import json
import os

json_path = r"E:\soft\code\Mask2former\新建文件夹\coco_dataset.json"

if not os.path.exists(json_path):
    print(f"文件不存在: {json_path}")
    exit()

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print("=== 图像信息 ===")
for img in data['images']:
    print(f"  id={img['id']}, file={img['file_name']}, size={img['width']}x{img['height']}")

print("\n=== 标注信息 ===")
for ann in data['annotations']:
    seg_count = len(ann['segmentation'])
    print(f"  id={ann['id']}, image_id={ann['image_id']}, category={ann['category_id']}, 多边形数={seg_count}")

# 检查测试图像
test_image = r"E:\soft\code\Mask2former\test\1.png"
if os.path.exists(test_image):
    from PIL import Image
    img = Image.open(test_image)
    print(f"\n=== 测试图像 ===")
    print(f"  路径: {test_image}")
    print(f"  尺寸: {img.size[0]}x{img.size[1]}")
