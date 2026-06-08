# Mask2Former 项目结构

## 目录布局

```
E:\soft\code\Mask2former\
├── scripts/                          # Python 脚本
│   ├── verify_instance_labels.py     # 验证实例标签正确性（5图布局）
│   ├── verify_semantic_labels.py     # 验证语义标签正确性（染色显示）
│   ├── instance_to_semantic.py       # 实例标签转语义标签
│   ├── generate_class_map.py         # 随机生成 class_map.json
│   ├── seg_finetune_custom.py        # 语义分割微调训练
│   ├── ins_finetune_custom.py        # 实例分割微调训练
│   ├── seg_inference.py              # 语义分割推理
│   ├── seg_inference_mask_utils.py   # 语义分割推理（带面ID工具）
│   ├── ins_inference_mask_utils.py   # 实例分割推理（带面ID工具）
│   ├── face_mask_utils.py            # 面掩码工具函数
│   ├── split_dataset.py              # 按零件划分训练集/验证集
│   ├── augment_data.py               # 数据增强
│   ├── color_to_json.py              # 面ID图转JSON
│   ├── visualize_one.py              # 可视化单个样本标签
│   ├── visualize_mapping.py          # 可视化标签映射关系
│   ├── check_json.py                 # 检查COCO格式JSON
│   ├── calculate_class_weights.py    # 计算类权重（处理类别不平衡）
│   └── example.py                    # 预训练模型加载示例
│
├── stp2png/                          # STP文件处理
│   ├── render_stp.py                 # STP文件渲染为图片（pyvista）
│   └── analyze_colors.py            # RGB像素颜色分析
│
├── train/                            # 训练数据
│   ├── images_png/                   # 训练图片（PNG格式）
│   ├── instance_masks_png/           # 实例掩码（灰度图）
│   ├── semantic_masks_png/           # 语义掩码（灰度图）
│   └── class_map.json                # 实例ID到类别ID的映射
│
├── results/                          # 输出结果
│   ├── models/                       # 保存的模型
│   │   └── finetuned_model/          # 微调后的模型
│   │       ├── config.json
│   │       ├── preprocessor_config.json
│   │       └── pytorch_model.bin
│   ├── verify_instance_labels/       # 实例标签验证结果
│   ├── verify_semantic_labels/       # 语义标签验证结果
│   └── visualizations/               # 可视化结果
│
├── models--facebook--mask2former-swin-tiny-coco-panoptic/  # 预训练模型缓存
│
├── download.py                       # 下载预训练模型
├── config.json                       # 模型配置
├── preprocessor_config.json          # 预处理器配置
├── pytorch_model.bin                 # 预训练模型权重
└── requirements.txt                  # Python依赖
```

---

## 脚本详细说明

### 数据准备类

#### `download.py` — 下载预训练模型
下载 Facebook 的 `mask2former-swin-tiny-coco-panoptic` 预训练模型到本地。
```bash
python download.py
```

#### `generate_class_map.py` — 随机生成 class_map.json
为实例分割任务随机生成 `class_map.json`，建立实例ID到类别的映射关系。类别定义：0=宽体槽, 1=封闭槽, 2=开放槽, 3=孔。
```bash
python scripts/generate_class_map.py
```

#### `split_dataset.py` — 按零件划分训练集/验证集
将图片按零件分组（每12张为一个零件），确保同一零件的所有图片在同一集合中，避免数据泄露。
```bash
python scripts/split_dataset.py --image_dir train/new_images_png --mask_dir train/new_masks_png --output_dir train --val_ratio 0.2 --images_per_part 12
```

#### `augment_data.py` — 数据增强
对图像和掩码同时进行相同的几何变换（翻转、旋转、缩放等），保持一一对应。颜色变换只应用于图像，不应用于掩码。
```bash
python scripts/augment_data.py --image_dir path/to/temp --mask_dir path/to/temp_mask --output_image_dir path/to/aug_images --output_mask_dir path/to/aug_masks --num_augment 6
```

#### `color_to_json.py` — 面ID图转JSON
将不同颜色表示不同面的二维图转换为JSON文件，记录每个面的像素坐标。
```bash
python scripts/color_to_json.py --image path/to/face_id_image.png --output path/to/output.json
```

#### `face_mask_utils.py` — 面掩码工具函数
提供从面ID图提取面掩码的工具函数，推理时可调用做面统计。支持从COCO格式标注文件提取，或直接从面ID图提取。
```python
from face_mask_utils import extract_face_masks_from_color_image
```

---

### 标签处理类

#### `instance_to_semantic.py` — 实例标签转语义标签
将实例灰度掩码图转换为语义灰度掩码图。输入一个文件夹（包含实例PNG和 `class_map.json`），自动输出到子文件夹 `semantic_masks`。
```bash
python scripts/instance_to_semantic.py
# 修改脚本中 input_dir 为你的文件夹路径
```

#### `calculate_class_weights.py` — 计算类权重
根据训练数据中各类别的像素数量，计算类权重，用于处理类别不平衡问题。
```bash
python scripts/calculate_class_weights.py
```

---

### 标签验证类

#### `verify_instance_labels.py` — 验证实例标签正确性
输入实例掩码文件夹 + `class_map.json`，输出5图布局的验证图：实例可视化、实例图例、语义可视化、语义图例、原始掩码图。
```bash
python scripts/verify_instance_labels.py
# 修改脚本中 instance_mask_dir 和 class_map_path 路径
```

#### `verify_semantic_labels.py` — 验证语义标签正确性
输入语义掩码文件夹，输出染色可视化图片。左图为灰度原图，右图为按类别染色的彩色图+图例。
```bash
python scripts/verify_semantic_labels.py
# 修改脚本中 input_dir 和 output_dir 路径
```

#### `visualize_one.py` — 可视化单个样本
可视化单个样本的标签掩码结果，用于快速检查。
```bash
python scripts/visualize_one.py
```

#### `visualize_mapping.py` — 可视化标签映射关系
可视化标签掩码的映射关系，验证标签的正确性。
```bash
python scripts/visualize_mapping.py
```

#### `check_json.py` — 检查COCO格式JSON
检查标注文件是否符合COCO格式，输出图像、标注、类别等统计信息。
```bash
python scripts/check_json.py
```

---

### 训练类

#### `seg_finetune_custom.py` — 语义分割微调训练
基于 Mask2Former 进行语义分割的微调训练。输入图片和语义掩码（同类同值），输出训练好的模型。
```bash
python scripts/seg_finetune_custom.py
```

#### `ins_finetune_custom.py` — 实例分割微调训练
基于 Mask2Former 进行实例分割的微调训练。输入图片、实例掩码（每个实例不同值）和 `class_map.json`，后处理使用 `post_process_instance_segmentation`。
```bash
python scripts/ins_finetune_custom.py
```

---

### 推理类

#### `seg_inference.py` — 语义分割推理
使用微调后的模型进行语义分割推理，输出分割结果可视化。
```bash
python scripts/seg_inference.py
```

#### `seg_inference_mask_utils.py` — 语义分割推理（带面ID工具）
推理时输入原始图片和面ID图，结合面掩码工具进行语义分割推理。
```bash
python scripts/seg_inference_mask_utils.py --image train/images_png/000001_0.png --unc_image train/images_png/000001.png --face_id_image temp/000001.png
```

#### `ins_inference_mask_utils.py` — 实例分割推理（带面ID工具）
推理时输入原始图片和面ID图，结合面掩码工具进行实例分割推理。
```bash
python scripts/ins_inference_mask_utils.py --image train/images_png/000001_0.png --unc_image train/images_png/000001.png --face_id_image temp/000001.png
```

#### `example.py` — 预训练模型加载示例
验证 Mask2Former 模型是否加载成功，展示基本的推理流程。
```bash
python scripts/example.py
```

---

### STP文件处理类

#### `stp2png/render_stp.py` — STP文件渲染
使用 pyvista 将 STP/STEP 三维模型文件渲染为二维图片。面按类型染色（平面=橙色, 圆柱面=紫色, 圆锥面=青色, 其他=粉色），边为黑色。输出有光照和无光照两个版本。
```bash
python stp2png/render_stp.py
# 修改脚本中的 stp_dir 和 output_base 路径
```

#### `stp2png/analyze_colors.py` — RGB像素颜色分析
分析图片中的RGB像素颜色分布，统计预定义颜色匹配率和未匹配的过渡色。支持单图分析和双图对比。
```bash
python stp2png/analyze_colors.py
```

---

## 类别定义

| 类别ID | 类别名 | 语义分割灰度值 | 实例分割灰度值 |
|--------|--------|--------------|--------------|
| 0 | 背景 | 0 | 255 |
| 1 | 宽体槽 | 1 | 0-254（随机分配） |
| 2 | 封闭槽 | 2 | 0-254（随机分配） |
| 3 | 开放槽 | 3 | 0-254（随机分配） |
| 4 | 孔 | 4 | 0-254（随机分配） |

## 掩码格式说明

- **语义分割**：灰度图中每个像素值代表类别ID（0=背景，1-4=各类特征）
- **实例分割**：灰度图中每个像素值代表实例ID（0-254=特征实例，255=背景），需要配合 `class_map.json` 确定每个实例的类别
- **class_map.json**：格式为 `{图片名: {实例ID: 类别ID, ...}, ...}`

## 训练配置

- **模型**: Mask2Former (facebook/mask2former-swin-tiny-coco-panoptic)
- **设备**: CPU（可切换至 CUDA）
- **Epochs**: 5
- **Batch Size**: 2
- **Learning Rate**: 5e-5
- **Image Size**: 512x512

## 环境依赖

安装 `requirements.txt` 中的依赖，PyTorch 需单独安装（见 `requirements.txt` 注释）。
```bash
pip install -r requirements.txt
```
