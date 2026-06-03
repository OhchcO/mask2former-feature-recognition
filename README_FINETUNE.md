# Mask2Former 微调指南

## 快速开始

### 1. 准备数据

运行数据准备脚本创建目录结构：

```bash
python prepare_data.py
```

这会创建以下目录：
- `train/images/` - 训练图像
- `train/masks/` - 训练标签掩码
- `val/images/` - 验证图像
- `val/masks/` - 验证标签掩码

### 2. 添加训练数据

**图像要求：**
- 格式：JPG、PNG、JPEG
- 内容：任何需要分割的图像

**标签掩码要求：**
- 格式：PNG（灰度图）
- 像素值：0=背景，1=类别1，2=类别2，...
- 文件名：与图像对应（如 `image.jpg` → `image.png`）

**示例：**
```
train/images/photo001.jpg
train/masks/photo001.png    (对应的标签掩码)
```

### 3. 运行微调

**简单版本（推荐初学者）：**
```bash
python finetune_simple.py
```

**完整版本（更多选项）：**
```bash
python finetune.py
```

### 4. 使用微调后的模型

```python
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
from PIL import Image

model_dir = "finetuned_model"
processor = Mask2FormerImageProcessor.from_pretrained(model_dir)
model = Mask2FormerForUniversalSegmentation.from_pretrained(model_dir)

image = Image.open("test.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

result = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
```

## 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 学习率 | 5e-5 | AdamW优化器学习率 |
| 批次大小 | 2 | 每次训练的样本数 |
| 训练轮数 | 5-10 | 完整遍历数据集的次数 |
| 图像尺寸 | 512x512 | 输入图像统一尺寸 |

## 微调策略

### 冻结层
默认冻结编码器（encoder）以加快训练：
```python
for param in model.model.pixel_level_module.encoder.parameters():
    param.requires_grad = False
```

### 解冻所有层
如果数据量大，可以解冻所有层进行微调：
```python
# 注释掉冻结代码
# for param in model.model.pixel_level_module.encoder.parameters():
#     param.requires_grad = False
```

## 常见问题

**Q: 训练时显存不足怎么办？**
- 减小批次大小（batch_size）
- 减小图像尺寸
- 使用梯度累积

**Q: 如何提高分割精度？**
- 增加训练数据量
- 使用数据增强
- 调整学习率
- 增加训练轮数

**Q: 标签掩码如何制作？**
- 使用标注工具：LabelMe、CVAT、VIA
- 每个类别用不同的像素值表示
- 保存为PNG格式的灰度图

## 文件说明

| 文件 | 说明 |
|------|------|
| `prepare_data.py` | 数据准备脚本 |
| `finetune_simple.py` | 简化版微调脚本 |
| `finetune.py` | 完整版微调脚本 |
| `example.py` | 推理示例脚本 |

## 训练数据示例

### 二分类（前景/背景）
```
像素值: 0 = 背景
像素值: 255 = 前景（目标物体）
```

### 多分类
```
像素值: 0 = 背景
像素值: 1 = 类别1（如：人）
像素值: 2 = 类别2（如：车）
像素值: 3 = 类别3（如：建筑）
```