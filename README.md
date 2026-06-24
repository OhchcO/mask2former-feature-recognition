# 基于 Mask2Former 的加工特征智能识别系统

## 项目简介

本项目利用 **Mask2Former**（通用图像分割模型）对 **3D CAD 模型**（STP/STEP 格式）中的加工特征进行实例分割识别。通过创新的面信息编码方案，将 3D 几何语义信息嵌入到 RGB 通道中，实现了从"看图识特征"到"理解几何结构"的跨越。

**目标**：自动识别零件中的开放型腔、封闭型腔、复合型腔等加工特征，为智能工艺规划提供基础。

---

## 技术方案

### 整体架构

```
STP 文件 → 面解析 → RGB 编码 → 多视角渲染 → Mask2Former 实例分割
                                              ↓
                                    面边界后处理 → 特征分类
```

### 创新点 1：面信息 RGB 通道编码

传统方法将 3D 模型渲染为普通 2D 图像，丢失了几何语义。本项目设计了一种**自适应 RGB 编码方案**，将面的几何类型和身份信息直接编码到图像通道中：

| 通道 | 信息 | 编码方式 |
|------|------|---------|
| R 通道 | 面的几何类型 | 平面=0, 圆柱面=51, 圆锥面=102, 球面=153, 其他面=204 |
| G + B 通道 | 面 ID | 自适应网格映射：K=⌈√N⌉, step=254/K, G×256+B |

**核心优势**：
- 编码后的图像同时包含**视觉特征**（边缘、纹理）和**几何语义**（面类型、面 ID）
- 模型输入保留了 3D 模型的完整拓扑信息，无需额外的特征工程
- 自适应编码支持不同数量的面（从几个到几百个）

### 创新点 2：多视角实例分割

单张 2D 投影无法完整表达 3D 结构。本项目采用**12 视角渲染策略**：

- 从 12 个固定角度（正视、俯视、侧视、等轴等）渲染同一模型
- 每个视角生成：编码图（RGB）+ 掩码图（灰度，像素值=面 ID）
- 不同视角下某些面不可见（被遮挡），自动过滤空掩码样本
- **同一模型的不同视角共享面 ID**，确保跨视角一致性

### 创新点 3：面边界约束后处理

模型推理输出的 mask 边界通常不够精确。本项目利用编码图中的**面几何信息**进行后处理：

1. 从编码图提取每个面的精确边界（基于 R/GB 通道值）
2. 对每个面区域进行**多数投票**，确定该面的最终分类
3. 利用面边界约束 mask 的形状，提升边界精度

**效果**：后处理将 79 个预测面精炼为有效实例，过滤掉低置信度误检。

### 创新点 4：统一配置管理

项目设计了**平台自适应配置系统**（`config.py`），支持 Windows/Linux 无缝切换：

- 自动检测操作系统，切换数据路径
- 类别权重通过 Median Frequency Balancing 从 class_map.json 自动计算
- 非连续类 ID（如 5/6/7）自动映射为连续索引（0/1/2）
- 换数据集时只需修改 `NUM_CLASSES` 和 `CLASS_NAMES`

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 基础模型 | Mask2Former (facebook/mask2former-swin-base-COCO) |
| 微调策略 | 冻结 backbone，训练分类头（41.9% 参数可训练） |
| 评估指标 | COCO mAP@0.5（实例级） |
| 3D 解析 | PythonOCC (STEP/STP) |
| 训练框架 | PyTorch + HuggingFace Transformers |
| 可视化 | TensorBoard + Matplotlib |

---

## 模型架构

```
输入编码图 (3, 1024, 1024)
        ↓
  Swin Transformer (backbone, 冻结)
        ↓
  Pixel Decoder (特征金字塔)
        ↓
  Transformer Decoder (100 queries)
        ↓
  ┌─────────────┬──────────────┐
  │ 分类头       │ 掩码头        │
  │ 3 类 + noobj │ 二值掩码预测   │
  └─────────────┴──────────────┘
```

- **100 个 query**：每个 query 负责预测一个实例的类别和掩码
- **二部图匹配**：训练时通过 Hungarian Matching 将预测与真值配对
- **多任务 Loss**：分类 Loss + 掩码 Loss + Dice Loss 联合优化

---

## 数据格式

```
dataset/
├── train_encoded_views/    # 训练编码图（RGB）
├── train_masks/            # 训练掩码图（灰度，像素值=面ID）
├── val_encoded_views/      # 验证编码图
├── val_masks/              # 验证掩码图
└── class_map.json          # 实例→类别映射
```

**class_map.json 示例**：
```json
{
  "000049.png": {
    "0": 5,    // 实例 0 → 类别 5（开放型腔）
    "1": 5,    // 实例 1 → 类别 5
    "4": 6,    // 实例 4 → 类别 6（封闭型腔）
    "6": 7     // 实例 6 → 类别 7（复合型腔）
  }
}
```

---

## 训练结果

| 指标 | 值 |
|------|-----|
| 训练集 | 84 张图，384 个实例 |
| 验证集 | 84 张图 |
| Epochs | 30 |
| Best mAP@0.5 | 0.5427 |
| 开放型腔 AP | 0.5697 |
| 封闭型腔 AP | 0.5933 |
| 复合型腔 AP | 0.3863 |

> 注：当前数据量有限（12 个模型 × 12 视角），增加数据后性能将持续提升。

---

## 快速开始

### 环境配置

```bash
conda create -n mask2former python=3.9
conda activate mask2former
pip install torch torchvision
pip install transformers datasets
pip install pycocotools
pip install Pillow numpy matplotlib tqdm tensorboard
```

### 训练

```bash
python scripts/ins_finetune_custom.py
```

### 推理

```bash
python scripts/ins_inference_encoded.py \
    --image <编码图路径> \
    --model_dir <模型目录> \
    --threshold 0.5
```

---

## 项目结构

```
Mask2former/
├── scripts/
│   ├── config.py                    # 统一配置（路径、类别、超参）
│   ├── ins_finetune_custom.py       # 实例分割训练
│   ├── ins_finetune_custom_linux.py # Linux 版训练
│   ├── ins_inference_encoded.py     # 实例分割推理
│   └── seg_finetune_custom.py       # 语义分割训练（备用）
├── stp2png/
│   ├── color_encoder.py             # 面信息 RGB 编码器
│   ├── stp_parser.py                # STP 文件解析
│   ├── multi_view_renderer.py       # 多视角渲染
│   └── mask_generator.py            # 掩码生成
└── README.md
```

---

## 应用场景

1. **智能工艺规划**：自动识别加工特征 → 匹配加工方法 → 生成工艺路线
2. **CAM 辅助**：为数控编程提供特征级的几何信息
3. **质量检测**：对比设计特征与加工结果，检测加工缺陷
4. **知识沉淀**：建立加工特征库，支持工艺知识复用

---

## 未来方向

- **数据增强**：增加更多零件模型，提升模型泛化能力
- **背景类训练**：加入背景类别，解决背景误分类问题
- **端到端流程**：STP → 特征识别 → 工艺推荐的全自动管线
- **多模态融合**：结合文本描述（材料、精度要求）进行联合推理
