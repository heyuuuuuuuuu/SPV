# AGENTS.md — SPV (Simulated Prosthetic Vision) 汉字识别

## 项目概述

本项目研究**模拟假体视觉 (SPV)** 下汉字的可识别性。将汉字经过低分辨率像素化 → 高斯磷光点仿真渲染 → 训练 CNN 识别模型 / 混淆分析，探讨不同分辨率 (6×6, 8×8, 10×10, 12×14) 下汉字的区分能力，并探索**复杂度自适应分辨率分配**策略。

## 技术栈

- **Python 3** + **PyTorch** (>=2.0) + **torchvision**
- 图像处理: Pillow, OpenCV
- 数据: NumPy, Pandas, scikit-learn
- 云存储: 腾讯云 COS (cos-python-sdk-v5)
- 可视化: matplotlib

## 项目结构

```
SPV/
├── src/                     # Python 包 (核心逻辑)
│   ├── __init__.py
│   ├── core/                # 核心流水线
│   │   ├── render.py        #   汉字字体渲染 (CharRenderer)
│   │   ├── complexity.py    #   BPSCA 复杂度计算
│   │   ├── pixelize.py      #   固定分辨率像素化
│   │   ├── adaptive.py      #   复杂度自适应分辨率分配
│   │   └── phosphene.py     #   高斯磷光点仿真渲染
│   ├── data/                # 数据模块
│   │   ├── augmentation.py  #   数据增强流水线
│   │   └── dataset.py       #   PyTorch Dataset + DataLoader
│   ├── models/              # 模型定义
│   │   ├── light_cnn.py     #   LightCNN (轻量级)
│   │   ├── resnet18.py      #   ResNet18SPV
│   │   └── factory.py       #   模型工厂函数 create_model()
│   ├── training/            # 训练与评估
│   │   ├── trainer.py       #   训练循环
│   │   └── evaluator.py     #   评估 (混淆矩阵/分组准确率)
│   ├── analysis/            # 分析工具
│   │   ├── confusion.py     #   自排除最近邻混淆分析
│   │   └── compare.py       #   固定 vs 自适应对比
│   ├── scan.py              # 部件顺序扫描显示
│   ├── utils/               # 工具函数
│   │   ├── metrics.py       #   Top-K 准确率/混淆矩阵等
│   │   └── preview.py       #   汉字预览工具
│   └── cloud/               # 云存储
│       ├── transfer.py      #   COS 传输 (纯标准库)
│       ├── upload.py        #   上传数据集 (with SDK)
│       └── download.py      #   下载数据集 (with SDK)
├── src/scripts/                 # CLI 入口脚本
│   ├── render.py
│   ├── complexity.py
│   ├── pixelize.py
│   ├── adaptive.py
│   ├── spv_render.py
│   ├── scan.py
│   ├── augment.py
│   ├── train.py
│   ├── evaluate.py
│   ├── confusion.py
│   ├── compare.py
│   ├── preview.py
│   ├── upload.py
│   └── download.py
├── _legacy/                 # 旧文件备份 (重构前的原始文件)
├── docs/                    # 文档
│   └── recognition.md       # 混淆分析详细文档
├── outputs/                 # 输出目录
├── requirements.txt
└── README.md
```

## 数据流水线

```
汉字原图 → pixelize (二值网格)
         → spv_render (高斯光斑仿真)
         → data_augmentation (增强: 平移/抖动/噪声)
         → dataset (PyTorch Dataset)
         → train (训练 CNN)
         → evaluate (评估)
```

## 关键设计决策

### 分辨率 (electrode array sizes)
- **6×6, 8×8, 10×10, 12×12** 是固定分辨率集合
- 8×8 作为参考基准分辨率 (确定混淆阈值 T)
- 12×12 足以完全消除汉字混淆

### 模型
- **LightCNN**: 轻量 4 层 CNN (~0.5M params)，输入 1 通道灰度图
- **ResNet18SPV**: 标准 ResNet18，支持 1 通道或 3 通道 (灰度复制)
- 默认类别数 999 (常用汉字)

### 混淆分析 (src/analysis/confusion.py)
- 自排除最近邻 (Self-Excluded Nearest-Neighbor)
- 余弦距离度量
- 8×8 基准确定统一阈值 T，跨分辨率比较

### 复杂度自适应 (src/core/adaptive.py, src/core/complexity.py)
- BPSCA (黑色像素统计复杂度算法)
- 低复杂度 → 8×8, 中复杂度 → 10×10, 高复杂度 → 12×12

## 路径约定

代码中硬编码了大量 Windows 路径 (`E:/dataset/...`, `E:/results/...`)，在 Linux 环境下运行前需修改为实际路径。数据集默认结构:

```
E:/dataset/
├── char_rendered_hei/       # 原始渲染汉字 (render.py)
│   ├── 64x64/
│   └── 128x128/
├── char_pixelized/          # 像素化网格 (pixelize.py)
│   ├── 6x6/
│   ├── 8x8/
│   ├── 10x10/
│   └── 12x12/
├── char_adaptive_pixelized/ # 自适应分辨率像素化
├── char_spv_fixed/          # 固定分辨率 SPV 图像
├── char_spv_adaptive/       # 自适应分辨率 SPV 图像
└── augmented_src/           # 增强后数据集 + labels.csv
```

## 编码规范

- 文件头使用中文注释，说明模块功能
- 使用 `═══` 分隔符标记代码段
- 类型注解 (type hints) 用于函数签名
- 参数使用 argparse 命令行接口

## 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 从 CLI 脚本运行 (推荐)
python src/scripts/render.py
python src/scripts/complexity.py
python src/scripts/pixelize.py
python src/scripts/adaptive.py
python src/scripts/spv_render.py
python src/scripts/augment.py
python src/scripts/train.py --resolution 8x8 --model light_cnn
python src/scripts/evaluate.py --checkpoint outputs/best_model.pth
python src/scripts/confusion.py --risk-percentile 20
python src/scripts/compare.py
python src/scripts/preview.py

# 单字查询模式
python src/scripts/complexity.py 明
python src/scripts/pixelize.py 明 8
python src/scripts/adaptive.py 明
python src/scripts/spv_render.py 明 8

# 从 Python API 调用
from src.core import pixelize, batch_pixelize
from src.models import create_model
from src.training import train, evaluate
```
