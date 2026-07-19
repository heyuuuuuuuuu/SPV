# SPV 汉字识别

模拟假体视觉（SPV）下的汉字识别、混淆分析与复杂度自适应分辨率实验。

## 环境初始化

项目使用 Python 3.12 和 `uv` 锁定环境。PyTorch 默认锁定为 2.3.0；在 CUDA 机器上可通过 PyTorch 官方索引安装对应 CUDA wheel。

```bash
# 国内 PyPI 镜像
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
export PATH="$HOME/.local/bin:$PATH"

uv sync
```

若需要 CUDA 12.1 版本 PyTorch：

```bash
uv sync --no-install-package torch --no-install-package torchvision
uv pip install torch==2.3.0 torchvision==0.18.0 \
  --index-url https://download.pytorch.org/whl/cu121
```

之后统一使用：

```bash
uv run python src/scripts/train.py --help
```

## 默认目录

所有默认路径均相对于仓库根目录；大数据盘可通过 `SPV_DATA_DIR`、`SPV_OUTPUTS_DIR` 环境变量覆盖：

```text
data/
├── dataset/labels.csv       # 当前增强识别数据集；图像路径相对此 CSV
├── char_rendered_hei/
├── char_pixelized/
├── char_adaptive_pixelized/
├── char_spv_fixed/
└── char_spv_adaptive/
outputs/                     # 训练与评估产物，不纳入版本控制
```

旧实验 checkpoint 已废弃。模型结构变化后必须使用当前代码重新训练。数据集归档只是传输缓存，解包成功后可以删除；不要将归档放入 Git。

## Baseline

每个分辨率应使用独立输出目录训练，避免覆盖：

```bash
for res in 6x6 8x8 10x10 12x12; do
  uv run python src/scripts/train.py \
    --resolution "$res" \
    --output-dir "outputs/baseline/light_cnn/$res" \
    --model light_cnn --epochs 50 --seed 42
done
```

评估示例：

```bash
uv run python src/scripts/evaluate.py \
  --checkpoint outputs/baseline/light_cnn/8x8/best_model.pth \
  --resolution 8x8 \
  --output-dir outputs/baseline/light_cnn/8x8/evaluate
```

其他入口：

```bash
uv run python src/scripts/complexity.py
uv run python src/scripts/pixelize.py
uv run python src/scripts/adaptive.py
uv run python src/scripts/spv_render.py
uv run python src/scripts/confusion.py
```
