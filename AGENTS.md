# AGENTS.md — SPV 汉字识别开发指南

## 目标

本项目研究模拟假体视觉（SPV）条件下 999 个常用汉字的可识别性，比较固定电极阵列 `6×6 / 8×8 / 10×10 / 12×12`，并评估基于 BPSCA 复杂度的自适应分辨率策略。

当前仓库只保留一套实现。旧 checkpoint、旧 CLI、兼容层和历史实验结果均不作为依据；模型或数据协议变化后必须重新训练。

## 权威配置

- Python：`3.12`
- 环境：`pyproject.toml` + `uv.lock`
- 运行命令：始终使用 `uv run python ...`
- 路径配置：`src/config.py`；可用 `SPV_DATA_DIR`、`SPV_OUTPUTS_DIR` 覆盖根目录
- 数据标签：`data/dataset/labels.csv`
- 生成产物：`data/` 和 `outputs/`，不提交 Git
- 不使用 `pip install -r requirements.txt`，不重新引入 requirements.txt

初始化：

```bash
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
uv sync
```

## 代码边界

```text
src/config.py       项目根目录及默认路径
src/core/           无 CLI 的图像处理算法
src/data/           数据生成、Dataset、DataLoader
src/models/         模型定义与工厂
src/training/       训练、验证、评估
src/analysis/       混淆和 fixed/adaptive 分析
src/cloud/          COS 归档上传/下载
src/scripts/        唯一 CLI 层
src/utils/          通用指标与预览
```

设计约束：

1. 核心模块不得包含 `if __name__ == "__main__"`、默认实验路径或命令行解析。
2. CLI 仅放在 `src/scripts/`，参数必须可覆盖默认路径。
3. 禁止硬编码 Windows 路径、用户主目录或机器相关绝对路径。
4. 默认路径从 `src.config` 导入；运行时数据路径可以由 CLI 参数覆盖。
5. 不保留同一功能的多个实现。重构时直接删除废弃代码，不增加兼容 wrapper。
6. checkpoint 必须由当前模型代码生成；结构变化时旧 checkpoint 直接作废。

## 数据协议

`labels.csv` 至少包含：

```text
image_path,label,split,resolution
```

推荐同时包含：

```text
resolution_config,augment_id,complexity_group
```

约定：

- `split` 只能是 `train / val / test`。
- `resolution` 表示实际电极阵列大小。
- `resolution_config` 表示实验配置，用于区分 fixed 与 adaptive；新数据必须写入该字段。
- CSV 中的新路径应写为相对于 CSV 所在目录的 POSIX 路径。
- train/val/test 按增强样本划分，只能解释为同字形扰动泛化；不得宣称跨字体泛化。
- adaptive 当前映射：low→8×8、medium→10×10、high→12×12。

## Baseline 协议

- 模型：LightCNN
- 类别：999
- 输入：单通道、`128×128`、归一化至 `[0,1]`
- 优化器：AdamW
- 调度器：CosineAnnealingLR
- 固定随机种子：42
- 每个分辨率独立训练、独立输出目录
- 正式报告至少包含 test Top-1、Top-5、逐类准确率和混淆矩阵
- smoke test 仅验证链路，不可作为实验结论

```bash
for res in 6x6 8x8 10x10 12x12; do
  uv run python src/scripts/train.py \
    --model light_cnn --resolution "$res" --epochs 50 --seed 42 \
    --output-dir "outputs/baseline/light_cnn/$res"
done
```

## 修改要求

- 函数签名使用类型注解；优先使用 `pathlib.Path` 处理新路径代码。
- 文件头简要说明职责，注释解释实验设计而非重复代码。
- 新增依赖只修改 `pyproject.toml`，随后执行 `uv lock`。
- 不提交数据集、归档、模型、日志、生成图片、缓存或凭据。
- COS 凭据只能通过 `COS_SECRET_ID`、`COS_SECRET_KEY` 环境变量提供。
- 删除文件前检查引用；不要修改用户未要求处理的论文参考资料。

## 最低验证

每次代码修改后至少执行：

```bash
uv run python -m compileall -q src
uv run python src/scripts/train.py --help
uv run python src/scripts/evaluate.py --help
git diff --check
```

涉及数据或训练代码时，再验证：

```bash
uv run python - <<'PY'
from src.config import LABELS_CSV
from src.data import SPVCharDataset
sample = SPVCharDataset(str(LABELS_CSV), split="train", resolution="8x8")[0]
print(sample["image"].shape, sample["meta"])
PY
```

不得为了通过验证而静默吞掉数据缺失、标签越界或 checkpoint 不匹配错误。
