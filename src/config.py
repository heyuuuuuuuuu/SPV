"""项目级路径配置。

所有默认路径均相对于仓库根目录，命令行参数仍可覆盖。
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("SPV_DATA_DIR", PROJECT_ROOT / "data")).expanduser().resolve()
OUTPUTS_DIR = Path(os.environ.get("SPV_OUTPUTS_DIR", PROJECT_ROOT / "outputs")).expanduser().resolve()
DATASET_DIR = DATA_DIR / "dataset"

LABELS_CSV = DATASET_DIR / "labels.csv"
RENDERED_DIR = DATA_DIR / "char_rendered_hei"
PIXELIZED_DIR = DATA_DIR / "char_pixelized"
ADAPTIVE_PIXELIZED_DIR = DATA_DIR / "char_adaptive_pixelized"
SPV_FIXED_DIR = DATA_DIR / "char_spv_fixed"
SPV_ADAPTIVE_DIR = DATA_DIR / "char_spv_adaptive"
COMPLEXITY_CSV = DATA_DIR / "complexity_scores.csv"
