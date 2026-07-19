#!/usr/bin/env python
"""固定分辨率 vs 自适应分辨率对比评估"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 直接运行原始 compare 模块逻辑
from src.analysis.compare import *  # noqa

if __name__ == "__main__":
    import csv
    import numpy as np
    from PIL import Image
    from collections import defaultdict

    from src.analysis.compare import (
        load_pixelized, reconstruct, compute_metrics,
    )

    ORIGINAL_DIR = "E:/dataset/char_rendered_hei/64x64"
    FIXED_DIR = "E:/dataset/char_pixelized"
    ADAPTIVE_DIR = "E:/dataset/char_adaptive_pixelized"
    COMPLEXITY_CSV = "E:/dataset/complexity_scores.csv"
    OUTPUT_CSV = "E:/dataset/comparison_results.csv"

    print("=" * 60)
    print("固定 vs 自适应分辨率对比评估")
    print("=" * 60)
    # 此处保留 compare.py 的完整主逻辑...

    print("\nTODO: 完整的 compare.py 主逻辑迁移")
    print("请直接运行 src/analysis/compare.py 完成对比评估")
