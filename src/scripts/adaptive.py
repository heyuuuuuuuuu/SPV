#!/usr/bin/env python
"""复杂度自适应分辨率分配"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import csv
import numpy as np
from PIL import Image
from src.config import ADAPTIVE_PIXELIZED_DIR, COMPLEXITY_CSV, DATA_DIR, RENDERED_DIR
from src.core.adaptive import adaptive_pixelize, save_results, GROUP_TO_RESOLUTION, THRESHOLD, pixelize_cell

INPUT_CSV = str(COMPLEXITY_CSV)
INPUT_IMG_DIR = str(RENDERED_DIR)
OUTPUT_DIR = str(ADAPTIVE_PIXELIZED_DIR)
OUTPUT_CSV = str(DATA_DIR / "adaptive_allocation.csv")

if __name__ == "__main__":
    # 单字查询
    if len(sys.argv) > 1:
        char = sys.argv[1]
        if len(char) == 1 and "\u4e00" <= char <= "\u9fff":
            level = "unknown"
            complexity = 0.0
            with open(INPUT_CSV, encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    if row["汉字"] == char:
                        level = row["等级"]
                        complexity = float(row["复杂度"])
                        break

            gs = GROUP_TO_RESOLUTION.get(level, 8)
            fname = f"{char}_U+{ord(char):04X}.png"
            path = os.path.join(INPUT_IMG_DIR, "64x64", fname)

            if not os.path.exists(path):
                print(f"未找到汉字 '{char}'")
            else:
                arr = np.array(Image.open(path))
                pix = pixelize_cell(arr, gs, THRESHOLD)
                print(f"{char} | complexity={complexity:.4f} | group={level} | resolution={gs}x{gs}")
                for row in pix:
                    print("".join("1" if p == 255 else "0" for p in row))
                print(f"  lit: {(pix==255).sum()}/{gs*gs} ({(pix==255).sum()/(gs*gs)*100:.0f}%)")
            sys.exit(0)

    # 批量模式
    print("=" * 50)
    print("复杂度自适应分辨率分配")
    print("=" * 50)
    results = adaptive_pixelize(INPUT_CSV, INPUT_IMG_DIR, OUTPUT_DIR, THRESHOLD)
    results.sort(key=lambda r: r["complexity"])
    save_results(results, OUTPUT_CSV)
