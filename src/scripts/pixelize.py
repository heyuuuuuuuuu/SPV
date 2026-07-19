#!/usr/bin/env python
"""固定分辨率像素化"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from src.config import PIXELIZED_DIR, RENDERED_DIR
from src.core.pixelize import pixelize, batch_pixelize

INPUT_DIR = str(RENDERED_DIR)
OUTPUT_DIR = str(PIXELIZED_DIR)
GRID_SIZES = [6, 8, 10, 12]
THRESHOLD = 0.20

if __name__ == "__main__":
    # 单字模式
    if len(sys.argv) > 1:
        char = sys.argv[1]
        if len(char) == 1 and "\u4e00" <= char <= "\u9fff":
            from PIL import Image
            gs = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 8
            fname = f"{char}_U+{ord(char):04X}.png"
            path = os.path.join(INPUT_DIR, "64x64", fname)
            if not os.path.exists(path):
                print(f"未找到汉字 '{char}'")
            else:
                img = pixelize(path, grid_size=gs, threshold=THRESHOLD)
                out_dir = os.path.join(OUTPUT_DIR, f"{gs}x{gs}")
                os.makedirs(out_dir, exist_ok=True)
                img.save(os.path.join(out_dir, fname))
                arr = np.array(img)
                print(f"{char} {gs}x{gs} (threshold={THRESHOLD}):")
                for row in arr:
                    print("".join("█" if p == 255 else "·" for p in row))
                print(f"  点亮率: {(arr==255).sum()}/{arr.size} = {(arr==255).sum()/arr.size*100:.1f}%")
            sys.exit(0)

    # 批量模式
    print("=" * 50)
    print(f"固定分辨率像素化 (threshold={THRESHOLD})")
    print("=" * 50)
    batch_pixelize(INPUT_DIR, OUTPUT_DIR, GRID_SIZES, THRESHOLD)
    print("\n全部完成!")
