#!/usr/bin/env python
"""高斯磷光点仿真渲染"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from PIL import Image
from src.config import ADAPTIVE_PIXELIZED_DIR, DATA_DIR, PIXELIZED_DIR, SPV_ADAPTIVE_DIR, SPV_FIXED_DIR
from src.core.phosphene import render_and_save, batch_render

OUTPUT_SIZE = 256

if __name__ == "__main__":
    # 单字模式
    if len(sys.argv) > 1:
        char = sys.argv[1]
        if len(char) == 1 and "\u4e00" <= char <= "\u9fff":
            gs = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 8
            code = ord(char)
            fname = f"{char}_U+{code:04X}.png"
            grid_path = os.path.join(PIXELIZED_DIR, f"{gs}x{gs}", fname)

            if not os.path.exists(grid_path):
                print(f"未找到 {char} 的 {gs}x{gs} 像素化图像")
                sys.exit(1)

            output_path = DATA_DIR / "spv_demo" / f"{char}_{gs}x{gs}.png"
            spv = render_and_save(grid_path, str(output_path), output_size=OUTPUT_SIZE)
            grid = np.array(Image.open(grid_path))
            lit = (grid == 255).sum()
            print(f"{char} {gs}x{gs} → SPV {OUTPUT_SIZE}x{OUTPUT_SIZE}  (phosphenes: {lit}/{gs*gs})")
            print(f"  已保存: {output_path}")
            sys.exit(0)

    # 批量: Fixed 分辨率
    print("=" * 50)
    print("高斯磷光点仿真 — Fixed 分辨率")
    print("=" * 50)
    batch_render(
        input_base=str(PIXELIZED_DIR),
        output_base=str(SPV_FIXED_DIR),
        grid_sizes=[6, 8, 10, 12],
        output_size=OUTPUT_SIZE,
    )

    # 批量: Adaptive 分辨率
    print(f"\n{'=' * 50}")
    print("高斯磷光点仿真 — Adaptive 分辨率")
    print("=" * 50)
    batch_render(
        input_base=str(ADAPTIVE_PIXELIZED_DIR),
        output_base=str(SPV_ADAPTIVE_DIR),
        grid_sizes=[8, 10, 12],
        output_size=OUTPUT_SIZE,
    )

    print("\n全部完成!")
