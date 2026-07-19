#!/usr/bin/env python
"""BPSCA 汉字复杂度计算"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.complexity import batch_compute, compute_complexity

INPUT_DIR = "E:/dataset/char_rendered_hei"

if __name__ == "__main__":
    # 单字查询
    if len(sys.argv) > 1:
        char = sys.argv[1]
        if len(char) == 1 and "\u4e00" <= char <= "\u9fff":
            size = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] in ("64", "128") else "64"
            import csv
            fname = f"{char}_U+{ord(char):04X}.png"
            path = os.path.join(INPUT_DIR, f"{size}x{size}", fname)
            if not os.path.exists(path):
                print(f"未找到汉字 '{char}' 的 {size}x{size} 图像")
            else:
                result = compute_complexity(path)
                csv_path = "E:/dataset/complexity_scores.csv" if size == "64" else "E:/dataset/complexity_scores_128.csv"
                level = "unknown"
                if os.path.exists(csv_path):
                    with open(csv_path, encoding="utf-8-sig") as f:
                        for row in csv.reader(f):
                            if row[0] == char:
                                level = row[2]
                                break
                print(f"{char},{result['complexity']},{level}")
            sys.exit(0)

    # 批量计算
    print("=" * 50)
    print("BPSCA 汉字复杂度计算 — 64×64")
    print("=" * 50)
    batch_compute(INPUT_DIR, "E:/dataset/complexity_scores.csv", size=64)

    print(f"\n{'=' * 50}")
    print("BPSCA 汉字复杂度计算 — 128×128")
    print("=" * 50)
    batch_compute(INPUT_DIR, "E:/dataset/complexity_scores_128.csv", size=128)
