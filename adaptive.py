"""
模块3: 复杂度自适应分辨率分配模型 (BPSCA-Driven)

基于 BPSCA (黑色像素统计复杂度算法) 复杂度分数的自适应分辨率分配
输入: BPSCA 复杂度分数 + 64×64 二值图
输出: 自适应分辨率像素化图 + 分配结果 CSV
规则: low→8×8, medium→10×10, high→12×12
"""

import os
import csv
import numpy as np
from PIL import Image


# 复杂度等级 → 分辨率映射
GROUP_TO_RESOLUTION = {
    "low":    8,
    "medium": 10,
    "high":   12,
}

THRESHOLD = 0.20  # 像素化点亮阈值


def pixelize_cell(arr: np.ndarray, grid_size: int, threshold: float) -> np.ndarray:
    """将二值数组像素化为 grid_size×grid_size，返回二值数组"""
    h, w = arr.shape
    cell_h = h // grid_size
    cell_w = w // grid_size

    output = np.zeros((grid_size, grid_size), dtype=np.uint8)
    for i in range(grid_size):
        for j in range(grid_size):
            y0, y1 = i * cell_h, (i + 1) * cell_h
            x0, x1 = j * cell_w, (j + 1) * cell_w
            cell = arr[y0:y1, x0:x1]
            stroke_ratio = (cell == 255).sum() / cell.size
            if stroke_ratio > threshold:
                output[i, j] = 255
    return output


def adaptive_pixelize(
    complexity_csv: str,
    char_image_dir: str,
    output_dir: str,
    threshold: float = THRESHOLD,
    image_size: int = 64,
) -> list[dict]:
    """
    根据复杂度自适应分配分辨率并生成像素化图
    Returns: 分配结果列表
    """
    # 读取复杂度
    char_info = {}
    with open(complexity_csv, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            char_info[row["汉字"]] = {
                "complexity": float(row["复杂度"]),
                "level": row["等级"],
            }

    results = []
    img_folder = os.path.join(char_image_dir, f"{image_size}x{image_size}")

    for char, info in char_info.items():
        level = info["level"]
        gs = GROUP_TO_RESOLUTION[level]

        # 查找图像文件
        code = ord(char)
        fname = f"{char}_U+{code:04X}.png"
        path = os.path.join(img_folder, fname)
        if not os.path.exists(path):
            # 模糊匹配
            for f in os.listdir(img_folder):
                if f.startswith(f"{char}_"):
                    path = os.path.join(img_folder, f)
                    break

        if not os.path.exists(path):
            print(f"[WARN] 图像缺失: {char}")
            continue

        # 像素化
        arr = np.array(Image.open(path))
        pix = pixelize_cell(arr, gs, threshold)
        pix_img = Image.fromarray(pix)

        # 保存
        out_subdir = os.path.join(output_dir, f"{gs}x{gs}")
        os.makedirs(out_subdir, exist_ok=True)
        pix_img.save(os.path.join(out_subdir, fname))

        # 统计
        num_lit = (pix == 255).sum()
        total_pixels = gs * gs

        results.append({
            "char": char,
            "complexity": info["complexity"],
            "group": level,
            "resolution": f"{gs}x{gs}",
            "num_pixels": total_pixels,
            "num_lit": num_lit,
            "lit_ratio": round(num_lit / total_pixels, 4),
        })

    return results


def save_results(results: list[dict], output_csv: str):
    """保存分配结果 CSV"""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["char", "complexity", "group", "resolution", "num_pixels", "num_lit", "lit_ratio"])
        for r in results:
            writer.writerow([
                r["char"], r["complexity"], r["group"],
                r["resolution"], r["num_pixels"], r["num_lit"], r["lit_ratio"],
            ])

    # 统计
    n = len(results)
    low = sum(1 for r in results if r["group"] == "low")
    med = sum(1 for r in results if r["group"] == "medium")
    high = sum(1 for r in results if r["group"] == "high")
    avg_pixels = sum(r["num_pixels"] for r in results) / n

    print(f"总计: {n} 个汉字")
    print(f"  low    → 8×8   : {low:4d} ({low/n*100:5.1f}%)  64 pixels")
    print(f"  medium → 10×10 : {med:4d} ({med/n*100:5.1f}%) 100 pixels")
    print(f"  high   → 12×12 : {high:4d} ({high/n*100:5.1f}%) 144 pixels")
    print(f"  平均像素点数: {avg_pixels:.1f}")
    print(f"\n结果已保存: {output_csv}")

    return results


# ═══════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    INPUT_CSV = "E:/dataset/complexity_scores.csv"
    INPUT_IMG_DIR = "E:/dataset/char_rendered_hei"
    OUTPUT_DIR = "E:/dataset/char_adaptive_pixelized"
    OUTPUT_CSV = "E:/dataset/adaptive_allocation.csv"

    # 单字查询: python adaptive.py 明
    if len(sys.argv) > 1:
        char = sys.argv[1]
        if len(char) == 1 and "\u4e00" <= char <= "\u9fff":
            # 查找复杂度等级
            level = "unknown"
            complexity = 0.0
            with open(INPUT_CSV, encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    if row["汉字"] == char:
                        level = row["等级"]
                        complexity = float(row["复杂度"])
                        break

            gs = GROUP_TO_RESOLUTION.get(level, 8)
            code = ord(char)
            fname = f"{char}_U+{code:04X}.png"
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
