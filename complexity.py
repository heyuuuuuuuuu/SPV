"""
模块1: 汉字复杂度计算模型 — BPSCA (黑色像素统计复杂度算法)

Black Pixel Statistical Complexity Algorithm (BPSCA)

输入: 二值汉字图像 (PNG)
输出: CSV — 汉字,复杂度分数,复杂度等级
计算: 复杂度 = 笔画像素数 / 图像总面积
分组: 按百分位三等分 (low/medium/high)
"""

import os
import csv
import numpy as np
from PIL import Image


def compute_complexity(img_path: str) -> dict:
    """
    BPSCA: 计算单个汉字的复杂度

    方法: 统计二值图中前景(笔画)像素数，计算其在图像总面积中的占比。
    Black Pixel Statistical Complexity Algorithm:
      complexity = stroke_pixels / total_pixels

    Returns: {complexity, bbox_w, bbox_h, bbox_area, stroke_pixels, ...}
    """
    img = Image.open(img_path)
    arr = np.array(img)

    h, w = arr.shape

    # 判断前景/背景：少数像素通常是笔画
    total = w * h
    white_cnt = (arr == 255).sum()
    black_cnt = total - white_cnt

    if white_cnt <= black_cnt:
        # 黑底白字：白色=笔画(前景), 黑色=背景
        fg_val = 255
        bg_val = 0
    else:
        # 白底黑字：黑色=笔画(前景), 白色=背景
        fg_val = 0
        bg_val = 255

    # 字符外接框 (前景像素的 bounding box)
    fg_coords = np.argwhere(arr == fg_val)
    if len(fg_coords) == 0:
        return {"complexity": 0.0, "level": "empty", "bbox_w": 0, "bbox_h": 0,
                "bbox_area": 0, "stroke_pixels": 0}

    y_min, x_min = fg_coords.min(axis=0)
    y_max, x_max = fg_coords.max(axis=0)
    bbox_w = x_max - x_min + 1
    bbox_h = y_max - y_min + 1
    bbox_area = bbox_w * bbox_h

    # bbox 内的前景像素数
    bbox_region = arr[y_min:y_max + 1, x_min:x_max + 1]
    stroke_pixels = (bbox_region == fg_val).sum()

    # 复杂度 = 笔画像素数 / 图像总面积（消除形状偏差）
    complexity = stroke_pixels / total

    return {
        "complexity": round(complexity, 4),
        "bbox_w": bbox_w,
        "bbox_h": bbox_h,
        "bbox_area": bbox_area,
        "stroke_pixels": stroke_pixels,
        "total_pixels": total,
        "fg_val": fg_val,
        "bg_val": bg_val,
    }


def assign_levels(results: list[dict]):
    """BPSCA 分级: 按复杂度百分位三等分分配 low/medium/high"""
    n = len(results)
    sorted_results = sorted(results, key=lambda r: r["complexity"])

    third = n // 3
    for i, r in enumerate(sorted_results):
        if i < third:
            r["level"] = "low"
        elif i < 2 * third:
            r["level"] = "medium"
        else:
            r["level"] = "high"

    return results


def batch_compute(input_dir: str, output_csv: str, size: int = 64):
    """BPSCA 批量计算：对所有汉字计算 BPSCA 分数并输出 CSV"""
    folder = os.path.join(input_dir, f"{size}x{size}")
    files = sorted([f for f in os.listdir(folder) if f.endswith(".png")])

    results = []
    for fname in files:
        char = fname.split("_")[0]
        path = os.path.join(folder, fname)
        info = compute_complexity(path)
        info["char"] = char
        results.append(info)

    # 百分位分级
    assign_levels(results)

    # 写入 CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["汉字", "复杂度", "等级", "外接框宽", "外接框高", "外接框面积", "笔画像素数"])
        for r in results:
            writer.writerow([
                r["char"],
                r["complexity"],
                r["level"],
                r["bbox_w"],
                r["bbox_h"],
                r["bbox_area"],
                r["stroke_pixels"],
            ])

    # 统计
    low = sum(1 for r in results if r["level"] == "low")
    med = sum(1 for r in results if r["level"] == "medium")
    high = sum(1 for r in results if r["level"] == "high")

    # 边界值
    lows = [r["complexity"] for r in results if r["level"] == "low"]
    meds = [r["complexity"] for r in results if r["level"] == "medium"]
    highs = [r["complexity"] for r in results if r["level"] == "high"]
    lo_hi = max(lows) if lows else 0
    med_hi = max(meds) if meds else 0

    print(f"总计: {len(results)} 个汉字")
    print(f"  低复杂度 (≤{lo_hi:.4f}): {low:4d} ({low/len(results)*100:5.1f}%)")
    print(f"  中复杂度 ({lo_hi:.4f}~{med_hi:.4f}): {med:4d} ({med/len(results)*100:5.1f}%)")
    print(f"  高复杂度 (≥{min(highs):.4f} if highs else '?'): {high:4d} ({high/len(results)*100:5.1f}%)")
    print(f"\nCSV 已保存: {output_csv}")

    return results


# ═══════════════════════════════════════════════════════════
# 主程序 — BPSCA (黑色像素统计复杂度算法)
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    INPUT_DIR = "E:/dataset/char_rendered_hei"

    # BPSCA 单字查询: python complexity.py 明 [64|128]
    if len(sys.argv) > 1:
        char = sys.argv[1]
        if len(char) == 1 and "\u4e00" <= char <= "\u9fff":
            size = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] in ("64", "128") else "64"
            fname = f"{char}_U+{ord(char):04X}.png"
            path = os.path.join(INPUT_DIR, f"{size}x{size}", fname)
            if not os.path.exists(path):
                print(f"未找到汉字 '{char}' 的 {size}x{size} 图像")
            else:
                result = compute_complexity(path)
                # 从 CSV 查等级
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

    # BPSCA 批量计算
    OUTPUT_CSV = "E:/dataset/complexity_scores.csv"
    print("=" * 50)
    print("BPSCA 汉字复杂度计算 — 64×64")
    print("=" * 50)
    batch_compute(INPUT_DIR, OUTPUT_CSV, size=64)

    print(f"\n{'=' * 50}")
    print("BPSCA 汉字复杂度计算 — 128×128")
    print("=" * 50)
    OUTPUT_CSV_128 = "E:/dataset/complexity_scores_128.csv"
    batch_compute(INPUT_DIR, OUTPUT_CSV_128, size=128)
