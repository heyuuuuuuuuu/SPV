"""
模块4: 固定分辨率 vs 自适应分辨率 — 对比评估

基于 BPSCA 复杂度分档，对比 Fixed-6/8/10/12 与 Adaptive 的性能
指标: 结构相似度(IoU/Dice)、平均磷光点数量、复杂度分档表现
"""

import os
import csv
import numpy as np
from PIL import Image
from collections import defaultdict

# ── 配置 ──
ORIGINAL_DIR = "E:/dataset/char_rendered_hei/64x64"
FIXED_DIR = "E:/dataset/char_pixelized"         # 6×6, 8×8, 10×10, 12×12
ADAPTIVE_DIR = "E:/dataset/char_adaptive_pixelized"  # 8×8, 10×10, 12×12
COMPLEXITY_CSV = "E:/dataset/complexity_scores.csv"
OUTPUT_CSV = "E:/dataset/comparison_results.csv"


def load_pixelized(path: str) -> np.ndarray:
    """加载像素化图像，返回 (h,w) 二值数组"""
    return np.array(Image.open(path))


def reconstruct(pixelized: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Nearest-neighbor 上采样到目标尺寸"""
    h, w = pixelized.shape
    scale_y = target_h / h
    scale_x = target_w / w
    y_idx = (np.arange(target_h) / scale_y).astype(int)
    x_idx = (np.arange(target_w) / scale_x).astype(int)
    return pixelized[y_idx][:, x_idx]


def compute_metrics(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    """
    计算重构质量指标
    original / reconstructed: 二值数组, 255=笔画(前景), 0=背景
    """
    orig_fg = original == 255
    recon_fg = reconstructed == 255

    intersection = (orig_fg & recon_fg).sum()
    union = (orig_fg | recon_fg).sum()
    tp = intersection
    fp = (recon_fg & ~orig_fg).sum()
    fn = (orig_fg & ~recon_fg).sum()
    tn = (~orig_fg & ~recon_fg).sum()

    iou = tp / union if union > 0 else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return {
        "iou": round(iou, 4),
        "dice": round(dice, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "pixel_accuracy": round(accuracy, 4),
        "phosphenes": int(tp + fp),
    }


def evaluate_method(
    method_name: str,
    char_to_resolution: dict,  # char → (grid_size, pixelized_path)
    originals: dict,           # char → np.ndarray
    target_size: int = 64,
) -> dict:
    """
    评估一种方法的所有汉字
    Returns: per_char_results, summary_stats
    """
    per_char = []
    group_stats = defaultdict(lambda: {"iou": [], "dice": [], "phosphenes": [], "count": 0})

    for char, (gs, pix_path) in char_to_resolution.items():
        if char not in originals:
            continue

        pix = load_pixelized(pix_path)
        recon = reconstruct(pix, target_size, target_size)
        metrics = compute_metrics(originals[char], recon)

        # 查复杂度等级
        level = char_levels.get(char, "unknown")

        metrics["char"] = char
        metrics["group"] = level
        metrics["resolution"] = f"{gs}x{gs}"
        per_char.append(metrics)

        for k in ["iou", "dice", "phosphenes"]:
            group_stats[level][k].append(metrics[k])
        group_stats[level]["count"] += 1

    # 汇总
    def avg(lst): return sum(lst) / len(lst) if lst else 0.0

    # 磷光点数 = 网格分辨率 (gs²)
    phosphenes_by_char = [gs * gs for gs, _ in char_to_resolution.values()]
    avg_grid_phosphenes = avg(phosphenes_by_char)

    summary = {
        "method": method_name,
        "total_chars": len(per_char),
        "avg_iou": avg([r["iou"] for r in per_char]),
        "avg_dice": avg([r["dice"] for r in per_char]),
        "avg_phosphenes": avg_grid_phosphenes,  # 网格分辨率 = 磷光点数
        "avg_pixel_accuracy": avg([r["pixel_accuracy"] for r in per_char]),
    }
    for level in ["low", "medium", "high"]:
        gs = group_stats[level]
        summary[f"{level}_n"] = gs["count"]
        summary[f"{level}_dice"] = avg(gs["dice"])
        summary[f"{level}_iou"] = avg(gs["iou"])
        summary[f"{level}_phosphenes"] = avg(gs["phosphenes"])

    return per_char, summary


# ═══════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("固定 vs 自适应分辨率 — 对比评估")
    print("=" * 60)

    # ── 1. 加载原始图像 ──
    print("\n[1] 加载原始 64×64 图像...")
    originals = {}
    for fname in os.listdir(ORIGINAL_DIR):
        if fname.endswith(".png"):
            char = fname.split("_")[0]
            originals[char] = np.array(Image.open(os.path.join(ORIGINAL_DIR, fname)))
    print(f"  加载 {len(originals)} 张原图")

    # ── 2. 加载复杂度等级 ──
    print("[2] 加载复杂度分数...")
    char_levels = {}
    with open(COMPLEXITY_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            char_levels[row["汉字"]] = row["等级"]

    # ── 3. 构建各方法的 char→resolution→path 映射 ──
    print("[3] 构建方法映射...")
    methods = {}

    for gs in [6, 8, 10, 12]:
        folder = os.path.join(FIXED_DIR, f"{gs}x{gs}")
        mapping = {}
        for fname in os.listdir(folder):
            if fname.endswith(".png"):
                char = fname.split("_")[0]
                mapping[char] = (gs, os.path.join(folder, fname))
        methods[f"Fixed-{gs}"] = mapping

    # Adaptive
    adaptive_mapping = {}
    for gs in [8, 10, 12]:
        folder = os.path.join(ADAPTIVE_DIR, f"{gs}x{gs}")
        if not os.path.exists(folder):
            continue
        for fname in os.listdir(folder):
            if fname.endswith(".png"):
                char = fname.split("_")[0]
                adaptive_mapping[char] = (gs, os.path.join(folder, fname))
    methods["Adaptive"] = adaptive_mapping

    # ── 4. 逐方法评估 ──
    print("\n[4] 逐方法评估...\n")
    all_summaries = []

    for method_name in ["Fixed-6", "Fixed-8", "Fixed-10", "Fixed-12", "Adaptive"]:
        print(f"  评估 {method_name} ...")
        _, summary = evaluate_method(method_name, methods[method_name], originals)
        all_summaries.append(summary)

    # ── 5. 输出汇总表 ──
    header_cn = ["方法", "样本数", "平均IoU", "平均Dice", "平均磷光点数", "平均像素准确率",
                 "低复杂度Dice", "中复杂度Dice", "高复杂度Dice",
                 "低复杂度磷光点", "中复杂度磷光点", "高复杂度磷光点"]

    key_map = {
        "方法": "method", "样本数": "total_chars",
        "平均IoU": "avg_iou", "平均Dice": "avg_dice",
        "平均磷光点数": "avg_phosphenes", "平均像素准确率": "avg_pixel_accuracy",
        "低复杂度Dice": "low_dice", "中复杂度Dice": "medium_dice", "高复杂度Dice": "high_dice",
        "低复杂度磷光点": "low_phosphenes", "中复杂度磷光点": "medium_phosphenes", "高复杂度磷光点": "high_phosphenes",
    }

    print(f"\n{'─'*90}")
    print(f"{'方法':<12}{'样本':>5}{'IoU':>8}{'Dice':>8}{'磷光点':>8}{'像素准确率':>10}  {'低Dice':>8}{'中Dice':>8}{'高Dice':>8}")
    print(f"{'─'*90}")

    for s in all_summaries:
        print(f"{s['method']:<12}{s['total_chars']:>5}"
              f"{s['avg_iou']:>8.4f}{s['avg_dice']:>8.4f}"
              f"{s['avg_phosphenes']:>8.1f}{s['avg_pixel_accuracy']:>8.4f}"
              f"  {s['low_dice']:>8.4f}{s['medium_dice']:>8.4f}{s['high_dice']:>8.4f}")

    print(f"{'─'*90}")

    # ── 6. 资源-准确率比 ──
    print(f"\n{'─'*90}")
    print(f"{'方法':<12}{'Dice':>8}{'磷光点':>8}{'Dice/磷光点':>14}{'vs Fixed-12 ΔDice':>18}{'Δ磷光点%':>10}")
    print(f"{'─'*90}")

    baseline = next(s for s in all_summaries if s["method"] == "Fixed-12")
    for s in all_summaries:
        dice_per_phos = s["avg_dice"] / s["avg_phosphenes"] * 1000 if s["avg_phosphenes"] > 0 else 0
        delta_dice = s["avg_dice"] - baseline["avg_dice"]
        delta_phos = (s["avg_phosphenes"] / baseline["avg_phosphenes"] - 1) * 100
        print(f"{s['method']:<12}{s['avg_dice']:>8.4f}{s['avg_phosphenes']:>8.1f}"
              f"{dice_per_phos:>12.2f}{delta_dice:>+15.4f}{delta_phos:>+9.1f}%")

    print(f"{'─'*90}")

    # ── 7. 保存详细 CSV ──
    print(f"\n[5] 保存结果...")
    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header_cn)
        writer.writeheader()
        for s in all_summaries:
            row = {cn: s[en] for cn, en in key_map.items()}
            writer.writerow(row)

    print(f"  → {OUTPUT_CSV}")

    # ── 8. 关键结论 ──
    adaptive = next(s for s in all_summaries if s["method"] == "Adaptive")
    fixed8 = next(s for s in all_summaries if s["method"] == "Fixed-8")
    fixed12 = next(s for s in all_summaries if s["method"] == "Fixed-12")

    print(f"\n{'='*60}")
    print("关键发现")
    print(f"{'='*60}")
    print(f"1. 自适应 vs Fixed-8:")
    print(f"   Dice 提升: {adaptive['avg_dice'] - fixed8['avg_dice']:+.4f}")
    print(f"   Phosphenes 增加: {adaptive['avg_phosphenes'] - fixed8['avg_phosphenes']:+.1f}")
    print(f"2. 自适应 vs Fixed-12:")
    print(f"   Dice 差异: {adaptive['avg_dice'] - fixed12['avg_dice']:+.4f}")
    print(f"   磷光点节省: {(1 - adaptive['avg_phosphenes']/fixed12['avg_phosphenes'])*100:.1f}%")
    print(f"3. 高复杂度汉字 Dice:")
    print(f"   Fixed-8: {fixed8['high_dice']:.4f}")
    print(f"   Adaptive: {adaptive['high_dice']:.4f}  (提升 {adaptive['high_dice'] - fixed8['high_dice']:+.4f})")
    print(f"   Fixed-12: {fixed12['high_dice']:.4f}")