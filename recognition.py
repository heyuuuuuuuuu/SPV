"""
模块5: 自排除最近邻混淆分析 (Self-Excluded Nearest-Neighbor Confusion Analysis)

方法：
  1. 对每个分辨率的 SPV 图像，转为 128×128 归一化向量
  2. 排除自身后，用余弦距离找 Top-K 最近邻
  3. 以 8×8 的 nearest_distance 分位数确定统一阈值 T
  4. 同一 T 应用于全部分辨率，判定 high_confusion_risk
  5. 支持 --query 单字查询，输出最近混淆字、Top-K 相似字等

用法：
  python recognition.py
  python recognition.py --input-dir E:/dataset/char_spv_fixed --output-dir E:/results/confusion
  python recognition.py --resolutions 6x6 8x8 10x10 12x12 --risk-percentile 20
  python recognition.py --query 日 --top-k 5
"""

import os
import csv
import argparse
import numpy as np
from PIL import Image
from pathlib import Path


# ═══════════════════════════════════════════════════════════
# 图像工具
# ═══════════════════════════════════════════════════════════

def load_and_preprocess(path: str, target_size: int = 128) -> np.ndarray:
    """加载 SPV 图像，resize 到 target_size×target_size，归一化到 [0,1]，flatten"""
    img = Image.open(path).convert("L")
    img = img.resize((target_size, target_size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float64) / 255.0          # 归一化 [0, 1]
    return arr.flatten()                                     # (target_size²,)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    余弦距离 = 1 - cosine_similarity
    
    cosine_similarity = dot(a,b) / (||a|| * ||b||)
    距离 ∈ [0, 2], 0=完全相同方向, 2=完全相反
    """
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 2.0   # 空向量视为最大距离
    sim = dot / (norm_a * norm_b)
    return 1.0 - sim


# ═══════════════════════════════════════════════════════════
# 核心分析
# ═══════════════════════════════════════════════════════════

def analyze_resolution(
    image_dir: str,
    target_size: int = 128,
    top_k: int = 5,
) -> list[dict]:
    """
    对某个分辨率目录下所有 SPV 图像做 self-excluded 最近邻分析

    Args:
        image_dir: 包含 {char}_U+{code}.png 的目录
        target_size: resize 目标尺寸
        top_k: 保留 Top-K 最近邻
    
    Returns:
        per_char 列表, 每项含:
          char, unicode, nearest_other, nearest_distance,
          second_other, second_distance, margin,
          top_k_similar: [{rank, char, unicode, distance}, ...]
    """
    # ── 1. 加载所有图像 ──
    files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
    
    chars = []         # 汉字列表
    vectors = []       # 对应向量列表
    
    for fname in files:
        path = os.path.join(image_dir, fname)
        vec = load_and_preprocess(path, target_size)
        
        char = fname.split("_")[0]
        chars.append(char)
        vectors.append(vec)
    
    n = len(chars)
    if n < 2:
        print(f"  [WARN] {image_dir}: only {n} images, skip")
        return []
    
    # 堆成矩阵 [n, D] 加速计算
    mat = np.stack(vectors, axis=0)  # (n, D)
    
    norm = np.linalg.norm(mat, axis=1, keepdims=True)  # (n, 1)
    norm[norm < 1e-10] = 1e-10
    
    # 余弦相似度矩阵: sim[i,j] = dot(v_i, v_j) / (||v_i|| * ||v_j||)
    dot_matrix = mat @ mat.T                                    # (n, n)
    sim_matrix = dot_matrix / (norm @ norm.T)                   # (n, n)
    dist_matrix = 1.0 - sim_matrix                              # (n, n)
    
    # ── 2. 对每个汉字, 排除自身, 找 Top-K 最近邻 ──
    per_char = []
    
    for i in range(n):
        # 取出第 i 个汉字到所有其他汉字的距离
        dists = dist_matrix[i, :].copy()
        dists[i] = np.inf   # ★ 排除自身, 距离设为无穷大 ★
        
        # 找最近的 K 个 (索引, 距离)
        sorted_idx = np.argsort(dists)
        
        nearest_idx = sorted_idx[0]
        second_idx = sorted_idx[1] if len(sorted_idx) > 1 else -1
        
        nearest_distance = float(dists[nearest_idx])
        second_distance = float(dists[second_idx]) if second_idx >= 0 else float('inf')
        margin = second_distance - nearest_distance
        
        # Top-K 相似字列表
        k_neighbors = []
        for rank in range(min(top_k, n - 1)):
            idx = sorted_idx[rank]
            k_neighbors.append({
                "rank": rank + 1,
                "char": chars[idx],
                "unicode": f"U+{ord(chars[idx]):04X}",
                "distance": round(float(dists[idx]), 6),
            })
        
        per_char.append({
            "char": chars[i],
            "unicode": f"U+{ord(chars[i]):04X}",
            "nearest_other": chars[nearest_idx],
            "nearest_distance": round(nearest_distance, 6),
            "second_other": chars[second_idx] if second_idx >= 0 else "",
            "second_distance": round(second_distance, 6),
            "margin": round(margin, 6),
            "top_k_similar": k_neighbors,
        })
    
    return per_char


def compute_unified_threshold(
    per_char_ref: list[dict],
    risk_percentile: float = 20.0,
) -> float:
    """
    用参考分辨率 (默认 8×8) 的 nearest_distance 分布确定统一阈值 T
    
    规则: T = 距离最小的前 risk_percentile% 对应的分位数
    也就是说, nearest_distance < T 意味着该汉字落在最短距离的 risk_percentile% 中
    """
    dists = [r["nearest_distance"] for r in per_char_ref]
    dists_sorted = sorted(dists)
    idx = int(len(dists_sorted) * risk_percentile / 100.0)
    idx = max(0, min(idx, len(dists_sorted) - 1))
    T = dists_sorted[idx]
    return round(T, 6)


def apply_threshold(
    per_char: list[dict],
    threshold: float,
    resolution: str,
) -> list[dict]:
    """给每个汉字打上 is_high_confusion_risk 标签"""
    for r in per_char:
        r["resolution"] = resolution
        r["threshold"] = threshold
        r["is_high_confusion_risk"] = r["nearest_distance"] < threshold
    return per_char


def build_summary(per_char: list[dict], resolution: str) -> dict:
    """汇总某个分辨率的统计"""
    n = len(per_char)
    if n == 0:
        return {}
    
    high_risk = sum(1 for r in per_char if r["is_high_confusion_risk"])
    dists = [r["nearest_distance"] for r in per_char]
    margins = [r["margin"] for r in per_char]
    
    return {
        "resolution": resolution,
        "num_chars": n,
        "num_high_confusion": high_risk,
        "confusion_rate": round(high_risk / n, 4),
        "mean_nearest_distance": round(np.mean(dists), 6),
        "median_nearest_distance": round(np.median(dists), 6),
        "mean_margin": round(np.mean(margins), 6),
    }


# ═══════════════════════════════════════════════════════════
# 单字查询报告
# ═══════════════════════════════════════════════════════════

def print_char_report(
    per_char_entry: dict,
    threshold: float,
    top_k_show: int = 5,
):
    """
    打印单个汉字的详细混淆分析报告
    
    输出:
      - 最近混淆字
      - 第二混淆字
      - 最近距离 / 第二距离 / Margin
      - Top-K 相似字
      - 混淆风险结果
    """
    r = per_char_entry
    char = r["char"]
    unicode_str = r.get("unicode", f"U+{ord(char):04X}")
    resolution = r.get("resolution", "?")
    
    top_k = r.get("top_k_similar", [])
    nearest = r["nearest_other"]
    nearest_dist = r["nearest_distance"]
    second = r.get("second_other", "")
    second_dist = r.get("second_distance", 0.0)
    margin = r.get("margin", 0.0)
    is_high_risk = r.get("is_high_confusion_risk", nearest_dist < threshold)
    
    # 风险等级判定
    if nearest_dist < threshold * 0.5:
        risk_level = "极高混淆风险"
    elif nearest_dist < threshold * 0.75:
        risk_level = "高混淆风险"
    elif nearest_dist < threshold:
        risk_level = "较高混淆风险"
    else:
        risk_level = "低混淆风险"
    
    # ═══ 输出报告 ═══
    print()
    print("  " + "=" * 52)
    print(f"    查询汉字: {char}  ({unicode_str})")
    print(f"    分辨率:   {resolution}")
    print("  " + "-" * 48)
    
    # ── 最近混淆字 ──
    print(f"    最近混淆字:    {nearest}  ({r['top_k_similar'][0]['unicode']})")

    # ── 第二混淆字 ──
    if second and second != nearest:
        second_unicode = r['top_k_similar'][1]['unicode'] if len(top_k) >= 2 else ""
        print(f"    第二混淆字:    {second}  ({second_unicode})")
    
    # ── 距离信息 ──
    print(f"    最近距离:      {nearest_dist:.6f}")
    if second_dist < float('inf'):
        print(f"    第二距离:      {second_dist:.6f}")
    print(f"    Margin:        {margin:.6f}")
    
    print("  " + "-" * 48)
    
    # ── Top-K 相似字 ──
    limit = min(top_k_show, len(top_k))
    print(f"    Top-{limit} 相似字:")
    for i in range(limit):
        nb = top_k[i]
        marker = " <-- 最近" if nb["rank"] == 1 else ""
        print(f"      {nb['rank']}. {nb['char']}  ({nb['unicode']})  距离 = {nb['distance']:.6f}{marker}")
    
    print("  " + "-" * 48)
    
    # ── 混淆风险结果 ──
    risk_yes_no = "是" if is_high_risk else "否"
    print(f"    统一阈值 T:         {threshold:.6f}")
    print(f"    最近距离 < T:       {nearest_dist:.6f} < {threshold:.6f} -> {risk_yes_no}")
    print(f"    是否高混淆风险:     {risk_yes_no}")
    
    if is_high_risk:
        print(f"    混淆风险结果:       !! {risk_level} !!")
        print(f"    风险说明:           该汉字在 {resolution} 分辨率下极易与 '{nearest}' 混淆")
        
        # 混淆簇信息
        cluster = [char]
        for i in range(min(limit, len(top_k))):
            if top_k[i]["distance"] < threshold * 1.5 and top_k[i]["char"] != nearest:
                cluster.append(top_k[i]["char"])
        if len(cluster) > 1:
            print(f"    混淆簇:             {' <-- '.join(cluster)}")
    else:
        print(f"    混淆风险结果:       [OK] 低混淆风险")
    
    print("  " + "=" * 52)
    print()


# ═══════════════════════════════════════════════════════════
# 可视化
# ═══════════════════════════════════════════════════════════

def plot_results(summaries: list[dict], all_per_char: list[dict], output_dir: str, T: float):
    """生成 3 张图表"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    
    resolutions = [s["resolution"] for s in summaries]
    
    # ── 图1: confusion_rate 柱状图 ──
    fig, ax = plt.subplots(figsize=(8, 5))
    rates = [s["confusion_rate"] * 100 for s in summaries]
    bars = ax.bar(resolutions, rates, color=["#e74c3c", "#f39c12", "#2ecc71", "#3498db"])
    ax.axhline(y=20, color="gray", linestyle="--", alpha=0.5, label=f"risk percentile = {20}%")
    ax.set_ylabel("Confusion Rate (%)")
    ax.set_xlabel("Resolution")
    ax.set_title(f"Confusion Rate by Resolution (threshold T = {T:.4f})")
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{rate:.1f}%", ha="center", fontsize=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "confusion_rate_bar.png"), dpi=150)
    plt.close(fig)
    
    # ── 图2: mean_nearest_distance 折线图 ──
    fig, ax = plt.subplots(figsize=(8, 5))
    means = [s["mean_nearest_distance"] for s in summaries]
    ax.plot(resolutions, means, "o-", linewidth=2, markersize=8, color="#2c3e50")
    ax.axhline(y=T, color="red", linestyle="--", alpha=0.6, label=f"threshold T = {T:.4f}")
    ax.set_ylabel("Mean Nearest Distance")
    ax.set_xlabel("Resolution")
    ax.set_title("Mean Cosine Distance to Nearest Neighbor")
    for i, (x, y) in enumerate(zip(resolutions, means)):
        ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "mean_nearest_distance_line.png"), dpi=150)
    plt.close(fig)
    
    # ── 图3: nearest_distance 分布直方图 (分面) ──
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for ax_i, summary in enumerate(summaries):
        ax = axes[ax_i]
        res = summary["resolution"]
        # 找出该分辨率的数据
        subset = [r for r in all_per_char if r.get("resolution") == res]
        dists = [r["nearest_distance"] for r in subset]
        
        ax.hist(dists, bins=40, color="#3498db", edgecolor="white", alpha=0.8)
        ax.axvline(x=T, color="red", linestyle="--", linewidth=1.5, label=f"T = {T:.4f}")
        
        high_risk_dists = [r["nearest_distance"] for r in subset if r["is_high_confusion_risk"]]
        if high_risk_dists:
            ax.hist(high_risk_dists, bins=20, color="#e74c3c", edgecolor="white", alpha=0.6)
        
        ax.set_title(f"{res}  (mean = {summary['mean_nearest_distance']:.4f})")
        ax.set_xlabel("Cosine Distance to Nearest Neighbor")
        ax.set_ylabel("Count")
        ax.legend()
    
    fig.suptitle(f"Nearest-Neighbor Distance Distribution (T = {T:.4f})", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "nearest_distance_histogram.png"), dpi=150)
    plt.close(fig)
    
    print(f"  图表已保存: {output_dir}/")


# ═══════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="自排除最近邻混淆分析 - SPV 图像"
    )
    parser.add_argument(
        "--input-dir",
        default="E:/dataset/char_spv_fixed",
        help="SPV 图像根目录 (含 6x6/ 8x8/ 子目录)"
    )
    parser.add_argument(
        "--output-dir",
        default="E:/results/confusion",
        help="输出目录"
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        default=["6x6", "8x8", "10x10", "12x12"],
        help="要分析的分辨率列表"
    )
    parser.add_argument(
        "--threshold-resolution",
        default="8x8",
        help="用于确定统一阈值 T 的参考分辨率"
    )
    parser.add_argument(
        "--risk-percentile",
        type=float,
        default=20.0,
        help="高风险分位数百分比 (默认 20, 表示最近邻距离最小的前 20%%)"
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=128,
        help="resize 目标尺寸 (默认 128)"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="查询指定汉字 (如 '日'), 空格分隔多个汉字 (如 '日 目 田')"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-K 相似字数量 (默认 5)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("自排除最近邻混淆分析 (Cosine Distance)")
    print("=" * 60)
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"分辨率:   {args.resolutions}")
    print(f"阈值参考分辨率: {args.threshold_resolution}")
    print(f"风险分位数:     {args.risk_percentile}%")
    print(f"Top-K:          {args.top_k}")
    print()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ── Step 1: 逐分辨率分析 ──
    all_per_char = []
    summaries = []
    threshold = None
    
    for res in args.resolutions:
        img_dir = os.path.join(args.input_dir, res)
        if not os.path.exists(img_dir):
            print(f"[SKIP] {img_dir} 不存在")
            continue
        
        print(f"分析 {res} ... ({img_dir})")
        per_char = analyze_resolution(img_dir, args.target_size, args.top_k)
        
        if not per_char:
            continue
        
        # 如果是阈值参考分辨率, 计算统一阈值 T
        if res == args.threshold_resolution:
            threshold = compute_unified_threshold(per_char, args.risk_percentile)
            print(f"  → 统一阈值 T = {threshold} (基于 {res}, top {args.risk_percentile}%)")
        
        all_per_char.extend(per_char)  # 临时存放, 后续 apply_threshold
    
    if threshold is None:
        print(f"[ERROR] 未能从 {args.threshold_resolution} 确定阈值")
        return
    
    # ── Step 2: 用统一阈值 T 重新分析 + 打标签 ──
    all_final = []
    for res in args.resolutions:
        img_dir = os.path.join(args.input_dir, res)
        if not os.path.exists(img_dir):
            continue
        per_char = analyze_resolution(img_dir, args.target_size, args.top_k)
        if not per_char:
            continue
        per_char = apply_threshold(per_char, threshold, res)
        all_final.extend(per_char)
        summary = build_summary(per_char, res)
        summaries.append(summary)
        print(f"\n{res}: 混淆率={summary['confusion_rate']*100:.1f}%  "
              f"(high_risk={summary['num_high_confusion']}/{summary['num_chars']})  "
              f"mean_dist={summary['mean_nearest_distance']:.4f}  "
              f"mean_margin={summary['mean_margin']:.4f}")
    
    # ── Step 3: 输出 CSV ──
    fieldnames = [
        "resolution", "char", "unicode", "nearest_other", "nearest_distance",
        "second_other", "second_distance", "margin", "threshold",
        "is_high_confusion_risk"
    ]
    # 动态添加 top_k 列
    for i in range(1, args.top_k + 1):
        fieldnames.append(f"top{i}_char")
        fieldnames.append(f"top{i}_unicode")
        fieldnames.append(f"top{i}_distance")
    
    per_char_csv = os.path.join(args.output_dir, "per_character_confusion.csv")
    with open(per_char_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in all_final:
            row = {k: r.get(k, "") for k in fieldnames if not k.startswith("top")}
            # 展开 top_k_similar → topN_char / topN_unicode / topN_distance
            for nb in r.get("top_k_similar", []):
                rank = nb["rank"]
                row[f"top{rank}_char"] = nb["char"]
                row[f"top{rank}_unicode"] = nb["unicode"]
                row[f"top{rank}_distance"] = nb["distance"]
            writer.writerow(row)
    print(f"\nper_character CSV → {per_char_csv}")
    
    summary_csv = os.path.join(args.output_dir, "resolution_summary.csv")
    with open(summary_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "resolution", "num_chars", "num_high_confusion", "confusion_rate",
            "mean_nearest_distance", "median_nearest_distance", "mean_margin"
        ])
        writer.writeheader()
        for s in summaries:
            writer.writerow(s)
    print(f"summary CSV → {summary_csv}")
    
    # ── Step 4: 单字查询报告 ──
    if args.query:
        query_chars = [c.strip() for c in args.query.split() if c.strip()]
        print("\n" + "=" * 60)
        print("单字混淆分析报告")
        print("=" * 60)
        
        for qchar in query_chars:
            found = False
            for res in args.resolutions:
                subset = [r for r in all_final if r.get("resolution") == res and r["char"] == qchar]
                if subset:
                    found = True
                    print_char_report(subset[0], threshold, args.top_k)
                    break  # 显示第一个 (参考分辨率) 的报告
            
            if not found:
                print(f"\n  [WARN] 未找到汉字 '{qchar}'")
    
    # ── Step 5: 可视化 ──
    print("\n生成图表...")
    plot_results(summaries, all_final, args.output_dir, threshold)
    
    print("\n全部完成!")


if __name__ == "__main__":
    main()
