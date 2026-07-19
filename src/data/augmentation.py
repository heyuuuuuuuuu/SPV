"""
模块6: SPV 汉字数据增强生成器 (Data Augmentation for SPV Chinese Character Recognition)

方法:
  1. 加载原始 64×64 二值汉字图像
  2. 轻微几何扰动 (平移 dx/dy)
  3. 像素化到指定分辨率 (6×6 / 8×8 / 10×10 / 12×12 / adaptive)
  4. 电极 dropout + 位置 jitter
  5. 高斯磷光点渲染 (sigma_scale)
  6. 亮度扰动 + 随机噪声
  7. 保存增强 SPV 图像，按 train/val/test 分集

可复现: 同一 (char, augment_id) → 相同随机种子 → 相同增强参数

用法:
  python data_augmentation.py
  python data_augmentation.py --num-aug 30 --resolutions 6x6 8x8 12x12
  python data_augmentation.py --num-aug 10 --resolutions adaptive --split 70 15 15
"""

import os
import csv
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict


# ═══════════════════════════════════════════════════════════
# 配置常量
# ═══════════════════════════════════════════════════════════

# 自适应分辨率映射 (与 adaptive.py 保持一致)
GROUP_TO_RESOLUTION = {
    "low": 8,
    "medium": 10,
    "high": 12,
}

PIXELIZE_THRESHOLD = 0.20   # 像素化点亮阈值
OUTPUT_SIZE = 256           # SPV 输出图像尺寸


# ═══════════════════════════════════════════════════════════
# 核心: 确定性随机参数生成
# ═══════════════════════════════════════════════════════════

def make_rng(char: str, augment_id: int, base_seed: int = 42) -> np.random.RandomState:
    """
    根据汉字 + augment_id 生成确定性随机数生成器
    
    同一 (char, augment_id) → 相同种子 → 相同随机参数
    不同分辨率可用同一 rng 生成相同增强参数，保证公平对比
    """
    # 用字符 Unicode 码点 + augment_id 生成种子
    char_code = ord(char) if len(char) == 1 else hash(char) % 100000
    seed = base_seed * 100000 + char_code * 1000 + augment_id
    return np.random.RandomState(seed)


def generate_aug_params(rng: np.random.RandomState, grid_size: int) -> dict:
    """
    为一次增强生成全部随机参数

    Args:
        rng:       确定性随机数生成器
        grid_size: 当前分辨率 (用于 jitter 尺度缩放)
    
    Returns:
        dict: {
            dx, dy:          平移量 (像素, int)
            brightness:      亮度系数 (float)
            sigma_scale:     高斯 sigma 缩放 (float)
            jitter:          位置抖动 (像素, float, 在原始坐标空间)
            noise_std:       噪声标准差 (float)
            dropout:         电极丢失率 (float)
        }
    """
    return {
        "dx":           int(rng.uniform(-2, 2)),           # -2~2 像素平移
        "dy":           int(rng.uniform(-2, 2)),
        "brightness":   float(rng.uniform(0.8, 1.2)),       # 0.8~1.2 亮度
        "sigma_scale":  float(rng.uniform(0.8, 1.3)),       # 0.8~1.3 sigma 缩放
        "jitter":       float(rng.uniform(0.0, 1.0)),       # 0~1 像素抖动
        "noise_std":    float(rng.uniform(0.0, 0.03)),      # 0~0.03 噪声
        "dropout":      float(rng.uniform(0.0, 0.05)),      # 0~5% 电极丢失
    }


# ═══════════════════════════════════════════════════════════
# 增强流水线各步骤
# ═══════════════════════════════════════════════════════════

def apply_translation(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """
    对二值图像做亚像素级平移 (pad + crop)
    
    Args:
        image: (H, W) 二值数组, 255=笔画
        dx, dy: 平移量 (像素)
    
    Returns:
        同尺寸平移后的图像
    """
    if dx == 0 and dy == 0:
        return image.copy()
    
    h, w = image.shape
    # 创建平移矩阵
    translated = np.zeros_like(image)
    
    # 计算源区域和目标区域的重叠部分
    src_y0 = max(0, -dy)
    src_y1 = min(h, h - dy)
    dst_y0 = max(0, dy)
    dst_y1 = min(h, h + dy)
    
    src_x0 = max(0, -dx)
    src_x1 = min(w, w - dx)
    dst_x0 = max(0, dx)
    dst_x1 = min(w, w + dx)
    
    # 确保高度和宽度一致
    h_src = src_y1 - src_y0
    h_dst = dst_y1 - dst_y0
    w_src = src_x1 - src_x0
    w_dst = dst_x1 - dst_x0
    
    if h_src > 0 and w_src > 0 and h_dst > 0 and w_dst > 0:
        # 取最小公共区域
        h_crop = min(h_src, h_dst)
        w_crop = min(w_src, w_dst)
        translated[dst_y0:dst_y0 + h_crop, dst_x0:dst_x0 + w_crop] = \
            image[src_y0:src_y0 + h_crop, src_x0:src_x0 + w_crop]
    
    return translated


def pixelize_grid(
    image: np.ndarray,
    grid_size: int,
    threshold: float = PIXELIZE_THRESHOLD,
) -> np.ndarray:
    """
    将图像像素化到 grid_size × grid_size 二值网格
    
    Args:
        image:     (H, W) 二值数组, 255=笔画
        grid_size: 目标网格尺寸
        threshold: 点亮阈值 (格内笔画像素比例 > threshold 则点亮)
    
    Returns:
        (grid_size, grid_size) 二值网格, 1=点亮
    """
    h, w = image.shape
    cell_h = h // grid_size
    cell_w = w // grid_size
    
    grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    
    for i in range(grid_size):
        for j in range(grid_size):
            y0, y1 = i * cell_h, (i + 1) * cell_h
            x0, x1 = j * cell_w, (j + 1) * cell_w
            cell = image[y0:y1, x0:x1]
            
            # 笔画像素占比
            stroke_ratio = (cell == 255).sum() / cell.size if cell.size > 0 else 0.0
            
            if stroke_ratio > threshold:
                grid[i, j] = 1
    
    return grid


def apply_dropout_and_jitter(
    grid: np.ndarray,
    dropout: float,
    jitter_px: float,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """
    对点亮电极做 dropout (随机熄灭) 和位置 jitter
    
    Args:
        grid:      (N, N) 二值网格, 1=点亮
        dropout:   电极丢失概率
        jitter_px: 位置抖动幅度 (像素, 在原始 64×64 坐标系)
        rng:       随机数生成器
    
    Returns:
        (grid_after_dropout, jittered_positions)
        jittered_positions: [(row, col), ...] 抖动后的电极坐标 (行列索引, 含子格偏移)
    """
    N = grid.shape[0]
    # dropout: 对每个点亮的电极, 以概率 dropout 熄灭
    lit_mask = (grid == 1)
    lit_indices = np.argwhere(lit_mask)  # [(r, c), ...]
    
    # 用 rng 决定哪些电极存活
    keep = rng.random(len(lit_indices)) >= dropout
    surviving = lit_indices[keep]
    
    # 构建 dropout 后的网格
    grid_after = np.zeros_like(grid)
    jittered = []
    
    for r, c in surviving:
        # 位置 jitter: 在 [-jitter_px, +jitter_px] 范围内均匀随机偏移
        # jitter 是在 64×64 原始图像坐标系下的像素偏移, 
        # 需要转换为网格内的子格偏移
        cell_h = 64.0 / N
        cell_w = 64.0 / N
        
        jr = rng.uniform(-jitter_px, jitter_px) / cell_h  # 转为行列偏移
        jc = rng.uniform(-jitter_px, jitter_px) / cell_w
        
        grid_after[r, c] = 1
        jittered.append((float(r) + jr, float(c) + jc))
    
    return grid_after, jittered


def render_spv_with_jitter(
    grid_after_dropout: np.ndarray,
    jittered_positions: list[tuple[float, float]],
    grid_size: int,
    sigma: float,
    output_size: int = OUTPUT_SIZE,
) -> np.ndarray:
    """
    渲染高斯磷光点 SPV 图像 (支持 jitter 子格偏移)
    
    Args:
        grid_after_dropout: dropout 后的二值网格
        jittered_positions: 抖动后的电极坐标 (行, 列, 含子格偏移)
        grid_size:          原始网格尺寸 N
        sigma:              高斯标准差
        output_size:        输出图像尺寸
    
    Returns:
        (output_size, output_size) SPV 图像 (float, 未归一化)
    """
    cell_size = output_size / grid_size
    
    y_coords = np.arange(output_size)
    x_coords = np.arange(output_size)
    yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")
    
    spv = np.zeros((output_size, output_size), dtype=np.float64)
    
    for (rj, cj) in jittered_positions:
        # jittered 坐标 (行, 列) 转为输出图像中的中心坐标
        cx = (cj + 0.5) * cell_size  # cj 已包含 jitter
        cy = (rj + 0.5) * cell_size  # rj 已包含 jitter
        
        dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
        blob = np.exp(-dist_sq / (2 * sigma ** 2))
        spv += blob
    
    return spv


# ═══════════════════════════════════════════════════════════
# 主增强流水线
# ═══════════════════════════════════════════════════════════

def augment_one_sample(
    char: str,
    orig_image: np.ndarray,
    augment_id: int,
    resolution_config: str,          # "6", "8", "10", "12" 或 "adaptive"
    complexity_level: str = None,    # 仅 adaptive 时需要
    base_seed: int = 42,
    output_size: int = OUTPUT_SIZE,
) -> dict:
    """
    对单个汉字生成一个增强样本

    Args:
        char:              汉字字符
        orig_image:        (64, 64) 二值原始图像
        augment_id:        增强编号 (0, 1, 2, ...)
        resolution_config: "6"|"8"|"10"|"12"|"adaptive"
        complexity_level:  "low"|"medium"|"high" (仅 adaptive)
        base_seed:         基础随机种子
        output_size:       SPV 输出尺寸
    
    Returns:
        dict: {
            spv_image: (output_size, output_size) uint8 图像
            grid_size: 实际使用的网格大小
            params:    {dx, dy, brightness, sigma_scale, jitter, noise_std, dropout}
        }
    """
    # ── 1. 确定分辨率 ──
    if resolution_config == "adaptive":
        if complexity_level is None:
            raise ValueError("adaptive 分辨率需要 complexity_level")
        grid_size = GROUP_TO_RESOLUTION.get(complexity_level, 8)
    else:
        grid_size = int(resolution_config)
    
    # ── 2. 确定性随机参数 ──
    rng = make_rng(char, augment_id, base_seed)
    params = generate_aug_params(rng, grid_size)
    
    # ── 3. 几何扰动 (平移) ──
    translated = apply_translation(orig_image, params["dx"], params["dy"])
    
    # ── 4. 像素化 ──
    grid = pixelize_grid(translated, grid_size, PIXELIZE_THRESHOLD)
    
    # ── 5. Dropout + Jitter ──
    grid_drop, jittered = apply_dropout_and_jitter(
        grid, params["dropout"], params["jitter"], rng
    )
    
    # ── 6. 高斯磷光点渲染 ──
    # 基础 sigma (与 spv_render.py 一致)
    base_sigma = (output_size / grid_size) / 2.5
    sigma = base_sigma * params["sigma_scale"]
    
    spv = render_spv_with_jitter(
        grid_drop, jittered, grid_size, sigma, output_size
    )
    
    # ── 7. 亮度扰动 ──
    spv = spv * params["brightness"]
    
    # ── 8. 随机噪声 ──
    noise = rng.randn(output_size, output_size).astype(np.float64) * params["noise_std"]
    spv = spv + noise * spv.max()  # 按峰值缩放噪声幅度
    
    # ── 9. 归一化到 uint8 ──
    spv = np.clip(spv, 0, None)
    spv_min = spv.min()
    spv_max = spv.max()
    if spv_max - spv_min > 1e-8:
        spv_uint8 = ((spv - spv_min) / (spv_max - spv_min) * 255).astype(np.uint8)
    else:
        spv_uint8 = np.zeros_like(spv, dtype=np.uint8)
    
    return {
        "spv_image": spv_uint8,
        "grid_size": grid_size,
        "params": params,
    }


# ═══════════════════════════════════════════════════════════
# 数据集划分
# ═══════════════════════════════════════════════════════════

def split_augment_ids(
    num_aug: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    mode: str = "interleave",
) -> dict[int, str]:
    """
    将 augment_id (0..num_aug-1) 分配到 train/val/test
    
    策略: 使用 interleave 保证每个 split 均匀分布 augment_id, 
          从而每个汉字在 train/val/test 中都有样本
    
    Args:
        num_aug:      每个汉字的增强样本总数
        train_ratio:  训练集比例
        val_ratio:    验证集比例
        test_ratio:   测试集比例
        mode:         "interleave" → 轮转分配, 确保每份均匀
    
    Returns:
        {augment_id: "train"|"val"|"test"}
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001
    
    n_train = int(num_aug * train_ratio)
    n_val   = int(num_aug * val_ratio)
    n_test  = num_aug - n_train - n_val
    # 确保每个 split 至少有 1 个样本
    if num_aug >= 3:
        if n_test == 0:
            n_test = 1
            if n_val > 1: n_val -= 1
            elif n_train > 1: n_train -= 1
        if n_val == 0:
            n_val = 1
            if n_train > 1: n_train -= 1
        if n_train == 0:
            n_train = 1
    
    assignment = {}
    idx = 0
    
    if mode == "interleave":
        # 轮转分配: train, train, ..., val, test, train, train, ...
        # 按比例分配槽位
        slots = (["train"] * n_train) + (["val"] * n_val) + (["test"] * n_test)
        # 打乱槽位顺序 (但保持确定性)
        rng = np.random.RandomState(42)
        rng.shuffle(slots)
        for aug_id in range(num_aug):
            assignment[aug_id] = slots[aug_id]
    else:
        # 简单连续分配
        for aug_id in range(num_aug):
            if idx < n_train:
                assignment[aug_id] = "train"
            elif idx < n_train + n_val:
                assignment[aug_id] = "val"
            else:
                assignment[aug_id] = "test"
            idx += 1
    
    return assignment


# ═══════════════════════════════════════════════════════════
# 批量生成
# ═══════════════════════════════════════════════════════════

def load_complexity_map(complexity_csv: str) -> dict[str, str]:
    """从 BPSCA 复杂度 CSV 加载 char → level 映射"""
    char_level = {}
    with open(complexity_csv, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            char_level[row["汉字"]] = row["等级"]
    return char_level


def load_original_images(input_dir: str, image_size: int = 64) -> dict[str, np.ndarray]:
    """
    加载所有原始汉字图像

    Returns:
        {char: (image_size, image_size) ndarray}
    """
    img_folder = os.path.join(input_dir, f"{image_size}x{image_size}")
    if not os.path.exists(img_folder):
        raise FileNotFoundError(f"原始图像目录不存在: {img_folder}")
    
    originals = {}
    for fname in sorted(os.listdir(img_folder)):
        if not fname.endswith(".png"):
            continue
        # 文件名格式: {char}_U+{code}.png
        char = fname.split("_")[0]
        path = os.path.join(img_folder, fname)
        img = np.array(Image.open(path))
        originals[char] = img
    
    print(f"  加载 {len(originals)} 张原始图像")
    return originals


def generate_dataset(
    input_dir: str,
    output_dir: str,
    resolutions: list[str],          # ["6", "8", "10", "12", "adaptive"]
    num_aug: int = 30,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    complexity_csv: str = None,
    base_seed: int = 42,
    output_size: int = OUTPUT_SIZE,
    max_chars: int = None,
) -> list[dict]:
    """
    批量生成增强数据集

    Args:
        input_dir:      原始汉字图像根目录 (含 64x64/ 子目录)
        output_dir:     输出根目录
        resolutions:    分辨率列表 ["6","8","10","12","adaptive"]
        num_aug:        每个汉字每分辨率的增强样本数
        train_ratio:    训练集比例
        val_ratio:      验证集比例
        test_ratio:     测试集比例
        complexity_csv: BPSCA 复杂度 CSV 路径 (adaptive 时需要)
        base_seed:      基础随机种子
        output_size:    SPV 输出图像尺寸
        max_chars:      限制处理的汉字数 (None=全部)

    Returns:
        labels: 全部样本的标签列表
    """
    # ── 加载 ──
    print("\n[1] 加载原始图像...")
    originals = load_original_images(input_dir)
    all_chars = sorted(originals.keys())
    
    char_level_map = {}
    if "adaptive" in resolutions:
        if complexity_csv is None:
            raise ValueError("--complexity-csv 必须指定 (adaptive 分辨率需要)")
        char_level_map = load_complexity_map(complexity_csv)
        print(f"  加载 {len(char_level_map)} 个汉字的复杂度等级")
    
    # ── 数据划分 ──
    split_map = split_augment_ids(num_aug, train_ratio, val_ratio, test_ratio)
    
    # ── 生成 ──
    labels = []
    total_samples = 0
    chars_to_process = all_chars[:max_chars] if max_chars else all_chars
    n_chars = len(chars_to_process)
    
    for res_config in resolutions:
        res_label = f"{res_config}x{res_config}" if res_config != "adaptive" else "adaptive"
        print(f"\n[2] 生成 {res_label} 分辨率...")
        
        for ci, char in enumerate(chars_to_process):
            orig = originals[char]
            level = char_level_map.get(char, "medium") if res_config == "adaptive" else None
            
            for aug_id in range(num_aug):
                split = split_map[aug_id]
                
                # 执行增强
                result = augment_one_sample(
                    char, orig, aug_id,
                    resolution_config=res_config,
                    complexity_level=level,
                    base_seed=base_seed,
                    output_size=output_size,
                )
                
                # 确定实际分辨率字符串
                gs = result["grid_size"]
                actual_res = f"{gs}x{gs}"
                
                # 构建输出路径: 分辨率在上层, split 在下层
                # outputs/augmented_spv/6x6/train/...
                if res_config == "adaptive":
                    res_dir = "adaptive"
                else:
                    res_dir = f"{res_config}x{res_config}"
                split_dir = os.path.join(output_dir, res_dir, split)
                os.makedirs(split_dir, exist_ok=True)
                
                fname = f"{char}_U+{ord(char):04X}_aug{aug_id:03d}.png"
                out_path = os.path.join(split_dir, fname)
                
                # 保存 SPV 图像
                Image.fromarray(result["spv_image"]).save(out_path)
                
                # 记录标签
                p = result["params"]
                labels.append({
                    "image_path": out_path,
                    "label": char,
                    "label_unicode": f"U+{ord(char):04X}",
                    "resolution": actual_res,
                    "augment_id": aug_id,
                    "dx": p["dx"],
                    "dy": p["dy"],
                    "brightness": round(p["brightness"], 4),
                    "sigma_scale": round(p["sigma_scale"], 4),
                    "jitter": round(p["jitter"], 4),
                    "noise_std": round(p["noise_std"], 6),
                    "dropout": round(p["dropout"], 4),
                    "split": split,
                })
                
                total_samples += 1
            
            # 进度
            if (ci + 1) % 200 == 0:
                print(f"    [{ci + 1}/{n_chars}] {total_samples} samples generated...")
        
        print(f"    {res_label}: 完成 ({n_chars * num_aug} samples)")
    
    # ── 保存 labels.csv ──
    labels_csv = os.path.join(output_dir, "labels.csv")
    fieldnames = [
        "image_path", "label", "label_unicode", "resolution",
        "augment_id", "dx", "dy", "brightness", "sigma_scale",
        "jitter", "noise_std", "dropout", "split",
    ]
    with open(labels_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in labels:
            writer.writerow(row)
    
    print(f"\n  labels.csv → {labels_csv}  ({len(labels)} rows)")
    
    # ── 统计 ──
    print("\n" + "=" * 60)
    print("数据集统计")
    print("=" * 60)
    splits = defaultdict(int)
    res_counts = defaultdict(int)
    for row in labels:
        splits[row["split"]] += 1
        res_counts[row["resolution"]] += 1
    
    print(f"  总样本数:          {len(labels)}")
    print(f"  汉字数:            {n_chars}")
    print(f"  每汉字增广数:      {num_aug}")
    for s in ["train", "val", "test"]:
        pct = splits[s] / len(labels) * 100 if labels else 0
        print(f"  {s}:  {splits[s]:>8d}  ({pct:.1f}%)")
    print(f"  实际网格分辨率分布:")
    for r in sorted(res_counts.keys()):
        print(f"    {r}: {res_counts[r]:>8d}")
    
    return labels


# ═══════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SPV 汉字数据增强 — 模拟假体视觉变化"
    )
    parser.add_argument(
        "--input-dir",
        default="E:/dataset/char_rendered_hei",
        help="原始汉字图像根目录 (含 64x64/ 子目录)"
    )
    parser.add_argument(
        "--output-dir",
        default="E:/dataset/augmented_spv",
        help="增强后 SPV 图像输出目录"
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        default=["6", "8", "10", "12", "adaptive"],
        help="目标分辨率: 6 8 10 12 adaptive (默认全部)"
    )
    parser.add_argument(
        "--num-aug",
        type=int,
        default=30,
        help="每个汉字每分辨率的增强样本数 (默认 30)"
    )
    parser.add_argument(
        "--split",
        nargs=3,
        type=float,
        default=[70, 15, 15],
        help="train/val/test 比例 (默认 70 15 15)"
    )
    parser.add_argument(
        "--complexity-csv",
        default="E:/dataset/complexity_scores.csv",
        help="BPSCA 复杂度 CSV (adaptive 分辨率需要)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="基础随机种子 (默认 42)"
    )
    parser.add_argument(
        "--output-size",
        type=int,
        default=256,
        help="SPV 输出图像尺寸 (默认 256)"
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="限制处理的汉字数量 (默认全部, 用于快速测试)"
    )
    
    args = parser.parse_args()
    
    # 转换比例
    train_r, val_r, test_r = [x / 100.0 for x in args.split]
    
    print("=" * 60)
    print("SPV 汉字数据增强生成器")
    print("=" * 60)
    print(f"输入目录:     {args.input_dir}")
    print(f"输出目录:     {args.output_dir}")
    print(f"分辨率:       {args.resolutions}")
    print(f"每字增广数:   {args.num_aug}")
    print(f"划分比例:     train={train_r:.0%} val={val_r:.0%} test={test_r:.0%}")
    print(f"随机种子:     {args.seed}")
    print()
    
    # 设置全局随机种子
    np.random.seed(args.seed)
    
    # 生成数据集
    generate_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        resolutions=args.resolutions,
        num_aug=args.num_aug,
        train_ratio=train_r,
        val_ratio=val_r,
        test_ratio=test_r,
        complexity_csv=args.complexity_csv,
        base_seed=args.seed,
        output_size=args.output_size,
        max_chars=args.max_chars,
    )
    
    print("\n全部完成!")


if __name__ == "__main__":
    main()
