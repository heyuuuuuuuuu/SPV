"""
模块5: 部件顺序扫描显示模型 (Sequential Component Scanning)

两层架构:
  Layer 1 - generate_scan_sequence(char_path, scan_order, ...)  → 单字帧序列
  Layer 2 - process_all_characters(input_dir, output_dir, ...)  → 批量处理

扫描模式: left-right / top-bottom / center-out / skeleton
"""

import os
import argparse
import numpy as np
from PIL import Image


# ═══════════════════════════════════════════════════════════
# 底层: 高斯光斑渲染
# ═══════════════════════════════════════════════════════════

def _render_spv(grid: np.ndarray, output_size: int = 256, sigma: float = None) -> np.ndarray:
    """将 N×N 二值网格渲染为高斯磷光点仿真图像 (uint8)"""
    N = grid.shape[0]
    if sigma is None:
        sigma = (output_size / N) / 2.5

    cell_size = output_size / N
    yy, xx = np.meshgrid(np.arange(output_size), np.arange(output_size), indexing="ij")

    spv = np.zeros((output_size, output_size), dtype=np.float64)
    for i, j in np.argwhere(grid == 1):
        cx, cy = (j + 0.5) * cell_size, (i + 0.5) * cell_size
        spv += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))

    mn, mx = spv.min(), spv.max()
    if mx - mn < 1e-8:
        return np.zeros_like(spv, dtype=np.uint8)
    return ((spv - mn) / (mx - mn) * 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════
# 底层: 扫描分区
# ═══════════════════════════════════════════════════════════

def _partition(grid: np.ndarray, scan_order: str, frame_num: int, frame_idx: int) -> np.ndarray:
    """
    根据扫描顺序和当前帧索引，返回该帧的点亮掩码。

    scan_order:
      left-right  → 按列分区
      top-bottom  → 按行分区
      center-out  → 按距离中心远近分区
      skeleton    → BFS 拓扑距离分区
    """
    N = grid.shape[0]
    mask = np.zeros_like(grid)
    lit_coords = np.argwhere(grid == 1)
    if len(lit_coords) == 0:
        return mask

    if scan_order == "left-right":
        for r, c in lit_coords:
            if int(c * frame_num / N) == frame_idx:
                mask[r, c] = 1

    elif scan_order == "top-bottom":
        for r, c in lit_coords:
            if int(r * frame_num / N) == frame_idx:
                mask[r, c] = 1

    elif scan_order == "center-out":
        center = (N - 1) / 2.0
        max_dist = np.sqrt(center ** 2 + center ** 2)
        for r, c in lit_coords:
            dist = np.sqrt((r - center) ** 2 + (c - center) ** 2)
            segment = min(int(dist / max_dist * frame_num) if max_dist > 0 else 0, frame_num - 1)
            if segment == frame_idx:
                mask[r, c] = 1

    elif scan_order == "skeleton":
        import collections
        start = tuple(lit_coords[0])
        visited = set()
        queue = collections.deque([(start, 0)])
        distances = {}
        while queue:
            (r, c), d = queue.popleft()
            if (r, c) in visited:
                continue
            visited.add((r, c))
            distances[(r, c)] = d
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < N and 0 <= nc < N and grid[nr, nc] == 1 and (nr, nc) not in visited:
                        queue.append(((nr, nc), d + 1))
        max_d = max(distances.values()) if distances else 1
        for (r, c), d in distances.items():
            segment = min(int(d * frame_num / (max_d + 1)), frame_num - 1)
            if segment == frame_idx:
                mask[r, c] = 1

    else:
        raise ValueError(f"Unknown scan_order: {scan_order}")

    return mask


# ═══════════════════════════════════════════════════════════
# Layer 1: 单字处理
# ═══════════════════════════════════════════════════════════

def generate_scan_sequence(
    char_path: str,
    scan_order: str = "left-right",
    frame_num: int = 4,
    frame_duration: int = 200,
    persistence: bool = True,
    output_size: int = 256,
    sigma: float = None,
) -> list[dict]:
    """
    Layer 1 - 单字扫描序列生成

    Args:
        char_path:      像素化汉字 PNG 路径 (如 "明_U+660E.png")
        scan_order:     扫描模式: left-right / top-bottom / center-out / skeleton
        frame_num:      分帧数
        frame_duration: 每帧持续时间 (ms)
        persistence:    True=累加模式, False=逐帧独立
        output_size:    SPV 输出图像尺寸
        sigma:          高斯光斑 σ (None=自动)

    Returns:
        frames: list of {
            "frame_idx":      int,
            "mask_grid":      np.ndarray,
            "spv":            np.ndarray (uint8),
            "duration_ms":    int,
        }
    """
    # 1. 加载像素化网格
    grid_img = Image.open(char_path)
    grid = (np.array(grid_img) == 255).astype(np.uint8)

    frames = []

    if persistence:
        acc_grid = np.zeros_like(grid)
        for fi in range(frame_num):
            part = _partition(grid, scan_order, frame_num, fi)
            acc_grid = np.clip(acc_grid + part, 0, 1)
            spv = _render_spv(acc_grid, output_size, sigma)
            frames.append({
                "frame_idx": fi,
                "mask_grid": acc_grid.copy(),
                "spv": spv,
                "duration_ms": frame_duration,
            })
    else:
        for fi in range(frame_num):
            part = _partition(grid, scan_order, frame_num, fi)
            spv = _render_spv(part, output_size, sigma)
            frames.append({
                "frame_idx": fi,
                "mask_grid": part.copy(),
                "spv": spv,
                "duration_ms": frame_duration,
            })

    return frames


# ═══════════════════════════════════════════════════════════
# Layer 2: 批量处理
# ═══════════════════════════════════════════════════════════

def process_all_characters(
    input_dir: str,
    output_dir: str,
    scan_modes: list[str] = None,
    frame_num: int = 4,
    persistence: bool = True,
    output_size: int = 256,
    resolution_tag: str = "",
):
    """
    Layer 2 - 批量生成所有汉字的扫描序列

    Args:
        input_dir:       像素化汉字目录 (含 xxx_U+XXXX.png)
        output_dir:      输出根目录
        scan_modes:      扫描模式列表, 默认全部
        frame_num:       分帧数
        persistence:     累加模式
        output_size:     SPV 输出尺寸
        resolution_tag:  分辨率标签 (如 "8x8"), 用于目录命名

    输出结构:
        output_dir/
          ├── left-right/
          │   ├── {char}_U+XXXX/
          │   │   ├── frame_00.png
          │   │   ├── frame_01.png
          │   │   └── ...
          │   └── ...
          ├── top-bottom/
          ├── center-out/
          └── skeleton/
    """
    if scan_modes is None:
        scan_modes = ["left-right", "top-bottom", "center-out", "skeleton"]

    files = sorted([f for f in os.listdir(input_dir) if f.endswith(".png")])
    total = len(files)

    for mode in scan_modes:
        mode_dir = os.path.join(output_dir, mode)
        ok = 0
        for fname in files:
            char = fname.split("_")[0]
            code = ord(char)
            char_dir = os.path.join(mode_dir, f"{char}_U+{code:04X}")
            os.makedirs(char_dir, exist_ok=True)

            path = os.path.join(input_dir, fname)
            try:
                frames = generate_scan_sequence(
                    path,
                    scan_order=mode,
                    frame_num=frame_num,
                    persistence=persistence,
                    output_size=output_size,
                )
                for f in frames:
                    out_path = os.path.join(char_dir, f"frame_{f['frame_idx']:02d}.png")
                    Image.fromarray(f["spv"]).save(out_path)
                ok += 1
            except Exception as e:
                print(f"[ERROR] {mode} {char}: {e}")

        print(f"  [{mode}] {ok}/{total} 完成 → {mode_dir}")

    print(f"\n全部完成! 输出: {output_dir}/")


# ═══════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="部件顺序扫描显示模型")
    parser.add_argument("--input-dir", default="E:/dataset/char_pixelized/8x8",
                        help="单个汉字的输入目录 (单字模式) 或用于批量")
    parser.add_argument("--output-dir", default="E:/dataset/scan_sequence",
                        help="输出目录")
    parser.add_argument("--scan-modes", nargs="+",
                        default=["left-right", "top-bottom", "center-out", "skeleton"],
                        help="扫描模式列表")
    parser.add_argument("--frame-num", type=int, default=4, help="分帧数")
    parser.add_argument("--persistence", action="store_true", default=True)
    parser.add_argument("--no-persistence", action="store_false", dest="persistence")
    parser.add_argument("--batch", action="store_true",
                        help="批量模式: 处理 input-dir 下所有 PNG")
    parser.add_argument("--batch-root", type=str, default=None,
                        help="批量根目录 (如 E:/dataset/char_pixelized, 自动处理 6x6/8x8/...)")
    parser.add_argument("--char", type=str, default=None,
                        help="指定演示汉字 (单字模式)")
    
    args = parser.parse_args()

    if args.batch_root:
        # 批量处理所有分辨率
        for res in ["6x6", "8x8", "10x10", "12x12"]:
            in_dir = os.path.join(args.batch_root, res)
            if not os.path.exists(in_dir):
                continue
            out_dir = os.path.join(args.batch_root.replace("pixelized", "scan"), res)
            print(f"\n{'='*50}\n[Batch] {res}\n{'='*50}")
            process_all_characters(
                in_dir, out_dir, args.scan_modes,
                frame_num=args.frame_num, persistence=args.persistence,
            )

    elif args.batch:
        # 单分辨率批量
        process_all_characters(
            args.input_dir, args.output_dir, args.scan_modes,
            frame_num=args.frame_num, persistence=args.persistence,
        )

    else:
        # 单字演示
        files = [f for f in os.listdir(args.input_dir) if f.endswith(".png")]
        if not files:
            print("没有找到 PNG 文件")
            return

        # 如果指定了 --char, 查找对应文件
        if args.char:
            target = args.char
            found = None
            for f in files:
                if f.startswith(target + "_"):
                    found = f
                    break
            if found is None:
                print(f"未找到汉字 '{target}'")
                return
            fname = found
        else:
            fname = files[0]

        char = fname.split("_")[0]
        path = os.path.join(args.input_dir, fname)

        print(f"演示汉字: {char}  |  分帧: {args.frame_num}  |  累加: {args.persistence}")
        print(f"模式: {args.scan_modes}\n")

        for mode in args.scan_modes:
            frames = generate_scan_sequence(
                path, scan_order=mode,
                frame_num=args.frame_num, persistence=args.persistence,
            )

            mask_dir = os.path.join(args.output_dir, mode)
            os.makedirs(mask_dir, exist_ok=True)

            print(f"  [{mode}]")
            for f in frames:
                lit = int(f["mask_grid"].sum())
                print(f"    Frame {f['frame_idx']}: {lit} lit, {f['duration_ms']}ms")

                # 保存 SPV
                out_path = os.path.join(mask_dir, f"frame_{f['frame_idx']:02d}.png")
                Image.fromarray(f["spv"]).save(out_path)
            print()

        print(f"SPV 已保存: {args.output_dir}/")


if __name__ == "__main__":
    main()