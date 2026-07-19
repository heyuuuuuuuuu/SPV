"""
模块4: 高斯磷光点仿真模型
输入: N×N 二值像素化网格 (0/1)
输出: 连续灰度 SPV 磷光图像
方法: 每个点亮格子 → 高斯光斑 → 叠加 → 归一化
"""

import os
import numpy as np
from PIL import Image


def gaussian_phosphene(
    grid: np.ndarray,
    output_size: int = 256,
    sigma: float = None,
    brightness: float = 1.0,
) -> np.ndarray:
    """
    将二值网格渲染为高斯磷光点仿真图像

    Args:
        grid:         N×N 二值数组, 1=点亮, 0=熄灭
        output_size:  输出图像尺寸 (output_size × output_size)
        sigma:        高斯标准差 (None=自动按单元格大小计算)
        brightness:   峰值亮度系数

    Returns:
        output_size × output_size 的灰度图像数组 (float, 未归一化)
    """
    N = grid.shape[0]

    if sigma is None:
        # 自动: sigma = 单元格宽度 / 2.5 (经验值,光斑柔和扩散)
        sigma = (output_size / N) / 2.5

    # 每个格子的中心坐标 (在 output 坐标系中)
    cell_size = output_size / N

    # 创建坐标网格
    y_coords = np.arange(output_size)
    x_coords = np.arange(output_size)
    yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")

    # 初始化输出
    spv_image = np.zeros((output_size, output_size), dtype=np.float64)

    # 对每个点亮格子叠加高斯光斑
    lit_positions = np.argwhere(grid == 1)
    for i, j in lit_positions:
        cx = (j + 0.5) * cell_size  # 中心 x
        cy = (i + 0.5) * cell_size  # 中心 y

        # 高斯: exp(-dist² / (2σ²))
        dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
        blob = brightness * np.exp(-dist_sq / (2 * sigma ** 2))
        spv_image += blob

    return spv_image


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """归一化到 0-255 uint8"""
    img_min = img.min()
    img_max = img.max()
    if img_max - img_min < 1e-8:
        return np.zeros_like(img, dtype=np.uint8)
    normalized = (img - img_min) / (img_max - img_min) * 255
    return normalized.astype(np.uint8)


def render_and_save(
    grid_path: str,
    output_path: str,
    output_size: int = 256,
    sigma: float = None,
):
    """读取像素化网格, 渲染高斯光斑, 保存"""
    grid_img = Image.open(grid_path)
    grid = (np.array(grid_img) == 255).astype(np.uint8)

    spv = gaussian_phosphene(grid, output_size=output_size, sigma=sigma)
    spv_uint8 = normalize_to_uint8(spv)
    img = Image.fromarray(spv_uint8)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
    return spv_uint8


def batch_render(
    input_base: str,
    output_base: str,
    grid_sizes: tuple[int, ...] = (6, 8, 10, 12),
    output_size: int = 256,
    sigma: float = None,
):
    """
    批量渲染所有像素化汉字为 SPV 磷光图像
    input_base:   char_pixelized 或 char_adaptive_pixelized 根目录
    output_base:  输出根目录
    """
    for gs in grid_sizes:
        in_dir = os.path.join(input_base, f"{gs}x{gs}")
        out_dir = os.path.join(output_base, f"{gs}x{gs}")
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.exists(in_dir):
            continue

        files = [f for f in os.listdir(in_dir) if f.endswith(".png")]
        count = 0
        for fname in files:
            in_path = os.path.join(in_dir, fname)
            out_path = os.path.join(out_dir, fname)
            try:
                render_and_save(in_path, out_path, output_size, sigma)
                count += 1
            except Exception as e:
                print(f"[ERROR] {fname}: {e}")

        print(f"  {gs}x{gs} → {count} 张  {out_dir}")
