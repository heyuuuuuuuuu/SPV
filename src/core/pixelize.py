"""
模块2: 固定分辨率像素化模型
输入: 64×64 二值汉字图 (黑底白字)
输出: 6×6 / 8×8 / 10×10 / 12×12 像素化汉字图
方法: 网格划分 → 每格笔画像素比例 → 阈值 0.20 → 二值点亮
"""

import os
import numpy as np
from PIL import Image


def pixelize(img_path: str, grid_size: int, threshold: float = 0.20) -> Image.Image:
    """
    将 64×64 二值汉字图像素化为 grid_size × grid_size

    Args:
        img_path:   输入图像路径
        grid_size:  目标网格 (6/8/10/12)
        threshold:  点亮阈值，格内笔画像素比例 > threshold 则点亮

    Returns:
        grid_size × grid_size 的二值 PIL Image (黑底白字)
    """
    img = Image.open(img_path)
    arr = np.array(img).astype(np.float64)

    h, w = arr.shape
    cell_h = h // grid_size
    cell_w = w // grid_size

    output = np.zeros((grid_size, grid_size), dtype=np.uint8)

    for i in range(grid_size):
        for j in range(grid_size):
            y0, y1 = i * cell_h, (i + 1) * cell_h
            x0, x1 = j * cell_w, (j + 1) * cell_w
            cell = arr[y0:y1, x0:x1]

            # 笔画（白色=255）像素占比
            stroke_ratio = (cell == 255).sum() / cell.size

            if stroke_ratio > threshold:
                output[i, j] = 255  # 点亮

    return Image.fromarray(output)


def batch_pixelize(
    input_dir: str,
    output_base: str,
    grid_sizes: tuple[int, ...] = (6, 8, 10, 12),
    threshold: float = 0.20,
    input_size: int = 64,
):
    """
    批量像素化所有汉字
    """
    folder = os.path.join(input_dir, f"{input_size}x{input_size}")
    files = sorted([f for f in os.listdir(folder) if f.endswith(".png")])

    for gs in grid_sizes:
        out_dir = os.path.join(output_base, f"{gs}x{gs}")
        os.makedirs(out_dir, exist_ok=True)
        count = 0

        for fname in files:
            char = fname.split("_")[0]
            code = ord(char)
            in_path = os.path.join(folder, fname)

            try:
                img = pixelize(in_path, grid_size=gs, threshold=threshold)
                out_name = f"{char}_U+{code:04X}.png"
                img.save(os.path.join(out_dir, out_name))
                count += 1
            except Exception as e:
                print(f"[ERROR] {char} {gs}x{gs}: {e}")

        print(f"  {gs}x{gs}: {count} 张 → {out_dir}")
