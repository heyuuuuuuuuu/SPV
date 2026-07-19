"""
预览渲染后的汉字二值图
用法: python preview.py          → 黑体 64x64 随机100个
用法: python preview.py 128      → 黑体 128x128 随机100个
用法: python preview.py 明       → 黑体 单个汉字
"""

import sys
import os
import random
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image

from src.config import RENDERED_DIR

# 修复中文乱码
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "KaiTi", "SimSun"]
matplotlib.rcParams["axes.unicode_minus"] = False

BASE_DIR = str(RENDERED_DIR)


def preview_grid(size: str = "64", n: int = 100):
    """随机展示 n 个汉字，排列成网格"""
    folder = f"{BASE_DIR}/{size}x{size}"
    files = [f for f in os.listdir(folder) if f.endswith(".png")]

    if n > len(files):
        n = len(files)

    samples = random.sample(files, n)
    samples.sort(key=lambda f: int(f.split("_U+")[1].split(".")[0], 16))

    cols = min(20, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 0.6, rows * 0.6))
    fig.suptitle(f"黑体 {size}×{size} 汉字二值图 (共 {n} 个)", fontsize=14)

    for i, ax in enumerate(axes.flat):
        if i < n:
            img = Image.open(os.path.join(folder, samples[i]))
            char = samples[i].split("_")[0]
            ax.imshow(img, cmap="gray")
            ax.set_title(char, fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def preview_single(size: str = "64", char: str = "明"):
    """单独展示某个汉字"""
    folder = f"{BASE_DIR}/{size}x{size}"
    code = ord(char)
    filename = f"{char}_U+{code:04X}.png"
    path = os.path.join(folder, filename)

    if not os.path.exists(path):
        for f in os.listdir(folder):
            if f.startswith(f"{char}_"):
                path = os.path.join(folder, f)
                break

    if not os.path.exists(path):
        print(f"未找到汉字 '{char}'")
        return

    img = Image.open(path)
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title(f"{char} ({size}×{size})")
    axes[0].axis("off")

    axes[1].imshow(img, cmap="gray", interpolation="nearest")
    axes[1].set_title(f"{char} - 像素格放大")
    axes[1].set_xticks(range(0, int(size), 8))
    axes[1].set_yticks(range(0, int(size), 8))
    axes[1].grid(True, color="red", linewidth=0.3, alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    size = "64"
    char_arg = None

    for arg in sys.argv[1:]:
        if arg in ("64", "128"):
            size = arg
        elif len(arg) == 1 and "\u4e00" <= arg <= "\u9fff":
            char_arg = arg

    if char_arg:
        preview_single(size, char_arg)
    else:
        preview_grid(size, n=100)
