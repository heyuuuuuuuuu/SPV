#!/usr/bin/env python
"""渲染汉字为二值图像 (黑底白字)"""
import sys
import os

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.render import CharRenderer, load_chars_from_file, batch_render

INPUT_FILE = "E:/dataset/char_handwriting_Chinese.txt"
OUTPUT_DIR_64 = "E:/dataset/char_rendered_hei/64x64"
OUTPUT_DIR_128 = "E:/dataset/char_rendered_hei/128x128"
FONT_PATH = "C:/Windows/Fonts/simhei.ttf"

if __name__ == "__main__":
    print(f"Loading chars from: {INPUT_FILE}")
    chars = load_chars_from_file(INPUT_FILE)
    print(f"Total unique chars: {len(chars)}")

    print("\n[1/2] Rendering 64x64 (HeiTi)...")
    batch_render(chars, FONT_PATH, OUTPUT_DIR_64, 64, 200, 128, True)

    print("\n[2/2] Rendering 128x128 (HeiTi)...")
    batch_render(chars, FONT_PATH, OUTPUT_DIR_128, 128, 400, 128, True)

    print("\nAll done!")
