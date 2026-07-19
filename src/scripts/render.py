#!/usr/bin/env python
"""渲染汉字为二值图像（黑底白字）。"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import DATA_DIR, RENDERED_DIR
from src.core.render import batch_render, load_chars_from_file


def main() -> None:
    parser = argparse.ArgumentParser(description="渲染汉字为二值图像")
    parser.add_argument("--input-file", default=str(DATA_DIR / "char_handwriting_Chinese.txt"))
    parser.add_argument("--output-dir", default=str(RENDERED_DIR))
    parser.add_argument("--font-path", required=True, help="支持汉字的 TTF/OTF 字体路径")
    args = parser.parse_args()

    chars = load_chars_from_file(args.input_file)
    print(f"加载 {len(chars)} 个汉字")
    for size, font_size in ((64, 200), (128, 400)):
        output_dir = os.path.join(args.output_dir, f"{size}x{size}")
        print(f"渲染 {size}x{size} → {output_dir}")
        batch_render(chars, args.font_path, output_dir, size, font_size, 128, True)


if __name__ == "__main__":
    main()
