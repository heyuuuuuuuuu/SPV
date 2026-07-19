#!/usr/bin/env python
"""预览渲染后的汉字二值图"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.preview import preview_grid, preview_single

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
