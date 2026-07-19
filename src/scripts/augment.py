#!/usr/bin/env python
"""SPV 汉字数据增强生成器"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.augmentation import main

if __name__ == "__main__":
    main()
