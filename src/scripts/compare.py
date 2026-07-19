#!/usr/bin/env python
"""固定分辨率与自适应分辨率对比评估入口。"""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

if __name__ == "__main__":
    runpy.run_module("src.analysis.compare", run_name="__main__")
