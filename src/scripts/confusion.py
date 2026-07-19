#!/usr/bin/env python
"""自排除最近邻混淆分析"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.analysis.confusion import main

if __name__ == "__main__":
    main()
