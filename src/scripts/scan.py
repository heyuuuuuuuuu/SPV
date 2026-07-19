#!/usr/bin/env python
"""部件顺序扫描显示模型"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.scan import main

if __name__ == "__main__":
    main()
