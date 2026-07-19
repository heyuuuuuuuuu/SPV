#!/usr/bin/env python
"""上传数据集到腾讯云 COS"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 直接执行原始上传逻辑
exec(open(os.path.join(os.path.dirname(__file__), "..", "cloud", "upload.py")).read())
