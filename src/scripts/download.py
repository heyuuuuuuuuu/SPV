#!/usr/bin/env python
"""从腾讯云 COS 下载数据集"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

exec(open(os.path.join(os.path.dirname(__file__), "..", "cloud", "download.py")).read())
