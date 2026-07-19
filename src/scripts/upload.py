#!/usr/bin/env python
"""将数据集打包为单个归档并上传到腾讯云 COS。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.cloud.archive_transfer import main


if __name__ == "__main__":
    main(["upload", *sys.argv[1:]])
