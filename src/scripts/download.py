#!/usr/bin/env python
"""从腾讯云 COS 下载单个数据集归档，校验后解包。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.cloud.archive_transfer import main


if __name__ == "__main__":
    main(["download", *sys.argv[1:]])
