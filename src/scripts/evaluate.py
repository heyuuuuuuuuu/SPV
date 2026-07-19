#!/usr/bin/env python
"""SPV 汉字识别模型评估"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import COMPLEXITY_CSV, LABELS_CSV, OUTPUTS_DIR
from src.training.evaluator import evaluate

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SPV 汉字识别模型评估")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--labels-csv", default=str(LABELS_CSV))
    parser.add_argument("--output-dir", default=str(OUTPUTS_DIR / "evaluate_recognition"))
    parser.add_argument("--resolution", default=None)
    parser.add_argument("--scan-mode", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--complexity-csv", default=str(COMPLEXITY_CSV))

    args = parser.parse_args()

    print("=" * 60)
    print("SPV 汉字识别 — 模型评估")
    print("=" * 60)

    evaluate(
        checkpoint_path=args.checkpoint,
        labels_csv=args.labels_csv,
        output_dir=args.output_dir,
        resolution=args.resolution,
        scan_mode=args.scan_mode,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers,
        complexity_csv=args.complexity_csv,
    )
