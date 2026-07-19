#!/usr/bin/env python
"""SPV 汉字识别模型训练"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import COMPLEXITY_CSV, LABELS_CSV, OUTPUTS_DIR
from src.training.trainer import train

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SPV 汉字识别模型训练")
    parser.add_argument("--labels-csv", default=str(LABELS_CSV))
    parser.add_argument("--output-dir", default=str(OUTPUTS_DIR / "train_recognition"))
    parser.add_argument("--model", default="light_cnn", choices=["light_cnn", "resnet18"])
    parser.add_argument("--resolution", default=None)
    parser.add_argument("--scan-mode", default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--complexity-csv", default=str(COMPLEXITY_CSV))

    args = parser.parse_args()

    print("=" * 60)
    print("SPV 汉字识别 — 模型训练")
    print("=" * 60)

    train(
        labels_csv=args.labels_csv,
        output_dir=args.output_dir,
        model_name=args.model,
        resolution=args.resolution,
        scan_mode=args.scan_mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        complexity_csv=args.complexity_csv,
    )
