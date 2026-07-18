"""
评估脚本: 加载训练好的 checkpoint, 在 test set 上全面评估

输出:
  - test_metrics.json          整体指标
  - test_predictions.csv       逐样本预测
  - confusion_matrix.csv       混淆矩阵
  - per_class_accuracy.csv     每类准确率
  - per_group_accuracy.csv     分组准确率 (低/中/高复杂度)

用法:
  python evaluate_recognition.py --checkpoint outputs/train_recognition/best_model.pth
  python evaluate_recognition.py --checkpoint best_model.pth --labels-csv data.csv
"""

import os
import csv
import json
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader

from models import create_model
from dataset import SPVCharDataset
from utils_metrics import (
    topk_accuracy,
    topk_predictions,
    softmax,
    compute_confusion_matrix,
    compute_raw_confusion_matrix,
    per_class_accuracy,
    per_group_accuracy,
    per_group_topk_accuracy,
    classification_report_dict,
)


# ═══════════════════════════════════════════════════════════
# 评估主函数
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(
    checkpoint_path: str,
    labels_csv: str,
    output_dir: str,
    resolution: str = None,
    scan_mode: str = None,
    batch_size: int = 64,
    device: str = "auto",
    num_workers: int = 4,
    target_size: int = 128,
    complexity_csv: str = None,
) -> dict:
    """
    加载模型并在 test set 上评估

    Args:
        checkpoint_path: .pth checkpoint 路径
        labels_csv:      labels.csv 路径
        output_dir:      输出目录
        resolution:      分辨率筛选
        scan_mode:       扫描模式筛选
        batch_size:      批大小
        device:          设备
        num_workers:     数据加载线程
        target_size:     图像 resize 尺寸
        complexity_csv:  复杂度 CSV

    Returns:
        评估指标 dict
    """
    # ── 设置 ──
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    os.makedirs(output_dir, exist_ok=True)
    print(f"设备: {device}")
    print(f"输出目录: {output_dir}")

    # ── 1. 加载 checkpoint ──
    print(f"\n加载 checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    num_classes = ckpt["num_classes"]
    model_name = ckpt.get("model_name", "light_cnn")
    id_to_label = ckpt.get("id_to_label", {})
    label_to_id = ckpt.get("label_to_id", {})
    train_resolution = ckpt.get("resolution", "")
    train_scan_mode = ckpt.get("scan_mode", "")

    print(f"  模型: {model_name}")
    print(f"  num_classes: {num_classes}")
    print(f"  训练时分辨率: {train_resolution}")
    print(f"  val_top1 (训练时): {ckpt.get('val_top1_accuracy', 'N/A')}")

    # ── 2. 加载模型 ──
    if model_name == "resnet18":
        model = create_model(model_name, num_classes=num_classes, in_channels=3)
    else:
        model = create_model(model_name, num_classes=num_classes, in_channels=1)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # ── 3. 加载 test 数据 ──
    print(f"\n加载 test 数据: {labels_csv}")
    test_dataset = SPVCharDataset(
        labels_csv=labels_csv,
        split="test",
        resolution=resolution,
        scan_mode=scan_mode,
        target_size=target_size,
        complexity_csv=complexity_csv,
        label_to_id=label_to_id,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"  test samples: {len(test_dataset)}")

    # ── 4. 推理 ──
    print("\n推理中...")
    all_outputs = []
    all_labels = []
    all_metas = []

    for batch in test_loader:
        images = batch["image"].to(device)
        labels = batch["label"]  # CPU

        outputs = model(images)

        all_outputs.append(outputs.cpu().numpy())
        all_labels.append(labels.numpy())

        # 收集元信息
        for i in range(len(labels)):
            all_metas.append({
                "image_path": batch["meta"]["image_path"][i] if isinstance(batch["meta"]["image_path"], list) else batch["meta"]["image_path"],
                "label": batch["meta"]["label"][i] if isinstance(batch["meta"]["label"], list) else batch["meta"]["label"],
                "resolution": batch["meta"]["resolution"][i] if isinstance(batch["meta"]["resolution"], list) else batch["meta"]["resolution"],
                "scan_mode": batch["meta"]["scan_mode"][i] if isinstance(batch["meta"]["scan_mode"], list) else batch["meta"]["scan_mode"],
                "complexity_group": batch["meta"]["complexity_group"][i] if isinstance(batch["meta"]["complexity_group"], list) else batch["meta"]["complexity_group"],
                "augment_id": batch["meta"]["augment_id"][i] if isinstance(batch["meta"]["augment_id"], list) else batch["meta"]["augment_id"],
            })

    all_outputs = np.concatenate(all_outputs, axis=0)  # (N, C)
    all_labels = np.concatenate(all_labels, axis=0)    # (N,)

    N = len(all_labels)
    print(f"  完成 {N} 个样本的推理")

    # ── 5. 计算 Top-K ──
    topk = topk_accuracy(all_outputs, all_labels, topk=(1, 5))
    preds = np.argmax(all_outputs, axis=1)             # Top-1 预测
    preds_detail = topk_predictions(all_outputs, all_labels, topk=(1, 5))
    probs = softmax(all_outputs)

    print(f"\n  Top-1 Accuracy: {topk[1]:.4f}")
    print(f"  Top-5 Accuracy: {topk[5]:.4f}")

    # ── 6. 混淆矩阵 ──
    cm_norm = compute_confusion_matrix(preds, all_labels, num_classes)
    cm_raw = compute_raw_confusion_matrix(preds, all_labels, num_classes)

    # ── 7. 每类准确率 ──
    per_class = per_class_accuracy(preds, all_labels, num_classes)
    class_report = classification_report_dict(preds, all_labels, num_classes, id_to_label)

    # ── 8. 分组准确率 ──
    complexity_groups = [m["complexity_group"] for m in all_metas]
    group_names = sorted(set(g for g in complexity_groups if g))

    per_group = {}
    per_group_top5 = {}
    if group_names:
        per_group = per_group_accuracy(preds, all_labels, complexity_groups, group_names)
        per_group_top5 = per_group_topk_accuracy(
            all_outputs, all_labels, complexity_groups, k=5, group_names=group_names
        )
        print(f"\n  分组准确率:")
        for g in group_names:
            print(f"    {g}: Top-1={per_group.get(g, 0):.4f}  Top-5={per_group_top5.get(g, 0):.4f}")

    # ── 9. 按分辨率分组 ──
    resolutions = [m["resolution"] for m in all_metas]
    res_groups = sorted(set(resolutions))
    per_resolution = {}
    if len(res_groups) > 1:
        per_resolution = per_group_accuracy(preds, all_labels, resolutions, res_groups)
        print(f"\n  按分辨率分组:")
        for r in res_groups:
            print(f"    {r}: Top-1={per_resolution.get(r, 0):.4f}")

    # ── 10. 保存 test_metrics.json ──
    metrics = {
        "num_samples": N,
        "num_classes": num_classes,
        "model_name": model_name,
        "resolution": resolution or train_resolution,
        "scan_mode": scan_mode or train_scan_mode,
        "top1_accuracy": topk[1],
        "top5_accuracy": topk[5],
        "per_group_accuracy": {
            g: {"top1": per_group.get(g, 0), "top5": per_group_top5.get(g, 0)}
            for g in group_names
        } if group_names else {},
        "per_resolution_accuracy": per_resolution if len(res_groups) > 1 else {},
    }

    metrics_path = os.path.join(output_dir, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n  test_metrics.json → {metrics_path}")

    # ── 11. 保存 test_predictions.csv ──
    preds_csv = os.path.join(output_dir, "test_predictions.csv")
    with open(preds_csv, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "image_path", "true_label", "pred_label",
            "top1_correct", "top5_correct",
            "top5_labels", "top5_probs",
            "resolution", "scan_mode", "complexity_group",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(N):
            top5_idx = preds_detail["topk_indices"][i]
            top5_label_strs = [id_to_label.get(int(idx), str(idx)) for idx in top5_idx[:5]]
            top5_prob_vals = preds_detail["topk_probs"][i][:5]

            writer.writerow({
                "image_path": all_metas[i]["image_path"],
                "true_label": all_metas[i]["label"],
                "pred_label": id_to_label.get(int(preds[i]), str(preds[i])),
                "top1_correct": bool(preds_detail["top1_correct"][i]),
                "top5_correct": bool(preds_detail["top5_correct"][i]),
                "top5_labels": "|".join(top5_label_strs),
                "top5_probs": "|".join(f"{p:.4f}" for p in top5_prob_vals),
                "resolution": all_metas[i]["resolution"],
                "scan_mode": all_metas[i]["scan_mode"],
                "complexity_group": all_metas[i]["complexity_group"],
            })
    print(f"  test_predictions.csv → {preds_csv}")

    # ── 12. 保存 confusion_matrix.csv ──
    cm_csv = os.path.join(output_dir, "confusion_matrix.csv")
    with open(cm_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        # 表头
        header = ["true_label"] + [id_to_label.get(i, str(i)) for i in range(num_classes)]
        writer.writerow(header)
        for i in range(num_classes):
            row = [id_to_label.get(i, str(i))] + [f"{cm_norm[i, j]:.4f}" for j in range(num_classes)]
            writer.writerow(row)
    print(f"  confusion_matrix.csv → {cm_csv}")

    # ── 13. 保存 per_class_accuracy.csv ──
    pca_csv = os.path.join(output_dir, "per_class_accuracy.csv")
    with open(pca_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class_id", "label", "n_samples", "n_correct", "accuracy"])
        writer.writeheader()
        for row in class_report:
            writer.writerow(row)
    print(f"  per_class_accuracy.csv → {pca_csv}")

    # ── 14. 保存 per_group_accuracy.csv ──
    if group_names:
        pga_csv = os.path.join(output_dir, "per_group_accuracy.csv")
        with open(pga_csv, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["group", "top1_accuracy", "top5_accuracy", "n_samples"])
            writer.writeheader()
            for g in group_names:
                n = sum(1 for m in all_metas if m["complexity_group"] == g)
                writer.writerow({
                    "group": g,
                    "top1_accuracy": round(per_group.get(g, 0), 6),
                    "top5_accuracy": round(per_group_top5.get(g, 0), 6),
                    "n_samples": n,
                })
        print(f"  per_group_accuracy.csv → {pga_csv}")

    print("\n评估完成!")
    return metrics


# ═══════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SPV 汉字识别模型评估"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="训练好的 .pth checkpoint 路径"
    )
    parser.add_argument(
        "--labels-csv",
        default="E:/dataset/augmented_spv/labels.csv",
        help="labels.csv 路径"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/evaluate_recognition",
        help="评估输出目录"
    )
    parser.add_argument(
        "--resolution",
        default=None,
        help="分辨率筛选 (None=全部)"
    )
    parser.add_argument(
        "--scan-mode",
        default=None,
        help="扫描模式筛选 (None=全部)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="批大小 (默认 64)"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="设备: auto/cuda/cpu"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="数据加载线程 (默认 4)"
    )
    parser.add_argument(
        "--complexity-csv",
        default="E:/dataset/complexity_scores.csv",
        help="复杂度 CSV (用于附加 complexity_group)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("SPV 汉字识别 — 模型评估")
    print("=" * 60)
    print(f"Checkpoint:   {args.checkpoint}")
    print(f"labels.csv:   {args.labels_csv}")
    print(f"输出目录:     {args.output_dir}")
    print(f"分辨率:       {args.resolution or '全部'}")
    print()

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


if __name__ == "__main__":
    main()
