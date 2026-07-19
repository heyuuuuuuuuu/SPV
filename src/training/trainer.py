"""
训练脚本: SPV 汉字识别模型训练

用法:
  from src.training import train
  train(labels_csv="labels.csv", output_dir="outputs/", model_name="light_cnn")
"""

import os
import csv
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models import create_model
from src.data import create_dataloaders
from src.utils import topk_accuracy


# ═══════════════════════════════════════════════════════════
# 训练一个 epoch
# ═══════════════════════════════════════════════════════════

def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> dict:
    """训练一个 epoch, 返回平均 loss"""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return {
        "loss": round(total_loss / max(n_batches, 1), 6),
    }


# ═══════════════════════════════════════════════════════════
# 验证一个 epoch
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """验证: 计算 loss, Top-1, Top-5 accuracy"""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_outputs = []
    all_labels = []

    for batch in dataloader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        n_batches += 1

        all_outputs.append(outputs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_outputs = np.concatenate(all_outputs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    acc = topk_accuracy(all_outputs, all_labels, topk=(1, 5))

    return {
        "loss": round(total_loss / max(n_batches, 1), 6),
        "top1_accuracy": acc.get(1, 0.0),
        "top5_accuracy": acc.get(5, 0.0),
    }


# ═══════════════════════════════════════════════════════════
# 主训练循环
# ═══════════════════════════════════════════════════════════

def train(
    labels_csv: str,
    output_dir: str,
    model_name: str = "light_cnn",
    resolution: str = None,
    scan_mode: str = None,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 0.001,
    weight_decay: float = 0.01,
    seed: int = 42,
    device: str = "auto",
    num_workers: int = 4,
    target_size: int = 128,
    complexity_csv: str = None,
    label_to_id: dict = None,
):
    """
    完整训练流程

    Args:
        labels_csv:    labels.csv 路径
        output_dir:    输出目录 (保存 checkpoint, log)
        model_name:    "light_cnn" | "resnet18"
        resolution:    分辨率筛选 (None=全部)
        scan_mode:     扫描模式筛选 (None=全部)
        epochs:        训练轮数
        batch_size:    批大小
        lr:            初始学习率
        weight_decay:  AdamW 权重衰减
        seed:          随机种子
        device:        "auto"|"cuda"|"cpu"
        num_workers:   数据加载线程
        target_size:   图像 resize 尺寸
        complexity_csv: 复杂度 CSV
        label_to_id:   外部 label→id 映射

    Returns:
        best_model_path, training_log
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    os.makedirs(output_dir, exist_ok=True)
    print(f"使用设备: {device}")
    print(f"输出目录: {output_dir}")

    loaders = create_dataloaders(
        labels_csv=labels_csv,
        resolution=resolution,
        scan_mode=scan_mode,
        batch_size=batch_size,
        num_workers=num_workers,
        target_size=target_size,
        complexity_csv=complexity_csv,
        label_to_id=label_to_id,
        seed=seed,
    )

    num_classes = loaders["num_classes"]
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    if model_name == "resnet18":
        model = create_model(model_name, num_classes=num_classes, in_channels=3)
    else:
        model = create_model(model_name, num_classes=num_classes, in_channels=1)
    model = model.to(device)
    print(f"模型: {model_name}  |  参数: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    log_path = os.path.join(output_dir, "training_log.csv")
    log_fields = [
        "epoch", "train_loss", "val_loss",
        "val_top1_accuracy", "val_top5_accuracy",
        "learning_rate", "time_sec",
    ]

    with open(log_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()

    best_val_top1 = 0.0
    best_model_path = os.path.join(output_dir, "best_model.pth")

    print("\n开始训练...")
    print(f"{'Epoch':>6s}  {'Train Loss':>10s}  {'Val Loss':>10s}  "
          f"{'Val Top1':>10s}  {'Val Top5':>10s}  {'LR':>10s}  {'Time':>8s}")
    print("-" * 72)

    for epoch in range(1, epochs + 1):
        t_start = time.time()

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t_start

        print(f"{epoch:>6d}  {train_metrics['loss']:>10.4f}  {val_metrics['loss']:>10.4f}  "
              f"{val_metrics['top1_accuracy']:>10.4f}  {val_metrics['top5_accuracy']:>10.4f}  "
              f"{current_lr:>10.6f}  {elapsed:>7.1f}s")

        log_entry = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_top1_accuracy": val_metrics["top1_accuracy"],
            "val_top5_accuracy": val_metrics["top5_accuracy"],
            "learning_rate": round(current_lr, 8),
            "time_sec": round(elapsed, 1),
        }
        with open(log_path, "a", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            writer.writerow(log_entry)

        if val_metrics["top1_accuracy"] > best_val_top1:
            best_val_top1 = val_metrics["top1_accuracy"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_top1_accuracy": best_val_top1,
                "val_top5_accuracy": val_metrics["top5_accuracy"],
                "model_name": model_name,
                "num_classes": num_classes,
                "label_to_id": loaders["label_to_id"],
                "id_to_label": loaders["id_to_label"],
                "resolution": resolution,
                "scan_mode": scan_mode,
                "seed": seed,
            }, best_model_path)
            print(f"  → 保存最佳模型 (val_top1={best_val_top1:.4f})")

    print(f"\n训练完成! 最佳 val Top-1: {best_val_top1:.4f}")
    print(f"  训练日志: {log_path}")
    print(f"  最佳模型: {best_model_path}")

    return best_model_path, log_path
