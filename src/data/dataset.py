"""
数据集模块: SPV 汉字识别 PyTorch Dataset 和 DataLoader 创建

用法:
  from dataset import SPVCharDataset, create_dataloaders

  train_loader, val_loader, test_loader = create_dataloaders(
      labels_csv="E:/dataset/char_spv_augmented/labels.csv",
      resolution="6x6",
      batch_size=64,
  )
"""

import os
import csv
from collections import defaultdict

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader


# ═══════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════

class SPVCharDataset(Dataset):
    """
    SPV 汉字图像数据集

    从 labels.csv 加载样本, 支持按 split / resolution / scan_mode 筛选

    labels.csv 必需字段:
      image_path, label, split, resolution

    labels.csv 可选字段:
      augment_id, scan_mode, complexity_group, dx, dy, ...

    输出:
      image:  (1, 128, 128) float tensor, 归一化到 [0, 1]
      label:  int, 类别 ID
      meta:   dict, 原始元信息
    """

    def __init__(
        self,
        labels_csv: str,
        split: str = "train",               # "train" | "val" | "test" | "all"
        resolution: str = None,             # None=所有, "6x6"|"8x8"|"10x10"|"12x12"
        scan_mode: str = None,              # None=所有, "left-right"|...
        transform: callable = None,         # 额外 transform (不被使用时为 None)
        target_size: int = 128,             # resize 目标尺寸
        complexity_csv: str = None,         # 可选: 复杂度 CSV, 用于附加 complexity_group
        label_to_id: dict = None,           # 外部预定义 label→id 映射
    ):
        super().__init__()

        self.split = split
        self.resolution = resolution
        self.scan_mode = scan_mode
        self.target_size = target_size
        self.transform = transform

        # ── 1. 读取 labels.csv ──
        self.samples = []
        with open(labels_csv, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 筛选 split (非 "all" 时)
                if split != "all" and row.get("split", "") != split:
                    continue
                # 筛选 resolution
                if resolution is not None and row.get("resolution", "") != resolution:
                    continue
                # 筛选 scan_mode (可选)
                if scan_mode is not None and row.get("scan_mode", "") != scan_mode:
                    continue

                self.samples.append(dict(row))

        if len(self.samples) == 0:
            raise ValueError(
                f"未找到匹配的样本 (split={split}, resolution={resolution}, "
                f"scan_mode={scan_mode})"
            )

        # ── 2. 构建 label → id 映射 ──
        if label_to_id is not None:
            self.label_to_id = label_to_id
        else:
            unique_labels = sorted(set(s["label"] for s in self.samples))
            self.label_to_id = {label: i for i, label in enumerate(unique_labels)}

        self.id_to_label = {i: label for label, i in self.label_to_id.items()}
        self.num_classes = len(self.label_to_id)

        # ── 3. 可选: 加载复杂度分组 ──
        self.char_to_group = {}
        if complexity_csv is not None and os.path.exists(complexity_csv):
            with open(complexity_csv, encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    self.char_to_group[row["汉字"]] = row.get("等级", "")

        # ── 4. 添加 complexity_group 到每个样本 ──
        for s in self.samples:
            if "complexity_group" not in s:
                s["complexity_group"] = self.char_to_group.get(s["label"], "")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # ── 加载图像 ──
        img_path = sample["image_path"]
        if not os.path.isabs(img_path):
            # 相对于 labels.csv 所在目录
            img_path = os.path.join(os.path.dirname(self._labels_csv), img_path)

        image = Image.open(img_path).convert("L")                   # 灰度
        image = image.resize((self.target_size, self.target_size), Image.LANCZOS)
        img_array = np.array(image, dtype=np.float32) / 255.0       # [0, 1]
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)        # (1, H, W)

        if self.transform:
            img_tensor = self.transform(img_tensor)

        # ── 标签 ──
        label_str = sample["label"]
        label_id = self.label_to_id[label_str]

        # ── 元信息 ──
        meta = {
            "image_path": sample["image_path"],
            "label": label_str,
            "label_id": label_id,
            "resolution": sample.get("resolution", ""),
            "scan_mode": sample.get("scan_mode", ""),
            "augment_id": sample.get("augment_id", ""),
            "complexity_group": sample.get("complexity_group", ""),
        }

        return {
            "image": img_tensor,
            "label": label_id,
            "meta": meta,
        }


# ═══════════════════════════════════════════════════════════
# DataLoader 创建
# ═══════════════════════════════════════════════════════════

def create_dataloaders(
    labels_csv: str,
    resolution: str = None,
    scan_mode: str = None,
    batch_size: int = 64,
    num_workers: int = 4,
    target_size: int = 128,
    complexity_csv: str = None,
    label_to_id: dict = None,
) -> dict[str, DataLoader]:
    """
    创建 train / val / test DataLoader

    Args:
        labels_csv:    labels.csv 路径
        resolution:    分辨率筛选 (None=全部)
        scan_mode:     扫描模式筛选 (None=全部)
        batch_size:    批大小
        num_workers:   数据加载线程数
        target_size:   图像 resize 尺寸
        complexity_csv: 复杂度 CSV (附加 complexity_group)
        label_to_id:   外部 label→id 映射 (None=从数据集自动构建)

    Returns:
        {
            "train": DataLoader,
            "val":   DataLoader,
            "test":  DataLoader,
            "label_to_id": dict,
            "id_to_label": dict,
            "num_classes": int,
        }
    """
    # 先用 train 构建 label→id 映射
    train_dataset = SPVCharDataset(
        labels_csv=labels_csv,
        split="train",
        resolution=resolution,
        scan_mode=scan_mode,
        target_size=target_size,
        complexity_csv=complexity_csv,
        label_to_id=label_to_id,
    )

    shared_label_to_id = train_dataset.label_to_id

    val_dataset = SPVCharDataset(
        labels_csv=labels_csv,
        split="val",
        resolution=resolution,
        scan_mode=scan_mode,
        target_size=target_size,
        complexity_csv=complexity_csv,
        label_to_id=shared_label_to_id,
    )

    test_dataset = SPVCharDataset(
        labels_csv=labels_csv,
        split="test",
        resolution=resolution,
        scan_mode=scan_mode,
        target_size=target_size,
        complexity_csv=complexity_csv,
        label_to_id=shared_label_to_id,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"数据集统计 (resolution={resolution or 'all'}, scan_mode={scan_mode or 'all'}):")
    print(f"  train: {len(train_dataset)} samples")
    print(f"  val:   {len(val_dataset)} samples")
    print(f"  test:  {len(test_dataset)} samples")
    print(f"  num_classes: {train_dataset.num_classes}")

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "label_to_id": shared_label_to_id,
        "id_to_label": train_dataset.id_to_label,
        "num_classes": train_dataset.num_classes,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
    }


# ═══════════════════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("测试 SPVCharDataset...")

    # 尝试加载数据
    possible_csvs = [
        "E:/dataset/char_spv_augmented/labels.csv",
    ]

    found = False
    for csv_path in possible_csvs:
        if os.path.exists(csv_path):
            print(f"  找到 labels.csv: {csv_path}")
            try:
                ds = SPVCharDataset(csv_path, split="train")
                print(f"  train samples: {len(ds)}")
                print(f"  num_classes: {ds.num_classes}")

                # 取一个样本检查
                sample = ds[0]
                print(f"  sample[0] image shape: {sample['image'].shape}")
                print(f"  sample[0] label: {sample['label']}")
                print(f"  sample[0] meta keys: {list(sample['meta'].keys())}")
                found = True
            except Exception as e:
                print(f"  加载失败: {e}")
            break

    if not found:
        print("  未找到 labels.csv, 跳过加载测试")
        print("  请先运行 data_augmentation.py 生成数据")
