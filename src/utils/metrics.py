"""
工具函数: 评估指标 (Metrics Utilities)

包含:
  - topk_accuracy:     Top-K 准确率
  - confusion_matrix:  混淆矩阵
  - per_class_accuracy: 每类准确率
  - per_group_accuracy: 分组准确率 (如低/中/高复杂度)
"""

import numpy as np
from collections import defaultdict


# ═══════════════════════════════════════════════════════════
# Top-K Accuracy
# ═══════════════════════════════════════════════════════════

def topk_accuracy(
    outputs: np.ndarray,          # (N, num_classes)  logits 或 probs
    targets: np.ndarray,          # (N,)            int 标签
    topk: tuple = (1, 5),
) -> dict[int, float]:
    """
    计算 Top-K 准确率

    Args:
        outputs: 模型输出 (logits 或 probabilities), shape (N, C)
        targets: 真实标签, shape (N,)
        topk:    要计算的 K 值元组, 如 (1, 5)

    Returns:
        {k: accuracy}  例: {1: 0.85, 5: 0.95}
    """
    max_k = max(topk)
    N = targets.shape[0]

    # 取 top-k 预测索引
    pred_topk = np.argsort(outputs, axis=1)[:, ::-1][:, :max_k]  # (N, max_k)

    # 正确标记矩阵
    correct = (pred_topk == targets.reshape(-1, 1))  # (N, max_k), bool

    result = {}
    for k in topk:
        # 对每个样本, 检查 top-k 中是否至少有一个正确
        topk_correct = correct[:, :k].any(axis=1).sum()
        result[k] = round(float(topk_correct) / N, 6) if N > 0 else 0.0

    return result


def topk_predictions(
    outputs: np.ndarray,          # (N, num_classes)
    targets: np.ndarray,          # (N,)
    topk: tuple = (1, 5),
) -> dict:
    """
    返回每个样本的 Top-K 预测详情

    Returns:
        {
            "topk_indices":  (N, max_k)  索引
            "topk_labels":   list[str]   标签名 (需外部映射)
            "topk_probs":    (N, max_k)  概率
            "top1_correct":  (N,) bool
            "top5_correct":  (N,) bool
        }
    """
    max_k = max(topk)
    N = targets.shape[0]

    # softmax 转概率
    probs = softmax(outputs)
    sorted_idx = np.argsort(probs, axis=1)[:, ::-1][:, :max_k]

    # 取 top-k 概率
    topk_probs = np.take_along_axis(probs, sorted_idx, axis=1)

    # 正确性
    correct = (sorted_idx == targets.reshape(-1, 1))

    return {
        "topk_indices": sorted_idx,
        "topk_probs": topk_probs,
        "top1_correct": correct[:, 0],
        "top5_correct": correct[:, :5].any(axis=1) if max_k >= 5 else correct.any(axis=1),
    }


def softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    """数值稳定的 softmax"""
    x_max = x.max(axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / e_x.sum(axis=axis, keepdims=True)


# ═══════════════════════════════════════════════════════════
# Confusion Matrix
# ═══════════════════════════════════════════════════════════

def compute_confusion_matrix(
    preds: np.ndarray,            # (N,) int
    targets: np.ndarray,          # (N,) int
    num_classes: int,
) -> np.ndarray:
    """
    计算混淆矩阵 (归一化行)

    Args:
        preds:   Top-1 预测标签
        targets: 真实标签
        num_classes: 类别数

    Returns:
        (num_classes, num_classes) 行归一化混淆矩阵 (每行和为 1)
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.float64)

    for t, p in zip(targets, preds):
        cm[t, p] += 1

    # 行归一化 (per-class recall)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # 避免除零
    cm_norm = cm / row_sums

    return cm_norm


def compute_raw_confusion_matrix(
    preds: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """原始计数混淆矩阵 (未归一化)"""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets, preds):
        cm[t, p] += 1
    return cm


# ═══════════════════════════════════════════════════════════
# Per-Class Accuracy
# ═══════════════════════════════════════════════════════════

def per_class_accuracy(
    preds: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
) -> dict[int, float]:
    """
    计算每个类别的准确率

    Returns:
        {class_id: accuracy}
    """
    acc = {}
    for c in range(num_classes):
        mask = (targets == c)
        if mask.sum() > 0:
            acc[c] = float((preds[mask] == c).sum()) / mask.sum()
        else:
            acc[c] = 0.0
    return acc


# ═══════════════════════════════════════════════════════════
# Per-Group Accuracy
# ═══════════════════════════════════════════════════════════

def per_group_accuracy(
    preds: np.ndarray,           # (N,)
    targets: np.ndarray,         # (N,)
    sample_groups: list[str],    # (N,)  每个样本的分组标签
    group_names: list[str] = None,
) -> dict[str, float]:
    """
    按分组计算准确率 (如低/中/高复杂度)

    Args:
        preds:         预测标签
        targets:       真实标签
        sample_groups: 每个样本的分组 (如 "low"/"medium"/"high")
        group_names:   要报告的分组列表 (None=全部)

    Returns:
        {group_name: accuracy}
    """
    if group_names is None:
        group_names = sorted(set(sample_groups))

    result = {}
    for g in group_names:
        mask = [sg == g for sg in sample_groups]
        mask = np.array(mask)
        if mask.sum() > 0:
            result[g] = float((preds[mask] == targets[mask]).sum()) / mask.sum()
        else:
            result[g] = 0.0

    return result


def per_group_topk_accuracy(
    outputs: np.ndarray,         # (N, C)
    targets: np.ndarray,         # (N,)
    sample_groups: list[str],
    k: int = 5,
    group_names: list[str] = None,
) -> dict[str, float]:
    """按分组计算 Top-K 准确率"""
    if group_names is None:
        group_names = sorted(set(sample_groups))

    pred_topk = np.argsort(outputs, axis=1)[:, ::-1][:, :k]
    correct = (pred_topk == targets.reshape(-1, 1)).any(axis=1)

    result = {}
    for g in group_names:
        mask = np.array([sg == g for sg in sample_groups])
        if mask.sum() > 0:
            result[g] = float(correct[mask].sum()) / mask.sum()
        else:
            result[g] = 0.0

    return result


# ═══════════════════════════════════════════════════════════
# 辅助: 分类报告
# ═══════════════════════════════════════════════════════════

def classification_report_dict(
    preds: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
    id_to_label: dict = None,
) -> list[dict]:
    """
    生成逐类分类报告

    Returns:
        [{class_id, label, n_samples, accuracy, ...}, ...]
    """
    cm_raw = compute_raw_confusion_matrix(preds, targets, num_classes)
    per_class = per_class_accuracy(preds, targets, num_classes)

    report = []
    for c in range(num_classes):
        n_samples = int(cm_raw[c].sum())
        n_correct = int(cm_raw[c, c])
        label_name = id_to_label.get(c, str(c)) if id_to_label else str(c)
        report.append({
            "class_id": c,
            "label": label_name,
            "n_samples": n_samples,
            "n_correct": n_correct,
            "accuracy": round(per_class[c], 6),
        })

    return report
