"""Metrics without sklearn: AUROC, balanced accuracy, confusion matrix, threshold-based detection."""
from __future__ import annotations

import numpy as np


def auroc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(order.size)
    ranks[order] = np.arange(1, order.size + 1)
    # average ranks for ties
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def confusion(pred, true, n):
    m = np.zeros((n, n), int)
    for p, t in zip(pred, true):
        m[t, p] += 1
    return m


def balanced_accuracy(pred, true, n):
    cm = confusion(pred, true, n)
    rec = [cm[i, i] / cm[i].sum() for i in range(n) if cm[i].sum() > 0]
    return float(np.mean(rec)) if rec else float("nan")


def detection_report(scores, anomaly, labels, class_names, threshold):
    """Detection rate per class at a fixed anomaly-score threshold."""
    scores, anomaly, labels = map(np.asarray, (scores, anomaly, labels))
    out = {"threshold": float(threshold), "false_positive_rate": float((scores[anomaly == 0] > threshold).mean())
           if (anomaly == 0).any() else float("nan")}
    for i, c in enumerate(class_names):
        m = labels == i
        if m.any():
            out[f"flagged_{c}"] = float((scores[m] > threshold).mean())
    return out
