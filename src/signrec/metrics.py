"""Classification metrics for isolated sign recognition."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score


def topk_accuracy(probs: np.ndarray, y: np.ndarray, k: int) -> float:
    topk = np.argpartition(-probs, kth=min(k, probs.shape[1]) - 1, axis=1)[:, :k]
    return float((topk == y[:, None]).any(axis=1).mean())


def macro_f1(y: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    return float(f1_score(y, pred, average="macro", labels=np.arange(n_classes), zero_division=0))


def per_group_accuracy(y: np.ndarray, pred: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"g": groups, "ok": y == pred})
    out = df.groupby("g")["ok"].agg(["mean", "size"]).rename(columns={"mean": "acc", "size": "n"})
    return out.sort_values("acc")


def per_class_accuracy(y: np.ndarray, pred: np.ndarray, n_classes: int) -> np.ndarray:
    acc = np.full(n_classes, np.nan)
    for c in range(n_classes):
        m = y == c
        if m.any():
            acc[c] = (pred[m] == c).mean()
    return acc


