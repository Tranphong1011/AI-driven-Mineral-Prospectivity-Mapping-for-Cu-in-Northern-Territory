from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score

def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_score))

def recall_at_k_percent(y_true: np.ndarray, y_score: np.ndarray, k_percent: float) -> float:
    n = len(y_true)
    k = max(1, int(round(n * (k_percent / 100.0))))
    idx = np.argsort(-y_score)[:k]
    denom = max(1, int(np.sum(y_true == 1)))
    return float(np.sum(y_true[idx] == 1) / denom)
