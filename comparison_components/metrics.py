from typing import Dict, List, Tuple

import numpy as np
import scipy.stats


def compute_auc(pos_scores: List[float], neg_scores: List[float]) -> float:
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float("nan")
    all_scores = np.array(pos_scores + neg_scores, dtype=float)
    ranks = scipy.stats.rankdata(all_scores, method="average")
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    sum_ranks_pos = float(np.sum(ranks[:n_pos]))
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def compute_accuracy(pos_detections: List[Dict], neg_detections: List[Dict]) -> float:
    if len(pos_detections) == 0 or len(neg_detections) == 0:
        return float("nan")
    tp = sum(bool(d.get("detected", False)) for d in pos_detections)
    tn = sum(not bool(d.get("detected", False)) for d in neg_detections)
    total = len(pos_detections) + len(neg_detections)
    return float((tp + tn) / total)


def bootstrap_ci(values: List[float], n_boot: int = 1000, alpha: float = 0.05) -> Tuple[float, float, float]:
    vals = np.array(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    boot_means = []
    for _ in range(n_boot):
        sample = np.random.choice(vals, size=len(vals), replace=True)
        boot_means.append(float(np.mean(sample)))
    lower = float(np.quantile(boot_means, alpha / 2.0))
    upper = float(np.quantile(boot_means, 1.0 - alpha / 2.0))
    mean = float(np.mean(vals))
    return mean, lower, upper
