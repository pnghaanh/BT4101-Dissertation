from typing import Dict, List

import numpy as np


def compute_calibration_stats(detections: List[Dict]) -> Dict[str, float]:
    """Additional test: calibration of detector confidence.

    Uses p-value when available and reports Brier-like score on detected labels.
    """
    if not detections:
        return {"mean_p_value": float("nan"), "brier_proxy": float("nan")}

    pvals = [float(d.get("p_value", 1.0)) for d in detections]
    preds = [1.0 if bool(d.get("detected", False)) else 0.0 for d in detections]
    conf = [1.0 - p for p in pvals]
    brier = float(np.mean([(c - y) ** 2 for c, y in zip(conf, preds)]))
    return {"mean_p_value": float(np.mean(pvals)), "brier_proxy": brier}


def compute_efficiency_stats(latencies: List[float], n_tokens: List[int]) -> Dict[str, float]:
    """Additional test: throughput and latency statistics."""
    if not latencies:
        return {
            "mean_latency_sec": float("nan"),
            "p95_latency_sec": float("nan"),
            "mean_tokens_per_sec": float("nan"),
        }

    tps = []
    for t, tok in zip(latencies, n_tokens):
        if t > 0:
            tps.append(tok / t)

    return {
        "mean_latency_sec": float(np.mean(latencies)),
        "p95_latency_sec": float(np.quantile(latencies, 0.95)),
        "mean_tokens_per_sec": float(np.mean(tps)) if tps else float("nan"),
    }
