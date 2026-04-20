from __future__ import annotations

from typing import Any

import numpy as np


def _quantile_higher(values: np.ndarray, quantile: float) -> float:
    clipped = float(np.clip(quantile, 0.0, 1.0))
    try:
        return float(np.quantile(values, clipped, method="higher"))
    except TypeError:
        # numpy<1.22 compatibility branch
        return float(np.quantile(values, clipped, interpolation="higher"))


def fit_split_conformal_interval(
    actual: np.ndarray,
    predictions: np.ndarray,
    alpha: float = 0.1,
) -> dict[str, Any]:
    """
    Fit split-conformal calibration over scenario intervals.

    Nonconformity score for sample i:
      s_i = max(lower_i - y_i, y_i - upper_i, 0)
    where lower_i/min and upper_i/max come from K hypotheses.
    """
    if predictions.ndim != 2:
        raise ValueError(f"predictions must have shape [N, K], got {predictions.shape}")
    if actual.ndim != 1:
        raise ValueError(f"actual must have shape [N], got {actual.shape}")
    if len(actual) != predictions.shape[0]:
        raise ValueError(
            f"actual/prediction length mismatch: {len(actual)} vs {predictions.shape[0]}"
        )

    alpha = float(np.clip(alpha, 1e-4, 0.499))
    lower = np.min(predictions, axis=1)
    upper = np.max(predictions, axis=1)
    nonconformity = np.maximum(np.maximum(lower - actual, actual - upper), 0.0)
    n = len(nonconformity)

    if n == 0:
        return {
            "enabled": False,
            "method": "split_conformal_interval",
            "alpha": alpha,
            "q_hat": 0.0,
            "sample_size": 0,
            "raw_coverage": 0.0,
            "calibrated_coverage": 0.0,
            "raw_interval_width": 0.0,
            "calibrated_interval_width": 0.0,
        }

    quantile_level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    q_hat = _quantile_higher(nonconformity, quantile_level)

    calibrated_lower = lower - q_hat
    calibrated_upper = upper + q_hat

    raw_coverage = float(np.mean((actual >= lower) & (actual <= upper)))
    calibrated_coverage = float(np.mean((actual >= calibrated_lower) & (actual <= calibrated_upper)))
    raw_width = float(np.mean(upper - lower))
    calibrated_width = float(np.mean(calibrated_upper - calibrated_lower))

    return {
        "enabled": True,
        "method": "split_conformal_interval",
        "alpha": alpha,
        "q_hat": float(q_hat),
        "quantile_level": float(quantile_level),
        "sample_size": int(n),
        "raw_coverage": raw_coverage,
        "calibrated_coverage": calibrated_coverage,
        "raw_interval_width": raw_width,
        "calibrated_interval_width": calibrated_width,
    }


def apply_conformal_interval(
    predictions: np.ndarray,
    calibration: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if predictions.ndim != 2:
        raise ValueError(f"predictions must have shape [N, K], got {predictions.shape}")

    lower = np.min(predictions, axis=1)
    upper = np.max(predictions, axis=1)

    if not calibration:
        return lower, upper

    q_hat = float(calibration.get("q_hat", 0.0))
    if q_hat <= 0:
        return lower, upper
    return lower - q_hat, upper + q_hat


def evaluate_conformal_interval(
    actual: np.ndarray,
    predictions: np.ndarray,
    calibration: dict[str, Any] | None,
) -> dict[str, float]:
    lower, upper = apply_conformal_interval(predictions, calibration)
    return {
        "coverage": float(np.mean((actual >= lower) & (actual <= upper))),
        "interval_width": float(np.mean(upper - lower)),
    }
