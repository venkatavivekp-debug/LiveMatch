from __future__ import annotations

import numpy as np

from ml.calibration import apply_conformal_interval, fit_split_conformal_interval


def test_split_conformal_interval_fit_and_apply() -> None:
    actual = np.array([150.0, 162.0, 174.0, 185.0], dtype=np.float32)
    preds = np.array(
        [
            [148.0, 152.0, 156.0],
            [158.0, 161.0, 165.0],
            [166.0, 170.0, 173.0],
            [176.0, 180.0, 183.0],
        ],
        dtype=np.float32,
    )

    calibration = fit_split_conformal_interval(actual=actual, predictions=preds, alpha=0.1)
    assert calibration["enabled"] is True
    assert calibration["q_hat"] >= 0.0
    assert 0.0 <= calibration["raw_coverage"] <= 1.0
    assert 0.0 <= calibration["calibrated_coverage"] <= 1.0

    lower, upper = apply_conformal_interval(preds, calibration)
    assert lower.shape == (4,)
    assert upper.shape == (4,)
    assert np.all(upper >= lower)


def test_apply_without_calibration_returns_head_range() -> None:
    preds = np.array([[1.0, 2.5, 3.0], [0.2, 0.7, 1.4]], dtype=np.float32)
    lower, upper = apply_conformal_interval(preds, calibration=None)
    assert np.allclose(lower, preds.min(axis=1))
    assert np.allclose(upper, preds.max(axis=1))
