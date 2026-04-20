from __future__ import annotations

import numpy as np

from ml.evaluate import (
    best_match_error,
    best_match_error_multi,
    coverage,
    coverage_multi,
    diversity_score,
    diversity_score_multi,
    interval_width,
    interval_width_multi,
    mae,
    rmse,
    winner_accuracy,
)


def test_evaluate_metrics_basic_shapes() -> None:
    actual = np.array([150.0, 170.0, 190.0], dtype=np.float32)
    predictions = np.array(
        [
            [140.0, 150.0, 160.0],
            [160.0, 170.0, 180.0],
            [175.0, 190.0, 205.0],
        ],
        dtype=np.float32,
    )

    center = predictions.mean(axis=1)

    assert mae(actual, center) >= 0.0
    assert rmse(actual, center) >= 0.0
    assert 0.0 <= coverage(actual, predictions) <= 1.0
    assert best_match_error(actual, predictions) >= 0.0
    assert diversity_score(predictions) > 0.0
    assert interval_width(predictions) > 0.0


def test_multi_output_metrics_and_winner_accuracy() -> None:
    actual_scores = np.array([[165.0, 160.0], [178.0, 174.0], [151.0, 155.0]], dtype=np.float32)
    predicted_scores = np.array(
        [
            [[160.0, 157.0], [166.0, 161.0], [172.0, 168.0]],
            [[171.0, 173.0], [179.0, 175.0], [186.0, 182.0]],
            [[146.0, 150.0], [152.0, 156.0], [159.0, 161.0]],
        ],
        dtype=np.float32,
    )
    winner_probs = np.array(
        [
            [0.52, 0.57, 0.63],
            [0.54, 0.61, 0.68],
            [0.42, 0.46, 0.49],
        ],
        dtype=np.float32,
    )
    actual_winner = np.array([1.0, 1.0, 0.0], dtype=np.float32)

    assert best_match_error_multi(actual_scores, predicted_scores) >= 0.0
    assert 0.0 <= coverage_multi(actual_scores, predicted_scores) <= 1.0
    assert diversity_score_multi(predicted_scores, winner_probs) > 0.0
    assert interval_width_multi(predicted_scores) > 0.0
    assert 0.0 <= winner_accuracy(actual_winner, winner_probs.mean(axis=1)) <= 1.0


def test_best_match_error_multi_uses_both_team_scores() -> None:
    actual_scores = np.array([[170.0, 150.0]], dtype=np.float32)
    predicted_scores = np.array(
        [
            [
                [170.0, 190.0],  # perfect team_a, poor team_b
                [181.0, 150.0],  # poor team_a, perfect team_b
                [172.0, 151.0],  # closest on both
            ]
        ],
        dtype=np.float32,
    )
    metric = best_match_error_multi(actual_scores, predicted_scores)
    expected = float(np.sqrt(((172.0 - 170.0) ** 2 + (151.0 - 150.0) ** 2) / 2.0))
    assert metric == np.float32(expected)
