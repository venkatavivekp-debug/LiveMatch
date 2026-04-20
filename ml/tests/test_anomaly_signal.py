from __future__ import annotations

from ml.inference import LiveMatchPredictor


def test_anomaly_signal_flags_odd_variant_for_large_deviation() -> None:
    predictor = LiveMatchPredictor.__new__(LiveMatchPredictor)
    predictor.feature_medians = {
        "team_a_avg_runs_last_5": 170.0,
        "venue_avg_score": 168.0,
        "team_a_run_rate_trend": 0.0,
        "team_a_win_rate_vs_b": 0.5,
    }

    anomaly = LiveMatchPredictor._compute_anomaly_signal(
        predictor,
        feature_dict={
            "team_a_avg_runs_last_5": 212.0,
            "venue_avg_score": 205.0,
            "team_a_run_rate_trend": 16.2,
            "team_a_win_rate_vs_b": 0.82,
        },
        residual_context={"residual_shift_score": 9.0},
    )

    assert 0.0 <= anomaly["anomaly_score"] <= 1.0
    assert anomaly["odd_variant_flag"] is True
    assert "signal_breakdown" in anomaly
    assert anomaly["signal_breakdown"]["feature_deviation_score"] > 0
