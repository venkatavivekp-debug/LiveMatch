from __future__ import annotations

from ml.live_features import blend_cricket_features


def test_blend_cricket_features_hybrid_mode() -> None:
    historical = {
        "batting_team_avg_runs_last5": 168.0,
        "bowling_team_avg_runs_conceded_last5": 171.0,
    }
    live = {
        "batting_team_avg_runs_last5": 190.0,
        "bowling_team_avg_runs_conceded_last5": 182.0,
    }
    blended, meta = blend_cricket_features(historical, live, recency_weight=0.5)
    assert blended["batting_team_avg_runs_last5"] == 179.0
    assert blended["bowling_team_avg_runs_conceded_last5"] == 176.5
    assert meta["data_mode"] == "HYBRID"
    assert meta["live_feature_count"] == 2


def test_blend_cricket_features_historical_mode_without_live() -> None:
    historical = {"batting_team_avg_runs_last5": 168.0}
    blended, meta = blend_cricket_features(historical, None, recency_weight=0.65)
    assert blended["batting_team_avg_runs_last5"] == 168.0
    assert meta["data_mode"] == "HISTORICAL"
    assert meta["live_feature_count"] == 0
