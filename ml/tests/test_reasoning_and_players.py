from __future__ import annotations

import re

import pandas as pd

from ml.inference import LiveMatchPredictor


def _predictor_stub() -> LiveMatchPredictor:
    predictor = LiveMatchPredictor.__new__(LiveMatchPredictor)
    predictor.baseline_score = 170.0
    predictor.feature_medians = {
        "team_a_avg_wickets_last_5": 7.0,
        "venue_chase_success_rate": 0.5,
        "recent_form_diff": 0.0,
        "head_to_head_win_diff": 0.0,
    }
    predictor.player_form = pd.DataFrame(
        [
            {
                "player": "Rohit Sharma",
                "team": "Mumbai Indians",
                "recent_runs": 54,
                "batting_form": 65,
                "strike_rate": 151,
                "win_rate": 0.62,
                "recent_wickets": 0.2,
                "avg_wickets": 0.1,
                "bowling_form": 8,
                "economy": 9.5,
                "impact_score": 78,
            },
            {
                "player": "Suryakumar Yadav",
                "team": "Mumbai Indians",
                "recent_runs": 47,
                "batting_form": 58,
                "strike_rate": 162,
                "win_rate": 0.57,
                "recent_wickets": 0.1,
                "avg_wickets": 0.1,
                "bowling_form": 6,
                "economy": 10.2,
                "impact_score": 71,
            },
            {
                "player": "Jasprit Bumrah",
                "team": "Mumbai Indians",
                "recent_runs": 8,
                "batting_form": 14,
                "strike_rate": 108,
                "win_rate": 0.63,
                "recent_wickets": 2.1,
                "avg_wickets": 1.6,
                "bowling_form": 74,
                "economy": 6.4,
                "impact_score": 76,
            },
            {
                "player": "Ruturaj Gaikwad",
                "team": "Chennai Super Kings",
                "recent_runs": 44,
                "batting_form": 57,
                "strike_rate": 145,
                "win_rate": 0.58,
                "recent_wickets": 0.0,
                "avg_wickets": 0.0,
                "bowling_form": 4,
                "economy": 10.8,
                "impact_score": 68,
            },
            {
                "player": "Ravindra Jadeja",
                "team": "Chennai Super Kings",
                "recent_runs": 28,
                "batting_form": 43,
                "strike_rate": 132,
                "win_rate": 0.56,
                "recent_wickets": 1.5,
                "avg_wickets": 1.2,
                "bowling_form": 61,
                "economy": 7.4,
                "impact_score": 73,
            },
            {
                "player": "Deepak Chahar",
                "team": "Chennai Super Kings",
                "recent_runs": 12,
                "batting_form": 17,
                "strike_rate": 116,
                "win_rate": 0.53,
                "recent_wickets": 1.3,
                "avg_wickets": 1.0,
                "bowling_form": 57,
                "economy": 7.8,
                "impact_score": 61,
            },
        ]
    )
    return predictor


def test_cricket_scenario_reasons_are_unique_and_capped() -> None:
    predictor = _predictor_stub()
    feature_dict = {
        "team_a_avg_runs_last_5": 178.0,
        "team_b_avg_runs_last_5": 166.0,
        "team_a_avg_wickets_last_5": 7.8,
        "team_b_avg_wickets_last_5": 6.9,
        "team_a_run_rate_trend": 0.25,
        "team_b_run_rate_trend": -0.1,
        "venue_avg_score": 174.0,
        "venue_chase_success_rate": 0.56,
        "venue_defend_bias": -0.08,
        "recent_form_diff": 6.5,
        "head_to_head_win_diff": 0.08,
        "team_a_runs_vs_opponent_avg": 5.2,
        "team_b_runs_vs_opponent_avg": -1.1,
    }

    low = predictor._cricket_scenario_reasons(
        label="Low",
        score=161.0,
        team_a="Mumbai Indians",
        team_b="Chennai Super Kings",
        feature_dict=feature_dict,
    )
    high = predictor._cricket_scenario_reasons(
        label="High",
        score=187.0,
        team_a="Mumbai Indians",
        team_b="Chennai Super Kings",
        feature_dict=feature_dict,
    )

    assert 1 <= len(low) <= 3
    assert 1 <= len(high) <= 3
    low_text = [row["explanation"] for row in low]
    high_text = [row["explanation"] for row in high]
    assert len(low_text) == len(set(low_text))
    assert len(high_text) == len(set(high_text))
    assert set(low_text) != set(high_text)
    assert all(not re.search(r"\d", text) for text in low_text)
    assert all(not re.search(r"\d", text) for text in high_text)


def test_cricket_player_predictions_return_multi_candidate_role_lists() -> None:
    predictor = _predictor_stub()
    payload = predictor._predict_cricket_players("Mumbai Indians", "Chennai Super Kings")
    players = payload["players"]

    assert len(players["top_batsmen"]) >= 2
    assert len(players["top_bowlers"]) >= 2
    assert len(players["top_match_impact"]) >= 2
    assert all(row["role"] == "batsman" for row in players["top_batsmen"])
    assert all(row["role"] == "bowler" for row in players["top_bowlers"])
    assert all(row["reason"] and len(row["reason"]) <= 2 for row in players["top_batsmen"])
    assert all(row["reason"] and len(row["reason"]) <= 2 for row in players["top_bowlers"])
    batter_reason_heads = [row["reason"][0]["explanation"] for row in players["top_batsmen"] if row.get("reason")]
    assert len(set(batter_reason_heads)) >= 2
    impact_reason_heads = [row["reason"][0]["explanation"] for row in players["top_match_impact"] if row.get("reason")]
    assert len(set(impact_reason_heads)) >= 2


def test_cricket_match_insight_is_concise_and_non_empty() -> None:
    predictor = _predictor_stub()
    insight = predictor._cricket_match_insight(
        team_a="Mumbai Indians",
        team_b="Chennai Super Kings",
        feature_dict={
            "team_a_avg_runs_last_5": 178.0,
            "team_b_avg_runs_last_5": 168.0,
            "team_a_chase_success_rate": 0.56,
            "team_b_chase_success_rate": 0.49,
            "venue_avg_score": 175.0,
        },
        scenarios=[
            {"team_a_first": {"winner": "Mumbai Indians"}, "team_b_first": {"winner": "Chennai Super Kings"}},
        ],
        anomaly={"odd_variant_flag": False},
    )
    assert isinstance(insight, str)
    assert len(insight.strip()) > 16
    assert "  " not in insight
    assert not re.search(r"\d", insight)
