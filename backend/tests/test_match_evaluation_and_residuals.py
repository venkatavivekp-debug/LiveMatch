from __future__ import annotations

from fastapi.testclient import TestClient

from app.services.prediction_service import PredictionService


def _first_historical_match(client: TestClient) -> dict:
    response = client.get(
        "/matches",
        params={"sport": "cricket", "tournament": "IPL", "state": "historical"},
    )
    assert response.status_code == 200
    rows = response.json()
    assert rows, "No historical cricket matches available for evaluation test."
    return rows[0]


def test_match_evaluation_endpoint_contract(client: TestClient) -> None:
    match = _first_historical_match(client)
    response = client.get(
        f"/matches/{match['match_id']}/evaluation",
        params={"k": 4},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    evaluation = payload["evaluation"]

    assert payload["match_id"] == match["match_id"]
    assert "prediction" in payload
    assert {"available", "target_type"}.issubset(evaluation.keys())
    assert isinstance(evaluation.get("evaluation_summary"), str)
    if evaluation["available"]:
        assert "best_match_error" in evaluation
        assert evaluation.get("best_match_error_method") in {"pair_rmse", "first_innings_abs"}
        assert "interval_covered" in evaluation
        assert "winner_correct" in evaluation
        if evaluation.get("best_match_error_method") == "pair_rmse":
            assert evaluation.get("best_matching_branch") in {"team_a_first", "team_b_first"}


def test_prediction_metadata_includes_residual_and_anomaly(
    client: TestClient,
) -> None:
    match = _first_historical_match(client)
    response = client.post(
        "/predict",
        json={
            "k": 4,
            "match": {
                "match_id": match["match_id"],
                "sport": "cricket",
            },
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    metadata = payload["metadata"]

    assert "residual_context" in metadata
    assert "anomaly_score" in metadata
    assert "odd_variant_flag" in metadata
    assert isinstance(metadata["odd_variant_flag"], bool)
    assert 0.0 <= float(metadata["anomaly_score"]) <= 1.0
    residual_context = metadata["residual_context"]
    assert "combined_bias" in residual_context
    assert "residual_shift_score" in residual_context

    # Service-level compatibility check for backward behavior.
    status = PredictionService.model_status()
    assert {"model", "data", "provider", "heads", "healthy"}.issubset(status.keys())


def test_evaluation_uses_correct_batting_order_branch() -> None:
    prediction_output = {
        "predictions": [
            {
                "scenario": "Low",
                "team_a_first": {"batting_team": "Team A", "batting_score": 160, "chase_score": 158, "winner": "Team A"},
                "team_b_first": {"batting_team": "Team B", "batting_score": 171, "chase_score": 169, "winner": "Team B"},
            },
            {
                "scenario": "Baseline",
                "team_a_first": {"batting_team": "Team A", "batting_score": 170, "chase_score": 168, "winner": "Team A"},
                "team_b_first": {"batting_team": "Team B", "batting_score": 176, "chase_score": 175, "winner": "Team B"},
            },
        ],
        "uncertainty": {"interval_low": 158, "interval_high": 176, "mean_prediction": 168},
    }
    summary = PredictionService._evaluate_prediction(
        prediction_output=prediction_output,
        actual_value=176.0,
        actual_second_innings=175.0,
        actual_winner="Team B",
        first_innings_team="Team B",
        actual_score_summary="Team B 176 vs Team A 175",
    )
    assert summary["available"] is True
    assert summary["best_match_error_method"] == "pair_rmse"
    assert summary["best_matching_scenario"] == "Baseline"
    assert summary["predicted_winner"] == "Team B"


def test_evaluation_handles_missing_batting_order_cleanly() -> None:
    prediction_output = {
        "predictions": [
            {"scenario": "Low", "score": 162},
            {"scenario": "Baseline", "score": 171},
            {"scenario": "High", "score": 184},
        ],
        "uncertainty": {"interval_low": 160, "interval_high": 186, "mean_prediction": 172},
    }
    summary = PredictionService._evaluate_prediction(
        prediction_output=prediction_output,
        actual_value=170.0,
        actual_second_innings=None,
        actual_winner=None,
        first_innings_team=None,
        actual_score_summary=None,
    )
    assert summary["available"] is True
    assert summary["best_match_error_method"] == "first_innings_abs"
    assert summary["best_matching_branch"] is None
