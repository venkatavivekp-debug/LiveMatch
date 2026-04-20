from __future__ import annotations

import re

from fastapi.testclient import TestClient


def _first_match(client: TestClient, sport: str = "cricket") -> dict:
    response = client.get("/matches", params={"sport": sport})
    assert response.status_code == 200
    rows = response.json()
    assert rows, f"No matches returned for sport={sport}"
    return rows[0]


def test_predict_response_shape(client: TestClient) -> None:
    match = _first_match(client, sport="cricket")
    payload = {
        "k": 4,
        "match": {
            "match_id": match["match_id"],
            "sport": match["sport"],
            "tournament": match["tournament"],
            "team_a": match["team_a"],
            "team_b": match["team_b"],
            "venue": match["venue"],
            "state": match["state"],
            "match_date": match.get("match_date"),
        },
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()

    assert data["match"]["sport"] == "cricket"
    assert len(data["predictions"]) == 4
    assert len(data["scenarios"]) == 4
    allowed_labels = {"Low", "Baseline", "High", "Aggressive"}
    assert all(row["scenario"] in allowed_labels for row in data["predictions"])
    assert all(row["confidence"] == round(row["confidence"], 2) for row in data["predictions"])
    assert {"spread", "interval_low", "interval_high", "mean_prediction"}.issubset(data["uncertainty"].keys())
    assert "forecast_summary" in data
    summary = data["forecast_summary"] or {}
    assert {
        "favored_team",
        "favored_team_confidence",
        "win_probability",
        "expected_score_range",
        "predicted_band_low",
        "predicted_band_high",
        "key_risk",
        "risk_level",
        "risk_explanation",
        "final_summary",
    }.issubset(
        summary.keys()
    )
    assert isinstance(summary.get("favored_team"), str)
    confidence = float(summary.get("favored_team_confidence") or 0.0)
    assert 0.0 <= confidence <= 1.0
    win_prob = float(summary.get("win_probability") or 0.0)
    assert 0.0 <= win_prob <= 1.0
    favored = str(summary.get("favored_team") or "")
    preds = data["predictions"]
    ta = str(data["match"]["team_a"]).strip().lower()
    tb = str(data["match"]["team_b"]).strip().lower()

    def _scenario_weight(row: dict) -> float:
        p = row.get("scenario_probability")
        if p is not None and float(p) > 0:
            return float(p)
        c = row.get("confidence")
        if c is not None and float(c) > 0:
            return float(c)
        return 1.0

    w_a = sum(_scenario_weight(p) for p in preds if str(p.get("winner") or "").strip().lower() == ta)
    w_b = sum(_scenario_weight(p) for p in preds if str(p.get("winner") or "").strip().lower() == tb)
    if w_a + w_b > 0 and favored:
        fav_lower = favored.strip().lower()
        expected = (w_a if fav_lower == ta else w_b) / (w_a + w_b)
        assert abs(win_prob - round(expected, 2)) <= 0.02
    assert summary.get("risk_level") in {"Low", "Medium", "High"}
    assert isinstance(summary.get("risk_explanation"), str)
    assert isinstance(summary.get("final_summary"), str)
    assert len(str(summary.get("final_summary")).strip()) >= 20
    assert "while" in str(summary.get("final_summary")).lower()
    assert str(summary.get("favored_team", "")).lower() in str(summary.get("final_summary", "")).lower()
    assert "performance_summary" in data
    perf = data["performance_summary"] or {}
    assert {"reliability", "accuracy", "avg_error", "in_range_pct", "samples"}.issubset(perf.keys())
    assert isinstance(data.get("match_insight"), str)
    assert len(str(data.get("match_insight")).strip()) >= 12
    assert data["best_player"]["name"]
    assert data["man_of_the_match"]["name"]
    assert not re.search(r"\bplayer\s*\d+\b", data["best_player"]["name"], flags=re.IGNORECASE)
    assert not re.search(r"\bplayer\s*\d+\b", data["man_of_the_match"]["name"], flags=re.IGNORECASE)
    players = data.get("players") or {}
    for key in ["top_batsmen", "top_bowlers", "top_match_impact"]:
        for row in players.get(key, []):
            assert not re.search(r"\bplayer\s*\d+\b", str(row.get("name", "")), flags=re.IGNORECASE)
            assert len(row.get("reason", [])) <= 2
    if players.get("top_batsmen"):
        assert len(players["top_batsmen"]) >= 2
    if players.get("top_bowlers"):
        assert len(players["top_bowlers"]) >= 2
    first_reason = data["predictions"][0]["reason"][0]
    assert {"feature", "impact", "explanation"}.issubset(first_reason.keys())
    first_branch = data["predictions"][0]
    assert {"team_a_first", "team_b_first"}.issubset(first_branch.keys())
    assert {"team_a_score", "team_b_score", "winner"}.issubset(first_branch.keys())
    assert {"batting_score", "chase_score", "winner"}.issubset(first_branch["team_a_first"].keys())
    assert {"batting_score", "chase_score", "winner"}.issubset(first_branch["team_b_first"].keys())
    score_pairs = {(row["team_a_score"], row["team_b_score"]) for row in data["predictions"]}
    assert len(score_pairs) >= 2
    for row in data["predictions"]:
        explanations = [str(reason.get("explanation", "")) for reason in row.get("reason", [])]
        assert len(row.get("reason", [])) <= 3
        assert len(explanations) == len(set(explanations))
    confidence_values = [float(row.get("confidence") or 0.0) for row in data["predictions"]]
    assert all(0.0 <= value <= 1.0 for value in confidence_values)
    reason_signatures = {
        tuple(str(reason.get("explanation", "")) for reason in row.get("reason", []))
        for row in data["predictions"]
    }
    assert len(reason_signatures) >= 2
    for row in data["predictions"]:
        branch_a = row.get("team_a_first") or {}
        branch_b = row.get("team_b_first") or {}
        branch_a_winner = str(branch_a.get("winner") or "").strip()
        branch_b_winner = str(branch_b.get("winner") or "").strip()
        if branch_a_winner and branch_a_winner == branch_b_winner:
            assert row["winner"] == branch_a_winner
        else:
            assert row["winner"] in {data["match"]["team_a"], data["match"]["team_b"]}
    assert "calibration" in data["metadata"]
    assert "data_mode" in data["metadata"]
    assert "provider_used" in data["metadata"]
    assert "ensemble_disagreement_score" in data["metadata"]
    assert "low_uncertainty_case" in data["metadata"]


def test_batch_and_forecast_endpoints(client: TestClient) -> None:
    cricket = _first_match(client, sport="cricket")
    football = _first_match(client, sport="football")

    batch_payload = {
        "requests": [
            {
                "k": 3,
                "match": {
                    "match_id": cricket["match_id"],
                    "sport": cricket["sport"],
                    "tournament": cricket["tournament"],
                    "team_a": cricket["team_a"],
                    "team_b": cricket["team_b"],
                    "venue": cricket["venue"],
                    "state": cricket["state"],
                },
            },
            {
                "k": 4,
                "match": {
                    "match_id": football["match_id"],
                    "sport": football["sport"],
                    "tournament": football["tournament"],
                    "team_a": football["team_a"],
                    "team_b": football["team_b"],
                    "venue": football["venue"],
                    "state": football["state"],
                },
            },
        ]
    }

    batch_response = client.post("/predict/batch", json=batch_payload)
    assert batch_response.status_code == 200, batch_response.text
    batch_data = batch_response.json()
    assert batch_data["total"] == 2
    assert batch_data["success"] >= 1

    scenarios_response = client.get(
        "/forecast/scenarios",
        params={"match_id": cricket["match_id"], "sport": "cricket", "k": 4},
    )
    assert scenarios_response.status_code == 200, scenarios_response.text
    assert len(scenarios_response.json()) == 4

    uncertainty_response = client.get(
        "/forecast/uncertainty",
        params={"match_id": football["match_id"], "sport": "football", "k": 4},
    )
    assert uncertainty_response.status_code == 200, uncertainty_response.text
    assert "spread" in uncertainty_response.json()

    football_predict = client.post(
        "/predict",
        json={
            "k": 4,
            "match": {
                "match_id": football["match_id"],
                "sport": football["sport"],
                "tournament": football["tournament"],
                "team_a": football["team_a"],
                "team_b": football["team_b"],
                "venue": football["venue"],
                "state": football["state"],
            },
        },
    )
    assert football_predict.status_code == 200, football_predict.text
    football_payload = football_predict.json()
    assert all("likely_result" in row for row in football_payload["predictions"])
    assert isinstance(football_payload.get("match_insight"), str)
    assert not re.search(r"\bplayer\s*\d+\b", football_payload["best_player"]["name"], flags=re.IGNORECASE)
    assert not re.search(r"\bplayer\s*\d+\b", football_payload["man_of_the_match"]["name"], flags=re.IGNORECASE)
