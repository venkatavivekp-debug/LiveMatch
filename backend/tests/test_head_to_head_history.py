from __future__ import annotations

from fastapi.testclient import TestClient


def _first_historical_match(client: TestClient) -> dict:
    response = client.get("/matches", params={"sport": "cricket", "state": "historical"})
    assert response.status_code == 200
    rows = response.json()
    assert rows, "No historical cricket matches found."
    return rows[0]


def test_head_to_head_history_shape(client: TestClient) -> None:
    match = _first_historical_match(client)
    response = client.get(
        "/matches/head-to-head",
        params={
            "sport": "cricket",
            "team_a": match["team_a"],
            "team_b": match["team_b"],
            "tournament": match["tournament"],
            "limit": 3,
            "include_evaluation": True,
            "k": 4,
        },
    )
    assert response.status_code == 200, response.text
    rows = response.json()
    assert isinstance(rows, list)
    if rows:
        first = rows[0]
        assert {"match_id", "team_a", "team_b", "actual_result"}.issubset(first.keys())
        if first.get("evaluation"):
            assert {"predicted_heads", "best_error", "in_range"}.issubset(first["evaluation"].keys())

