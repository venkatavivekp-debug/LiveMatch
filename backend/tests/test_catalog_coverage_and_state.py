from __future__ import annotations


def test_tournaments_include_coverage_fields(client) -> None:
    response = client.get("/tournaments", params={"sport": "cricket"})
    assert response.status_code == 200, response.text
    rows = response.json()
    assert isinstance(rows, list)
    if rows:
        first = rows[0]
        assert "matches_available" in first
        assert "seasons" in first


def test_matches_state_validation_and_completed_alias(client) -> None:
    invalid = client.get("/matches", params={"sport": "cricket", "state": "weird"})
    assert invalid.status_code == 422

    completed = client.get("/matches", params={"sport": "cricket", "state": "completed"})
    assert completed.status_code == 200, completed.text
    rows = completed.json()
    assert isinstance(rows, list)
