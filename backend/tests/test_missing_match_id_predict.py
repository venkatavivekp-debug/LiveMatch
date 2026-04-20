from __future__ import annotations

from fastapi.testclient import TestClient


def test_predict_allows_unknown_match_id_with_full_payload(
    client: TestClient,
) -> None:
    response = client.post(
        "/predict",
        json={
            "k": 4,
            "match": {
                "match_id": "mock_upcoming_rcb_kkr",
                "sport": "cricket",
                "tournament": "IPL",
                "team_a": "Royal Challengers Bengaluru",
                "team_b": "Kolkata Knight Riders",
                "venue": "M Chinnaswamy Stadium",
                "state": "upcoming",
            },
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["match"]["team_a"] == "Royal Challengers Bengaluru"
    assert payload["match"]["team_b"] == "Kolkata Knight Riders"
