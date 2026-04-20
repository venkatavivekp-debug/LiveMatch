from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_catalog_endpoints(client: TestClient) -> None:
    sports = client.get("/sports")
    assert sports.status_code == 200
    sport_rows = sports.json()
    assert any(row["code"] == "cricket" for row in sport_rows)
    assert any(row["code"] == "football" for row in sport_rows)

    tournaments = client.get("/tournaments", params={"sport": "football"})
    assert tournaments.status_code == 200
    assert all(row["sport"] == "football" for row in tournaments.json())

    matches = client.get("/matches", params={"sport": "cricket", "state": "upcoming"})
    assert matches.status_code == 200
    assert isinstance(matches.json(), list)


def test_model_status_endpoint(client: TestClient) -> None:
    response = client.get("/model/status")
    assert response.status_code == 200
    payload = response.json()
    assert {"model", "data", "provider", "heads", "last_update", "healthy"}.issubset(payload.keys())
    assert "runtime_mode" in payload
    assert "model_mode" in payload
    assert "data_mode" in payload
    assert "num_heads" in payload
    assert "sports_supported" in payload
    assert "model_version" in payload
    assert "checkpoint_updated_at" in payload
    assert "latest_evaluation_summary" in payload
    assert "artifact_paths" in payload
    assert "encoder_type" in payload
    assert "encoder_config" in payload
    assert "calibration" in payload
    assert "provider_status" in payload


def test_system_status_and_refresh_routes(client: TestClient) -> None:
    system = client.get("/system/status")
    assert system.status_code == 200, system.text
    payload = system.json()
    assert "generated_at" in payload
    assert "model" in payload
    assert "provider" in payload

    refresh = client.post(
        "/admin/refresh-live-data",
        json={"sport": "cricket", "tournament": "IPL"},
    )
    assert refresh.status_code == 200, refresh.text
    refresh_payload = refresh.json()
    assert refresh_payload["status"] in {"ok", "failed", "ignored"}
    assert "provider" in refresh_payload
