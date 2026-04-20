from __future__ import annotations

from app.services.prediction_service import PredictionService


class _UnavailableLiveProvider:
    provider_name = "test-unavailable"

    @staticmethod
    def fetch_match_context(**kwargs):  # noqa: ANN003, ANN001
        return {"provider": "test-unavailable", "source": "unavailable", "features": {}}

    @staticmethod
    def healthcheck():
        return {"provider": "test-unavailable", "status": "unavailable", "source": "error"}


class _LiveContextProvider:
    provider_name = "test-live"

    @staticmethod
    def fetch_match_context(**kwargs):  # noqa: ANN003, ANN001
        return {
            "provider": "test-live",
            "source": "live",
            "updated_at": "2026-04-16T00:00:00+00:00",
            "freshness_seconds": 12,
            "features": {
                "batting_team_avg_runs_last5": 191.0,
                "bowling_team_avg_runs_conceded_last5": 184.0,
            },
        }

    @staticmethod
    def healthcheck():
        return {"provider": "test-live", "status": "ok", "source": "live"}


def test_prediction_data_mode_fallback_when_live_unavailable(monkeypatch) -> None:
    PredictionService._load_predictor.cache_clear()
    monkeypatch.setattr(PredictionService, "_load_predictor", staticmethod(lambda: None))
    monkeypatch.setattr("app.services.prediction_service.get_live_provider", lambda: _UnavailableLiveProvider())

    result = PredictionService.predict(
        match_payload={
            "sport": "cricket",
            "tournament": "IPL",
            "team_a": "Mumbai Indians",
            "team_b": "Chennai Super Kings",
            "venue": "Wankhede Stadium",
            "state": "upcoming",
        },
        k=4,
    )
    assert result["metadata"]["model_mode"] == "fallback"
    assert result["metadata"]["data_mode"] == "FALLBACK"


def test_prediction_data_mode_live_when_live_context_present(monkeypatch) -> None:
    PredictionService._load_predictor.cache_clear()
    monkeypatch.setattr(PredictionService, "_load_predictor", staticmethod(lambda: None))
    monkeypatch.setattr("app.services.prediction_service.get_live_provider", lambda: _LiveContextProvider())

    result = PredictionService.predict(
        match_payload={
            "sport": "cricket",
            "tournament": "IPL",
            "team_a": "Mumbai Indians",
            "team_b": "Chennai Super Kings",
            "venue": "Wankhede Stadium",
            "state": "upcoming",
        },
        k=4,
    )
    assert result["metadata"]["data_mode"] == "LIVE"
    assert result["metadata"]["provider_used"] == "test-live"
