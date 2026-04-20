from __future__ import annotations

from app.services.prediction_service import PredictionService


def test_prediction_service_fallback_mode(monkeypatch) -> None:
    # Ensure cache is clear before forcing fallback predictor path.
    PredictionService._load_predictor.cache_clear()
    monkeypatch.setattr(PredictionService, "_load_predictor", staticmethod(lambda: None))

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
    assert len(result["predictions"]) == 4
    assert "spread" in result["uncertainty"]
    assert result["best_player"]["name"]
    assert "calibration" in result["metadata"]


class _BrokenPredictor:
    model_loaded = True
    num_heads = 3

    @staticmethod
    def predict(match_payload, k):  # noqa: ANN001
        raise RuntimeError("boom")


def test_prediction_service_degrades_to_fallback_when_predictor_raises(monkeypatch) -> None:
    PredictionService._load_predictor.cache_clear()
    monkeypatch.setattr(PredictionService, "_load_predictor", staticmethod(lambda: _BrokenPredictor()))

    result = PredictionService.predict(
        match_payload={
            "sport": "football",
            "tournament": "EPL",
            "team_a": "Arsenal",
            "team_b": "Manchester City",
            "venue": "Emirates Stadium",
            "state": "upcoming",
        },
        k=4,
    )

    assert result["metadata"]["model_mode"] == "fallback"
    assert len(result["predictions"]) == 4
    assert "calibration" in result["metadata"]
