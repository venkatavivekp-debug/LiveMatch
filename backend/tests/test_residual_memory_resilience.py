from __future__ import annotations

from app.services.residual_memory_service import ResidualMemoryService


class _BrokenSession:
    @staticmethod
    def query(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("db unavailable")

    @staticmethod
    def close() -> None:
        return None

    @staticmethod
    def rollback() -> None:
        return None


def test_inference_context_handles_missing_table(monkeypatch) -> None:
    monkeypatch.setattr("app.services.residual_memory_service.SessionLocal", lambda: _BrokenSession())
    context = ResidualMemoryService.inference_context(
        {
            "sport": "cricket",
            "team_a": "Mumbai Indians",
            "team_b": "Chennai Super Kings",
            "venue": "Wankhede",
            "tournament": "IPL",
            "state": "historical",
        }
    )
    assert context["source"] == "unavailable"
    assert context["combined_bias"] == 0.0


def test_record_handles_missing_table(monkeypatch) -> None:
    monkeypatch.setattr("app.services.residual_memory_service.SessionLocal", lambda: _BrokenSession())
    record_id = ResidualMemoryService.record(
        match_payload={"sport": "cricket", "team_a": "A", "team_b": "B"},
        prediction_output={"uncertainty": {"mean_prediction": 170, "interval_low": 160, "interval_high": 180}},
        evaluation_summary={"actual_value": 175, "best_head_value": 172, "best_match_error": 3},
    )
    assert record_id is None
