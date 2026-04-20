from __future__ import annotations

from app.services.insights_service import InsightsService


class _NoLiveProvider:
    provider_name = "test-live"

    @staticmethod
    def fetch_live_matches(**kwargs):  # noqa: ANN001, ANN003
        return []

    @staticmethod
    def fetch_upcoming_matches(**kwargs):  # noqa: ANN001, ANN003
        return []


def test_live_insights_do_not_fallback_to_historical(monkeypatch) -> None:
    monkeypatch.setattr("app.services.insights_service.get_live_provider", lambda: _NoLiveProvider())

    payload = InsightsService.live_insights(
        sport="cricket",
        tournament="IPL",
        state="live",
        limit=4,
    )
    assert payload["provider"] == "test-live"
    assert payload["cards"] == []


def test_live_insights_ignore_historical_state(monkeypatch) -> None:
    monkeypatch.setattr("app.services.insights_service.get_live_provider", lambda: _NoLiveProvider())

    payload = InsightsService.live_insights(
        sport="cricket",
        tournament="IPL",
        state="historical",
        limit=4,
    )
    assert payload["cards"] == []
