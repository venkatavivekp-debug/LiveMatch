from __future__ import annotations

from types import SimpleNamespace

from app.services.catalog_service import CatalogService
from app.services.providers.local_demo_provider import LocalDemoProvider


class _MockLiveProvider:
    @staticmethod
    def fetch_live_matches(tournament=None, limit=20):  # noqa: ANN001, ANN201
        return [
            {
                "match_id": "mock_live_test",
                "sport": "cricket",
                "tournament": "IPL",
                "team_a": "Mumbai Indians",
                "team_b": "Chennai Super Kings",
                "venue": "Wankhede Stadium",
                "match_date": "2026-04-17",
                "state": "live",
                "provider": "mock-realtime",
                "provider_source": "mock",
            }
        ]

    @staticmethod
    def fetch_upcoming_matches(tournament=None, limit=20):  # noqa: ANN001, ANN201
        return []

    @staticmethod
    def healthcheck():  # noqa: ANN201
        return {"provider": "mock-realtime", "status": "ok"}


class _UnavailableLiveProvider:
    @staticmethod
    def fetch_live_matches(tournament=None, limit=20):  # noqa: ANN001, ANN201
        return []

    @staticmethod
    def fetch_upcoming_matches(tournament=None, limit=20):  # noqa: ANN001, ANN201
        return []

    @staticmethod
    def healthcheck():  # noqa: ANN201
        return {"provider": "cricapi", "status": "unavailable"}


class _EmptyHealthyLiveProvider:
    @staticmethod
    def fetch_live_matches(tournament=None, limit=20):  # noqa: ANN001, ANN201
        return []

    @staticmethod
    def fetch_upcoming_matches(tournament=None, limit=20):  # noqa: ANN001, ANN201
        return []

    @staticmethod
    def healthcheck():  # noqa: ANN201
        return {"provider": "cricapi", "status": "ok", "source": "live"}


class _SparseLiveProvider:
    @staticmethod
    def fetch_live_matches(tournament=None, limit=20):  # noqa: ANN001, ANN201
        return []

    @staticmethod
    def fetch_upcoming_matches(tournament=None, limit=20):  # noqa: ANN001, ANN201
        return [
            {
                "match_id": "sparse_upcoming_1",
                "sport": "cricket",
                "tournament": "",
                "team_a": "Mumbai Indians",
                "team_b": "Chennai Super Kings",
                "venue": "",
                "match_date": "2026-05-15",
                "state": "upcoming",
                "provider": "cricapi",
                "provider_source": "live",
            }
        ]

    @staticmethod
    def healthcheck():  # noqa: ANN201
        return {"provider": "cricapi", "status": "ok", "source": "live"}


def test_catalog_merge_skips_mock_rows_in_primary_mode(monkeypatch) -> None:
    monkeypatch.setattr("app.services.catalog_service.get_live_provider", lambda: _MockLiveProvider())
    monkeypatch.setattr(
        "app.services.catalog_service.get_settings",
        lambda: SimpleNamespace(realtime_provider="cricapi"),
    )

    base_rows = [
        {
            "match_id": "historical_1",
            "sport": "cricket",
            "tournament": "IPL",
            "team_a": "Team A",
            "team_b": "Team B",
            "venue": "Venue",
            "match_date": "2025-05-01",
            "state": "historical",
        }
    ]
    merged = CatalogService._merge_live_rows(
        rows=base_rows,
        sport="cricket",
        tournament="IPL",
        state="live",
        team=None,
        venue=None,
    )
    assert merged == []


def test_catalog_historical_mode_keeps_only_historical_rows(monkeypatch) -> None:
    monkeypatch.setattr("app.services.catalog_service.get_live_provider", lambda: _SparseLiveProvider())
    monkeypatch.setattr(
        "app.services.catalog_service.get_settings",
        lambda: SimpleNamespace(realtime_provider="cricapi"),
    )
    base_rows = [
        {
            "match_id": "historical_1",
            "sport": "cricket",
            "tournament": "IPL",
            "team_a": "Team A",
            "team_b": "Team B",
            "venue": "Venue",
            "match_date": "2025-05-01",
            "state": "historical",
        }
    ]
    merged = CatalogService._merge_live_rows(
        rows=base_rows,
        sport="cricket",
        tournament="IPL",
        state="historical",
        team=None,
        venue=None,
    )
    assert len(merged) == 1
    assert merged[0]["match_id"] == "historical_1"
    assert merged[0]["state"] == "historical"


def test_local_provider_without_player_files_returns_empty_players(tmp_path) -> None:
    provider = LocalDemoProvider(processed_dir=tmp_path)
    assert provider.get_top_players(sport="cricket", limit=5) == []
    assert provider.get_top_players(sport="football", limit=5) == []


def test_local_provider_unsupported_tournament_filter_returns_empty(tmp_path) -> None:
    provider = LocalDemoProvider(processed_dir=tmp_path)
    rows = provider.list_matches(sport="cricket", tournament="NON_EXISTENT_LEAGUE")
    assert rows == []


def test_catalog_upcoming_empty_when_live_provider_unavailable(monkeypatch) -> None:
    monkeypatch.setattr("app.services.catalog_service.get_live_provider", lambda: _UnavailableLiveProvider())
    monkeypatch.setattr(
        "app.services.catalog_service.get_settings",
        lambda: SimpleNamespace(realtime_provider="cricapi"),
    )

    merged = CatalogService._merge_live_rows(
        rows=[],
        sport="cricket",
        tournament="IPL",
        state="upcoming",
        team=None,
        venue=None,
    )
    assert merged == []


def test_catalog_upcoming_no_fallback_when_provider_is_healthy_but_empty(monkeypatch) -> None:
    monkeypatch.setattr("app.services.catalog_service.get_live_provider", lambda: _EmptyHealthyLiveProvider())
    monkeypatch.setattr(
        "app.services.catalog_service.get_settings",
        lambda: SimpleNamespace(realtime_provider="cricapi"),
    )
    merged = CatalogService._merge_live_rows(
        rows=[],
        sport="cricket",
        tournament="IPL",
        state="upcoming",
        team=None,
        venue=None,
    )
    assert merged == []


def test_catalog_live_merge_accepts_sparse_provider_rows(monkeypatch) -> None:
    monkeypatch.setattr("app.services.catalog_service.get_live_provider", lambda: _SparseLiveProvider())
    monkeypatch.setattr(
        "app.services.catalog_service.get_settings",
        lambda: SimpleNamespace(realtime_provider="cricapi"),
    )
    merged = CatalogService._merge_live_rows(
        rows=[],
        sport="cricket",
        tournament="IPL",
        state="upcoming",
        team=None,
        venue=None,
    )
    assert merged
    assert merged[0]["match_id"] == "sparse_upcoming_1"
    assert merged[0]["venue"] == "Venue TBA"
    assert merged[0]["tournament"] == "IPL"
