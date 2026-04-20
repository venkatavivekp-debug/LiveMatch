from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from app.services.providers.cache_store import JSONFileCacheStore
from app.services.providers.cricapi_realtime_provider import CricAPIRealtimeProvider


def test_cricapi_provider_uses_cache_without_api_key(tmp_path) -> None:
    cache_path = tmp_path / "live_cache.json"
    cache = JSONFileCacheStore(path=cache_path)
    cache.set(
        "cricapi_current_matches",
        {
            "data": [
                {
                    "id": "live_match_1",
                    "teams": ["Team A", "Team B"],
                    "status": "Live",
                    "venue": "Demo Ground",
                    "date": "2026-04-16",
                    "score": [{"r": 178, "w": 5, "o": 19.0}],
                }
            ]
        },
    )

    provider = CricAPIRealtimeProvider(
        api_key="",
        base_url="https://api.cricapi.com/v1",
        timeout_seconds=2.0,
        max_retries=0,
        backoff_seconds=0.1,
        cache_ttl_seconds=1200,
        stale_cache_ttl_seconds=3600,
        cache_store=cache,
    )
    live_rows = provider.fetch_live_matches(limit=5)
    assert live_rows
    assert live_rows[0]["match_id"] == "live_match_1"
    assert str(live_rows[0].get("provider_source", "")).startswith("cache")


def test_cricapi_provider_handles_corrupted_cache_file(tmp_path) -> None:
    cache_path = tmp_path / "broken_cache.json"
    cache_path.write_text("{broken-json", encoding="utf-8")
    cache = JSONFileCacheStore(path=cache_path)

    provider = CricAPIRealtimeProvider(
        api_key="",
        base_url="https://api.cricapi.com/v1",
        timeout_seconds=2.0,
        max_retries=0,
        backoff_seconds=0.1,
        cache_ttl_seconds=1200,
        stale_cache_ttl_seconds=3600,
        cache_store=cache,
    )

    rows = provider.fetch_live_matches(limit=5)
    assert rows == []
    health = provider.healthcheck()
    assert health["status"] == "unavailable"


def test_cricapi_provider_uses_stale_cache_on_live_api_error(tmp_path, monkeypatch) -> None:
    cache_path = tmp_path / "live_cache.json"
    old_time = (datetime.now(timezone.utc) - timedelta(minutes=3)).isoformat()
    payload = {
        "cricapi_current_matches": {
            "updated_at": old_time,
            "payload": {
                "data": [
                    {
                        "id": "stale_match_1",
                        "teams": ["Team A", "Team B"],
                        "status": "Live",
                        "venue": "Demo Ground",
                        "date": "2026-04-16",
                        "score": [{"r": 175, "w": 4, "o": 19.2}],
                    }
                ]
            },
        }
    }
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    cache = JSONFileCacheStore(path=cache_path)

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001, ANN201
            return False

        @staticmethod
        def read() -> bytes:
            return b'{"status":"error","message":"Invalid API key"}'

    monkeypatch.setattr("app.services.providers.cricapi_realtime_provider.urlopen", lambda *args, **kwargs: _FakeResponse())

    provider = CricAPIRealtimeProvider(
        api_key="bad-key",
        base_url="https://api.cricapi.com/v1",
        timeout_seconds=2.0,
        max_retries=0,
        backoff_seconds=0.1,
        cache_ttl_seconds=60,
        stale_cache_ttl_seconds=3600,
        cache_store=cache,
    )

    live_rows = provider.fetch_live_matches(limit=5)
    assert live_rows
    assert live_rows[0]["match_id"] == "stale_match_1"
    assert live_rows[0]["provider_source"] == "cache-stale"


def test_cricapi_provider_handles_partial_match_payload(tmp_path) -> None:
    cache_path = tmp_path / "live_cache.json"
    cache = JSONFileCacheStore(path=cache_path)
    cache.set(
        "cricapi_current_matches",
        {
            "data": [
                {
                    "id": "partial_1",
                    "teamInfo": [{"name": "Only Team A"}],
                    "status": "Live",
                }
            ]
        },
    )

    provider = CricAPIRealtimeProvider(
        api_key="",
        base_url="https://api.cricapi.com/v1",
        timeout_seconds=2.0,
        max_retries=0,
        backoff_seconds=0.1,
        cache_ttl_seconds=1200,
        stale_cache_ttl_seconds=3600,
        cache_store=cache,
    )

    rows = provider.fetch_live_matches(limit=5)
    assert rows == []


def test_cricapi_provider_tournament_alias_filter(tmp_path) -> None:
    cache_path = tmp_path / "live_cache.json"
    cache = JSONFileCacheStore(path=cache_path)
    cache.set(
        "cricapi_matches",
        {
            "data": [
                {
                    "id": "upcoming_alias_1",
                    "teams": ["Mumbai Indians", "Chennai Super Kings"],
                    "status": "Scheduled",
                    "series": {"name": "Indian Premier League 2026"},
                    "venue": "Wankhede Stadium",
                    "dateTimeGMT": "2026-04-30T14:00:00Z",
                }
            ]
        },
    )
    provider = CricAPIRealtimeProvider(
        api_key="",
        base_url="https://api.cricapi.com/v1",
        timeout_seconds=2.0,
        max_retries=0,
        backoff_seconds=0.1,
        cache_ttl_seconds=1200,
        stale_cache_ttl_seconds=3600,
        cache_store=cache,
    )
    rows = provider.fetch_upcoming_matches(tournament="IPL", limit=5)
    assert rows
    assert rows[0]["match_id"] == "upcoming_alias_1"


def test_cricapi_provider_skips_tbc_and_completed_rows_for_upcoming(tmp_path) -> None:
    cache_path = tmp_path / "live_cache.json"
    cache = JSONFileCacheStore(path=cache_path)
    cache.set(
        "cricapi_current_matches",
        {
            "data": [
                {
                    "id": "live_bad_tbc",
                    "teams": ["Tbc", "Mumbai Indians"],
                    "status": "Live",
                    "venue": "Demo Ground",
                    "dateTimeGMT": "2026-04-30T14:00:00Z",
                },
                {
                    "id": "done_row",
                    "teams": ["Team A", "Team B"],
                    "status": "Match over",
                    "venue": "Demo Ground",
                    "dateTimeGMT": "2026-04-25T14:00:00Z",
                },
            ]
        },
    )
    cache.set(
        "cricapi_matches",
        {
            "data": [
                {
                    "id": "scheduled_good",
                    "teams": ["Mumbai Indians", "Chennai Super Kings"],
                    "status": "Scheduled",
                    "series": {"name": "Indian Premier League 2026"},
                    "venue": "Wankhede Stadium",
                    "dateTimeGMT": "2026-04-30T14:00:00Z",
                },
                {
                    "id": "scheduled_bad_tbd",
                    "teams": ["TBD", "TBC"],
                    "status": "Scheduled",
                    "series": {"name": "Indian Premier League 2026"},
                    "venue": "Venue TBA",
                    "dateTimeGMT": "2026-04-30T15:00:00Z",
                },
            ]
        },
    )

    provider = CricAPIRealtimeProvider(
        api_key="",
        base_url="https://api.cricapi.com/v1",
        timeout_seconds=2.0,
        max_retries=0,
        backoff_seconds=0.1,
        cache_ttl_seconds=1200,
        stale_cache_ttl_seconds=3600,
        cache_store=cache,
    )
    rows = provider.fetch_upcoming_matches(tournament="IPL", limit=10)
    assert len(rows) == 1
    assert rows[0]["match_id"] == "scheduled_good"
    assert rows[0]["state"] == "upcoming"


def test_cricapi_provider_reads_upcoming_from_matches_endpoint(tmp_path) -> None:
    cache_path = tmp_path / "live_cache.json"
    cache = JSONFileCacheStore(path=cache_path)
    cache.set("cricapi_current_matches", {"data": []})
    cache.set(
        "cricapi_matches",
        {
            "data": [
                {
                    "id": "sched_only_matches_endpoint",
                    "teams": ["Delhi Capitals", "Gujarat Titans"],
                    "status": "Scheduled",
                    "series": {"name": "Indian Premier League 2026"},
                    "venue": "Arun Jaitley Stadium",
                    "dateTimeGMT": "2026-05-02T14:00:00Z",
                }
            ]
        },
    )
    provider = CricAPIRealtimeProvider(
        api_key="",
        base_url="https://api.cricapi.com/v1",
        timeout_seconds=2.0,
        max_retries=0,
        backoff_seconds=0.1,
        cache_ttl_seconds=1200,
        stale_cache_ttl_seconds=3600,
        cache_store=cache,
    )
    rows = provider.fetch_upcoming_matches(tournament="IPL", limit=5)
    assert rows
    assert rows[0]["match_id"] == "sched_only_matches_endpoint"


def test_cricapi_provider_keeps_not_started_rows_as_upcoming(tmp_path) -> None:
    cache_path = tmp_path / "live_cache.json"
    cache = JSONFileCacheStore(path=cache_path)
    cache.set(
        "cricapi_matches",
        {
            "data": [
                {
                    "id": "not_started_status",
                    "teams": ["Mumbai Indians", "Chennai Super Kings"],
                    "status": "Match not started",
                    "series": {"name": "Indian Premier League 2026"},
                    "venue": "Wankhede Stadium",
                    "dateTimeGMT": "2026-05-04T14:00:00Z",
                }
            ]
        },
    )
    provider = CricAPIRealtimeProvider(
        api_key="",
        base_url="https://api.cricapi.com/v1",
        timeout_seconds=2.0,
        max_retries=0,
        backoff_seconds=0.1,
        cache_ttl_seconds=1200,
        stale_cache_ttl_seconds=3600,
        cache_store=cache,
    )
    rows = provider.fetch_upcoming_matches(tournament="IPL", limit=5)
    assert rows
    assert rows[0]["match_id"] == "not_started_status"
    assert rows[0]["state"] == "upcoming"


def test_cricapi_provider_uses_start_time_when_status_is_blank(tmp_path) -> None:
    cache_path = tmp_path / "live_cache.json"
    cache = JSONFileCacheStore(path=cache_path)
    cache.set(
        "cricapi_matches",
        {
            "data": [
                {
                    "id": "future_time_without_status",
                    "teams": ["Delhi Capitals", "Gujarat Titans"],
                    "status": "",
                    "series": {"name": "Indian Premier League 2026"},
                    "venue": "Arun Jaitley Stadium",
                    "dateTimeGMT": "2026-06-02T14:00:00Z",
                }
            ]
        },
    )
    provider = CricAPIRealtimeProvider(
        api_key="",
        base_url="https://api.cricapi.com/v1",
        timeout_seconds=2.0,
        max_retries=0,
        backoff_seconds=0.1,
        cache_ttl_seconds=1200,
        stale_cache_ttl_seconds=3600,
        cache_store=cache,
    )
    rows = provider.fetch_upcoming_matches(tournament="IPL", limit=5)
    assert rows
    assert rows[0]["match_id"] == "future_time_without_status"
    assert rows[0]["state"] == "upcoming"


def test_cricapi_provider_supports_team1_team2_payload_shape(tmp_path) -> None:
    cache_path = tmp_path / "live_cache.json"
    cache = JSONFileCacheStore(path=cache_path)
    cache.set(
        "cricapi_matches",
        {
            "matches": [
                {
                    "id": "team1_team2_shape",
                    "team1": {"name": "Royal Challengers Bengaluru"},
                    "team2": {"name": "Kolkata Knight Riders"},
                    "status": "scheduled",
                    "series": {"name": "Indian Premier League 2026"},
                    "venue": "M Chinnaswamy Stadium",
                    "dateTimeGMT": "2026-06-10T14:00:00Z",
                }
            ]
        },
    )
    provider = CricAPIRealtimeProvider(
        api_key="",
        base_url="https://api.cricapi.com/v1",
        timeout_seconds=2.0,
        max_retries=0,
        backoff_seconds=0.1,
        cache_ttl_seconds=1200,
        stale_cache_ttl_seconds=3600,
        cache_store=cache,
    )
    rows = provider.fetch_upcoming_matches(tournament="IPL", limit=5)
    assert rows
    assert rows[0]["team_a"] == "Royal Challengers Bengaluru"
    assert rows[0]["team_b"] == "Kolkata Knight Riders"


def test_cricapi_provider_treats_match_ended_false_as_upcoming_when_not_live(tmp_path) -> None:
    cache_path = tmp_path / "live_cache.json"
    cache = JSONFileCacheStore(path=cache_path)
    cache.set(
        "cricapi_matches",
        {
            "data": [
                {
                    "id": "ended_false_upcoming",
                    "teams": ["Punjab Kings", "Rajasthan Royals"],
                    "status": "",
                    "matchEnded": False,
                    "series": {"name": "Indian Premier League 2026"},
                    "venue": "Jaipur",
                    "dateTimeGMT": "2026-06-12T14:00:00Z",
                }
            ]
        },
    )
    provider = CricAPIRealtimeProvider(
        api_key="",
        base_url="https://api.cricapi.com/v1",
        timeout_seconds=2.0,
        max_retries=0,
        backoff_seconds=0.1,
        cache_ttl_seconds=1200,
        stale_cache_ttl_seconds=3600,
        cache_store=cache,
    )
    rows = provider.fetch_upcoming_matches(tournament="IPL", limit=5)
    assert rows
    assert rows[0]["match_id"] == "ended_false_upcoming"
    assert rows[0]["state"] == "upcoming"
