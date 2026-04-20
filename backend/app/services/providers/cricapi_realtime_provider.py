from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from app.services.providers.base import RealtimeContextProvider
from app.services.providers.cache_store import CacheHit, JSONFileCacheStore

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_lower(value: Any) -> str:
    return str(value or "").strip().lower()


def _safe_text(value: Any, default: str) -> str:
    text = str(value or "").strip()
    return text or default


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", _safe_lower(value))


_APIKEY_REDACT_RE = re.compile(r"(apikey=)([^&\s]+)", flags=re.IGNORECASE)


def _redact_sensitive(text: Any) -> str:
    if text is None:
        return ""
    raw = str(text)
    if not raw:
        return ""
    return _APIKEY_REDACT_RE.sub(r"\1REDACTED", raw)


TOURNAMENT_ALIASES: dict[str, set[str]] = {
    "ipl": {"ipl", "indianpremierleague", "tataipl"},
    "icc_t20_wc": {"icct20wc", "iccmensworldcup", "worldcupt20", "t20worldcup"},
}


def _tournament_matches(filter_value: Optional[str], tournament_value: Optional[str]) -> bool:
    if not filter_value:
        return True
    filter_key = _normalize_key(filter_value)
    tournament_key = _normalize_key(tournament_value)
    if not filter_key or not tournament_key:
        return False

    aliases = TOURNAMENT_ALIASES.get(filter_key, {filter_key})
    for alias in aliases:
        if alias and (alias in tournament_key or tournament_key in alias):
            return True
    return False


def _parse_match_state(
    status: str,
    match_started: Any = None,
    match_ended: Any = None,
    start_time: Optional[str] = None,
) -> str:
    match_ended_bool = _coerce_bool(match_ended)
    match_started_bool = _coerce_bool(match_started)
    text = _safe_lower(status)

    if match_ended_bool is True:
        return "historical"
    if match_started_bool is True:
        return "live"
    if match_started_bool is False:
        return "upcoming"
    if match_ended_bool is False:
        if any(
            token in text
            for token in ["live", "in progress", "innings", "stumps", "day ", "chasing", "break", "toss"]
        ):
            return "live"
        if start_time:
            try:
                dt = datetime.fromisoformat(str(start_time).replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if dt > now:
                    return "upcoming"
            except ValueError:
                pass
        return "upcoming"

    if any(
        token in text
        for token in [
            "result",
            "won by",
            "completed",
            "match over",
            "abandoned",
            "cancelled",
            "drawn",
            "no result",
        ]
    ):
        return "historical"
    if any(
        token in text
        for token in [
            "live",
            "in progress",
            "innings",
            "stumps",
            "day ",
            "chasing",
            "break",
            "toss",
        ]
    ):
        return "live"
    if any(
        token in text
        for token in [
            "upcoming",
            "fixture",
            "scheduled",
            "not started",
            "starts",
            "yet to begin",
            "starting",
            "match starts",
        ]
    ):
        return "upcoming"
    if start_time:
        try:
            dt = datetime.fromisoformat(str(start_time).replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt > now:
                return "upcoming"
        except ValueError:
            pass
    return "historical"


def _is_placeholder_team_name(value: str) -> bool:
    key = _normalize_key(value)
    if not key:
        return True
    bad = {
        "tbc",
        "tbd",
        "to be confirmed",
        "to be decided",
        "unknown",
        "na",
        "n/a",
    }
    lowered = value.strip().lower()
    if lowered in bad:
        return True
    return key in {"tbc", "tbd", "tobeconfirmed", "tobedecided", "unknown", "na"}


class CricAPIRealtimeProvider(RealtimeContextProvider):
    provider_name = "cricapi"

    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout_seconds: float,
        max_retries: int,
        backoff_seconds: float,
        cache_ttl_seconds: int,
        stale_cache_ttl_seconds: int,
        cache_store: JSONFileCacheStore,
    ) -> None:
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.max_retries = max(0, int(max_retries))
        self.backoff_seconds = max(0.0, float(backoff_seconds))
        self.cache_ttl_seconds = max(30, int(cache_ttl_seconds))
        self.stale_cache_ttl_seconds = max(self.cache_ttl_seconds, int(stale_cache_ttl_seconds))
        self.cache_store = cache_store
        if not self.api_key:
            logger.warning(
                "LIVE_CRICKET_API_KEY is not configured. CricAPI provider will only use cache or return unavailable."
            )

    def _cache_hit_payload(self, cache_hit: CacheHit, source: str) -> dict[str, Any]:
        return {
            "source": source,
            "updated_at": cache_hit.updated_at,
            "freshness_seconds": int(round(cache_hit.age_seconds)),
            "payload": cache_hit.payload,
        }

    def _request_json(self, endpoint: str, params: dict[str, Any], cache_key: str) -> dict[str, Any]:
        fresh_hit = self.cache_store.get(cache_key, max_age_seconds=self.cache_ttl_seconds)
        if fresh_hit is not None:
            return self._cache_hit_payload(fresh_hit, source="cache")

        if not self.api_key:
            stale_hit = self.cache_store.get(cache_key, max_age_seconds=self.stale_cache_ttl_seconds)
            if stale_hit is not None:
                return self._cache_hit_payload(stale_hit, source="cache-stale")
            return {
                "source": "unavailable",
                "updated_at": None,
                "freshness_seconds": None,
                "payload": {},
                "error": "missing_api_key",
            }

        request_params = dict(params)
        request_params["apikey"] = self.api_key
        url = f"{self.base_url}/{endpoint.lstrip('/')}?{urlencode(request_params)}"

        last_error: str | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urlopen(url, timeout=self.timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                logger.debug("CricAPI response ok endpoint=%s keys=%s", endpoint, list(payload.keys()) if isinstance(payload, dict) else type(payload).__name__)
                if not isinstance(payload, (dict, list)):
                    raise ValueError("unexpected_payload_shape")
                if self._response_contains_error(payload):
                    raise ValueError(self._extract_error_message(payload))
                cache_hit = self.cache_store.set(cache_key, payload)
                return self._cache_hit_payload(cache_hit, source="live")
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
                last_error = _redact_sensitive(exc)
                logger.warning(
                    "CricAPI request failed endpoint=%s attempt=%s/%s error=%s",
                    endpoint,
                    attempt + 1,
                    self.max_retries + 1,
                    last_error or exc.__class__.__name__,
                )
                if attempt < self.max_retries:
                    sleep_seconds = self.backoff_seconds * (2**attempt)
                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)

        stale_hit = self.cache_store.get(cache_key, max_age_seconds=self.stale_cache_ttl_seconds)
        if stale_hit is not None:
            result = self._cache_hit_payload(stale_hit, source="cache-stale")
            result["error"] = last_error
            return result
        return {
            "source": "unavailable",
            "updated_at": None,
            "freshness_seconds": None,
            "payload": {},
            "error": last_error,
        }

    @staticmethod
    def _response_contains_error(payload: dict[str, Any] | list[Any]) -> bool:
        if isinstance(payload, list):
            return False
        status = payload.get("status")
        if isinstance(status, bool):
            if status:
                return False
            return True
        if isinstance(status, str):
            lowered = status.strip().lower()
            if lowered in {"success", "ok", "true"}:
                return False
            if lowered in {"error", "failed", "failure", "unauthorized", "false"}:
                return True
        message = str(payload.get("message") or payload.get("msg") or payload.get("reason") or "").lower()
        if any(token in message for token in ["invalid api", "unauthorized", "forbidden", "rate limit", "error"]):
            return True
        return False

    @staticmethod
    def _extract_error_message(payload: dict[str, Any] | list[Any]) -> str:
        if isinstance(payload, list):
            return "live_api_error"
        text = str(payload.get("message") or payload.get("msg") or payload.get("reason") or "").strip()
        return text or "live_api_error"

    @staticmethod
    def _extract_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
        root_payload = payload.get("payload") or {}

        def _extract(container: Any) -> list[dict[str, Any]]:
            if isinstance(container, list):
                return [row for row in container if isinstance(row, dict)]
            if isinstance(container, dict):
                for key in ("data", "matches", "matchList", "items", "response", "results"):
                    value = container.get(key)
                    if isinstance(value, list):
                        return [row for row in value if isinstance(row, dict)]
                    if isinstance(value, dict):
                        nested = _extract(value)
                        if nested:
                            return nested
            return []

        rows = _extract(root_payload)
        if rows:
            return rows
        if isinstance(root_payload, dict):
            dict_rows = [value for value in root_payload.values() if isinstance(value, dict)]
            if dict_rows:
                return dict_rows
        return []

    @staticmethod
    def _payload_rank(source: str) -> int:
        key = str(source or "").strip().lower()
        if key == "live":
            return 4
        if key == "cache":
            return 3
        if key == "cache-stale":
            return 2
        return 1

    @staticmethod
    def _merge_payload_meta(entries: list[dict[str, Any]]) -> dict[str, Any]:
        if not entries:
            return {"source": "unavailable", "updated_at": None, "freshness_seconds": None, "payload": {}}
        best = max(entries, key=lambda row: CricAPIRealtimeProvider._payload_rank(str(row.get("source") or "")))
        freshness_values = [
            int(float(row.get("freshness_seconds")))
            for row in entries
            if row.get("freshness_seconds") is not None
            and str(row.get("freshness_seconds")).strip() not in {"", "nan"}
        ]
        freshest = min(freshness_values) if freshness_values else None
        updated_candidates: list[datetime] = []
        for row in entries:
            raw = str(row.get("updated_at") or "").strip()
            if not raw:
                continue
            parsed = raw.replace("Z", "+00:00")
            try:
                updated_candidates.append(datetime.fromisoformat(parsed))
            except ValueError:
                continue
        latest = max(updated_candidates).astimezone(timezone.utc).isoformat() if updated_candidates else best.get("updated_at")
        merged = {
            "source": str(best.get("source") or "unavailable"),
            "updated_at": latest,
            "freshness_seconds": freshest,
            "payload": best.get("payload") if isinstance(best.get("payload"), (dict, list)) else {},
        }
        errors = [str(row.get("error") or "").strip() for row in entries if str(row.get("error") or "").strip()]
        if errors:
            merged["error"] = "; ".join(dict.fromkeys(errors))
        return merged

    @staticmethod
    def _normalize_tournament(raw: dict[str, Any]) -> str:
        series = raw.get("series")
        if isinstance(series, dict):
            name = str(series.get("name") or "").strip()
            if name:
                return name
        if isinstance(series, str) and series.strip():
            return series.strip()
        tournament = str(raw.get("tournament") or "").strip()
        return tournament

    @staticmethod
    def _normalize_start_time(raw: dict[str, Any]) -> Optional[str]:
        for key in ("dateTimeGMT", "dateTime", "date", "matchDate", "match_time"):
            value = raw.get(key)
            if isinstance(value, (int, float)):
                try:
                    as_float = float(value)
                    if as_float > 0:
                        seconds = as_float / 1000.0 if as_float > 1e12 else as_float
                        dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
                        return dt.isoformat()
                except (TypeError, ValueError, OSError):
                    pass
            text = str(value or "").strip()
            if not text:
                continue
            if text.isdigit():
                try:
                    as_float = float(text)
                    seconds = as_float / 1000.0 if as_float > 1e12 else as_float
                    dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
                    return dt.isoformat()
                except (TypeError, ValueError, OSError):
                    pass
            parsed = text.replace("Z", "+00:00")
            try:
                dt = datetime.fromisoformat(parsed)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc).isoformat()
            except ValueError:
                if len(text) >= 10:
                    return f"{text[:10]}T00:00:00+00:00"
        return None

    @staticmethod
    def _resolve_teams(raw: dict[str, Any]) -> tuple[str, str]:
        def _extract_team_name(value: Any) -> str:
            if isinstance(value, dict):
                return str(value.get("name") or value.get("teamName") or value.get("shortname") or "").strip()
            return str(value or "").strip()

        def _from_title() -> tuple[str, str]:
            title = str(raw.get("name") or raw.get("matchTitle") or raw.get("title") or "").strip()
            if " vs " in title.lower():
                split = re.split(r"\s+vs\s+", title, maxsplit=1, flags=re.IGNORECASE)
                if len(split) == 2:
                    return split[0].strip(), split[1].split(",")[0].strip()
            if " v " in title.lower():
                split = re.split(r"\s+v\s+", title, maxsplit=1, flags=re.IGNORECASE)
                if len(split) == 2:
                    return split[0].strip(), split[1].split(",")[0].strip()
            return "", ""

        teams = raw.get("teams") if isinstance(raw.get("teams"), list) else []
        if len(teams) >= 2:
            team_a = _extract_team_name(teams[0])
            team_b = _extract_team_name(teams[1])
            if team_a and team_b:
                return team_a, team_b

        info = raw.get("teamInfo") if isinstance(raw.get("teamInfo"), list) else []
        team_a = ""
        team_b = ""
        if len(info) >= 1 and isinstance(info[0], dict):
            team_a = str(info[0].get("name") or "").strip()
        if len(info) >= 2 and isinstance(info[1], dict):
            team_b = str(info[1].get("name") or "").strip()
        if team_a and team_b:
            return team_a, team_b

        for key_a, key_b in (
            ("team1", "team2"),
            ("teamA", "teamB"),
            ("homeTeam", "awayTeam"),
            ("home", "away"),
        ):
            direct_a = _extract_team_name(raw.get(key_a))
            direct_b = _extract_team_name(raw.get(key_b))
            if direct_a and direct_b:
                return direct_a, direct_b

        title_a, title_b = _from_title()
        if title_a and title_b:
            return title_a, title_b
        return team_a, team_b

    @staticmethod
    def _normalize_venue(raw: dict[str, Any]) -> str:
        for key in ("venue", "ground", "stadium", "venueName", "city"):
            text = str(raw.get(key) or "").strip()
            if text:
                return text
        return "Venue TBA"

    @staticmethod
    def _row_has_valid_teams(team_a: str, team_b: str) -> bool:
        if _is_placeholder_team_name(team_a) or _is_placeholder_team_name(team_b):
            return False
        return _normalize_key(team_a) != _normalize_key(team_b)

    @staticmethod
    def _row_key(row: dict[str, Any]) -> str:
        match_id = str(row.get("match_id") or "").strip()
        if match_id:
            return f"id:{match_id}"
        team_a = _normalize_key(row.get("team_a"))
        team_b = _normalize_key(row.get("team_b"))
        date_key = str(row.get("match_date") or row.get("start_time") or "").strip()
        if team_a and team_b and date_key:
            pair = "|".join(sorted([team_a, team_b]))
            return f"pair:{pair}:{date_key}"
        return ""

    def _dedupe_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        state_priority = {"live": 3, "upcoming": 2, "historical": 1}
        merged: dict[str, dict[str, Any]] = {}
        for row in rows:
            key = self._row_key(row)
            if not key:
                continue
            source = str(row.get("provider_source") or "")
            rank = self._payload_rank(source)
            existing = merged.get(key)
            if existing is None:
                merged[key] = dict(row)
                continue
            existing_rank = self._payload_rank(str(existing.get("provider_source") or ""))
            if rank > existing_rank:
                merged[key] = dict(row)
                continue
            if rank == existing_rank:
                existing_state = str(existing.get("state") or "")
                candidate_state = str(row.get("state") or "")
                if state_priority.get(candidate_state, 0) > state_priority.get(existing_state, 0):
                    merged[key] = dict(row)
                    continue
                existing_start = str(existing.get("start_time") or "")
                candidate_start = str(row.get("start_time") or "")
                if candidate_start and not existing_start:
                    merged[key] = dict(row)
        output = list(merged.values())
        output.sort(key=lambda row: (str(row.get("start_time") or ""), str(row.get("match_id") or "")))
        return output

    def _to_match_row(self, raw: dict[str, Any], payload_meta: dict[str, Any]) -> dict[str, Any]:
        team_a, team_b = self._resolve_teams(raw)
        match_started_raw = (
            raw.get("matchStarted")
            if raw.get("matchStarted") is not None
            else raw.get("isStarted")
        )
        match_ended_raw = (
            raw.get("matchEnded")
            if raw.get("matchEnded") is not None
            else raw.get("isEnded")
        )
        status_text = str(
            raw.get("status")
            or raw.get("matchStatus")
            or raw.get("ms")
            or raw.get("matchState")
            or ""
        ).strip()
        start_time = self._normalize_start_time(raw)
        match_id = str(
            raw.get("id")
            or raw.get("matchId")
            or raw.get("unique_id")
            or raw.get("match_id")
            or raw.get("name")
            or f"{_normalize_key(team_a)}_{_normalize_key(team_b)}_{_normalize_key(start_time)}"
        ).strip()
        score_payload = raw.get("score")
        score_rows = score_payload if isinstance(score_payload, list) else []
        tournament = self._normalize_tournament(raw)
        venue = self._normalize_venue(raw)
        match_date = start_time[:10] if start_time else None
        return {
            "match_id": match_id,
            "sport": "cricket",
            "tournament": tournament,
            "team_a": team_a,
            "team_b": team_b,
            "venue": venue,
            "match_date": match_date,
            "start_time": start_time,
            "state": _parse_match_state(
                status_text,
                match_started=match_started_raw,
                match_ended=match_ended_raw,
                start_time=start_time,
            ),
            "status_text": status_text,
            "_match_started": match_started_raw,
            "_match_ended": match_ended_raw,
            "score": score_rows,
            "provider": self.provider_name,
            "provider_source": payload_meta.get("source"),
            "provider_updated_at": payload_meta.get("updated_at"),
            "freshness_seconds": payload_meta.get("freshness_seconds"),
            "data_source": "live-provider",
        }

    def _rows_from_endpoint(
        self,
        *,
        endpoint: str,
        params: dict[str, Any],
        cache_key: str,
        tournament: Optional[str],
        allowed_states: Optional[set[str]] = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        payload_meta = self._request_json(endpoint=endpoint, params=params, cache_key=cache_key)
        raw_rows = self._extract_rows(payload_meta)
        if endpoint.lower().endswith("matches") and not raw_rows:
            logger.info("Provider returned 0 matches")
        all_rows = [self._to_match_row(raw, payload_meta) for raw in raw_rows]
        rows: list[dict[str, Any]] = []
        rejected_counts: dict[str, int] = {
            "missing_match_id": 0,
            "invalid_teams": 0,
            "tournament_mismatch": 0,
            "state_filtered": 0,
        }

        def _is_future(start_iso: str) -> bool:
            if not start_iso:
                return False
            try:
                dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
            except ValueError:
                return False
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt > datetime.now(timezone.utc)

        for row in all_rows:
            if not str(row.get("match_id") or "").strip():
                rejected_counts["missing_match_id"] += 1
                continue
            team_a = str(row.get("team_a") or "").strip()
            team_b = str(row.get("team_b") or "").strip()
            if not self._row_has_valid_teams(team_a, team_b):
                rejected_counts["invalid_teams"] += 1
                continue
            row_tournament = str(row.get("tournament") or "").strip()
            if not row_tournament:
                row["tournament"] = str(tournament or "Unknown League").strip() or "Unknown League"
            if tournament and not _tournament_matches(tournament, row.get("tournament")):
                rejected_counts["tournament_mismatch"] += 1
                continue
            if allowed_states is not None:
                row_state = str(row.get("state") or "").strip().lower()
                if row_state not in allowed_states:
                    status_text = str(row.get("status_text") or "").strip().lower()
                    start_time = str(row.get("start_time") or "").strip()
                    started = _coerce_bool(row.get("_match_started"))
                    ended = _coerce_bool(row.get("_match_ended"))
                    live_candidate = bool(
                        started is True
                        or any(
                            token in status_text
                            for token in ["live", "in progress", "innings", "stumps", "day ", "chasing", "break", "toss"]
                        )
                    )
                    upcoming_candidate = bool(
                        started is False
                        or ended is False
                        or any(
                            token in status_text
                            for token in ["match not started", "scheduled", "upcoming", "fixture", "yet to begin", "starting", "starts"]
                        )
                        or status_text == ""
                        or _is_future(start_time)
                    )
                    if "live" in allowed_states and live_candidate and ended is not True:
                        row["state"] = "live"
                    elif "upcoming" in allowed_states and upcoming_candidate and not live_candidate:
                        row["state"] = "upcoming"
                    else:
                        rejected_counts["state_filtered"] += 1
                        continue
            row.pop("_match_started", None)
            row.pop("_match_ended", None)
            rows.append(row)
        rows = self._dedupe_rows(rows)
        if not rows:
            logger.info("Provider returned 0 valid matches")
        if raw_rows:
            retained_ratio = float(len(rows)) / float(len(raw_rows))
            if retained_ratio < 0.2:
                logger.warning(
                    "CricAPI filtering retained only %.1f%% rows endpoint=%s state_filter=%s",
                    retained_ratio * 100.0,
                    endpoint,
                    ",".join(sorted(allowed_states)) if allowed_states else "all",
                )
        logger.info(
            "CricAPI rows endpoint=%s source=%s raw=%s accepted=%s rejected=%s state_filter=%s tournament=%s",
            endpoint,
            str(payload_meta.get("source") or "unavailable"),
            len(raw_rows),
            len(rows),
            max(0, len(raw_rows) - len(rows)),
            ",".join(sorted(allowed_states)) if allowed_states else "all",
            tournament or "all",
        )
        if any(rejected_counts.values()):
            logger.info(
                "CricAPI rejection breakdown endpoint=%s %s",
                endpoint,
                ", ".join(f"{key}={value}" for key, value in rejected_counts.items() if value > 0),
            )
        return rows, payload_meta

    def _context_rows_from_api(self, tournament: Optional[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        matches_rows, matches_meta = self._rows_from_endpoint(
            endpoint="matches",
            params={"offset": 0},
            cache_key="cricapi_matches",
            tournament=tournament,
            allowed_states=None,
        )
        live_rows, live_meta = self._rows_from_endpoint(
            endpoint="currentMatches",
            params={"offset": 0},
            cache_key="cricapi_current_matches",
            tournament=tournament,
            allowed_states={"live"},
        )
        merged_rows = self._dedupe_rows(matches_rows + live_rows)
        merged_meta = self._merge_payload_meta([matches_meta, live_meta])
        return merged_rows, merged_meta

    def fetch_live_matches(self, tournament: Optional[str] = None, limit: int = 20) -> list[dict[str, Any]]:
        rows, _ = self._rows_from_endpoint(
            endpoint="currentMatches",
            params={"offset": 0},
            cache_key="cricapi_current_matches",
            tournament=tournament,
            allowed_states={"live"},
        )
        return rows[: max(1, limit)]

    def fetch_upcoming_matches(self, tournament: Optional[str] = None, limit: int = 20) -> list[dict[str, Any]]:
        rows, _ = self._rows_from_endpoint(
            endpoint="matches",
            params={"offset": 0},
            cache_key="cricapi_matches",
            tournament=tournament,
            allowed_states={"upcoming"},
        )
        return rows[: max(1, limit)]

    @staticmethod
    def _find_row(rows: list[dict[str, Any]], match_id: Optional[str], team_a: Optional[str], team_b: Optional[str]) -> dict[str, Any] | None:
        if match_id:
            for row in rows:
                if str(row.get("match_id")) == str(match_id):
                    return row

        a = _safe_lower(team_a)
        b = _safe_lower(team_b)
        if a and b:
            for row in rows:
                row_a = _safe_lower(row.get("team_a"))
                row_b = _safe_lower(row.get("team_b"))
                if (row_a == a and row_b == b) or (row_a == b and row_b == a):
                    return row
        return None

    def _context_features_from_row(self, row: dict[str, Any]) -> dict[str, float]:
        score_rows = row.get("score") if isinstance(row.get("score"), list) else []
        first_score = score_rows[0] if score_rows else {}

        runs = _safe_float(first_score.get("r"), 168.0)
        wickets = _safe_float(first_score.get("w"), 6.0)
        overs = max(1.0, _safe_float(first_score.get("o"), 20.0))
        run_rate = runs / overs

        batting_form_runs = max(110.0, min(250.0, runs + (run_rate * 2.4)))
        bowling_conceded = max(110.0, min(245.0, runs + 3.0))
        batting_wickets = max(2.5, min(10.0, wickets + 0.8))
        bowling_wickets = max(2.0, min(10.0, wickets + 0.5))

        momentum = max(-1.0, min(1.0, (run_rate - 8.0) / 3.0))
        batting_win_rate = max(0.2, min(0.82, 0.5 + momentum * 0.18))
        bowling_win_rate = max(0.18, min(0.8, 1.0 - batting_win_rate))

        team_a_avg_runs_last_5 = round(batting_form_runs, 3)
        team_a_avg_runs_last_10 = round(max(95.0, batting_form_runs - 4.5), 3)
        team_b_avg_runs_last_5 = round(max(95.0, bowling_conceded - 2.0), 3)
        team_b_avg_runs_last_10 = round(max(90.0, bowling_conceded - 6.0), 3)
        team_a_chase_success = max(0.1, min(0.9, 0.46 + (momentum * 0.1)))
        team_b_chase_success = max(0.1, min(0.9, 0.54 - (momentum * 0.1)))
        team_a_defend_success = max(0.1, min(0.9, 1.0 - team_b_chase_success))
        team_b_defend_success = max(0.1, min(0.9, 1.0 - team_a_chase_success))
        recent_form_diff = team_a_avg_runs_last_5 - team_b_avg_runs_last_5
        recent_run_rate_diff = (team_a_avg_runs_last_5 - team_a_avg_runs_last_10) - (
            team_b_avg_runs_last_5 - team_b_avg_runs_last_10
        )
        head_to_head_win_diff = (2.0 * batting_win_rate) - 1.0
        venue_batting_adv = max(-1.0, min(1.0, (batting_win_rate - 0.5) * 1.7))
        return {
            "team_a_avg_runs_last_5": team_a_avg_runs_last_5,
            "team_a_avg_runs_last_10": team_a_avg_runs_last_10,
            "team_a_avg_wickets_last_5": round(batting_wickets, 3),
            "team_a_run_rate_trend": round(team_a_avg_runs_last_5 - team_a_avg_runs_last_10, 3),
            "team_b_avg_runs_last_5": team_b_avg_runs_last_5,
            "team_b_avg_runs_last_10": team_b_avg_runs_last_10,
            "team_b_avg_wickets_last_5": round(bowling_wickets, 3),
            "team_b_run_rate_trend": round(team_b_avg_runs_last_5 - team_b_avg_runs_last_10, 3),
            "team_a_win_rate_vs_b": round(batting_win_rate, 3),
            "avg_score_team_a_vs_b": round(max(120.0, min(235.0, runs * 0.95 + 8.0)), 3),
            "avg_score_team_b_vs_a": round(max(115.0, min(230.0, runs * 0.92 + 5.0)), 3),
            "venue_avg_score": round(max(110.0, min(235.0, (runs + (runs - 2.0)) / 2.0)), 3),
            "venue_chase_success_rate": round(max(0.1, min(0.9, bowling_win_rate)), 3),
            "venue_defend_bias": round(-venue_batting_adv, 3),
            "team_a_runs_vs_opponent_avg": round(team_a_avg_runs_last_5 - team_b_avg_runs_last_5, 3),
            "team_b_runs_vs_opponent_avg": round(team_b_avg_runs_last_5 - team_a_avg_runs_last_5, 3),
            "batting_first": 1.0,
            "team_a_bats_first": 1.0,
            "team_b_bats_first": 0.0,
            "team_a_chase_success_rate": round(team_a_chase_success, 3),
            "team_b_chase_success_rate": round(team_b_chase_success, 3),
            "team_a_defend_success_rate": round(team_a_defend_success, 3),
            "team_b_defend_success_rate": round(team_b_defend_success, 3),
            "chase_defend_edge_team_a_first": round(team_b_chase_success - team_a_defend_success, 3),
            "chase_defend_edge_team_b_first": round(team_a_chase_success - team_b_defend_success, 3),
            "venue_batting_first_advantage": round(venue_batting_adv, 3),
            "recent_form_diff": round(recent_form_diff, 3),
            "recent_run_rate_diff": round(recent_run_rate_diff, 3),
            "head_to_head_win_diff": round(head_to_head_win_diff, 3),
            "wickets_taken_diff": round(bowling_wickets - batting_wickets, 3),
        }

    def fetch_match_context(
        self,
        match_id: Optional[str] = None,
        team_a: Optional[str] = None,
        team_b: Optional[str] = None,
        match_date: Optional[str] = None,
        tournament: Optional[str] = None,
    ) -> dict[str, Any]:
        rows, payload_meta = self._context_rows_from_api(tournament=tournament)
        match = self._find_row(rows, match_id=match_id, team_a=team_a, team_b=team_b)
        if match is None:
            return {
                "provider": self.provider_name,
                "source": payload_meta.get("source", "unavailable"),
                "updated_at": payload_meta.get("updated_at"),
                "freshness_seconds": payload_meta.get("freshness_seconds"),
                "features": {},
                "reason": "match_not_found_in_live_feed",
            }

        match_details = dict(match)
        features = self._context_features_from_row(match_details)
        return {
            "provider": self.provider_name,
            "source": payload_meta.get("source", "live"),
            "updated_at": payload_meta.get("updated_at"),
            "freshness_seconds": payload_meta.get("freshness_seconds"),
            "match_id": match_details.get("match_id"),
            "status_text": match_details.get("status_text"),
            "state": match_details.get("state"),
            "features": features,
            "summary": "Live feed merged into historical baseline.",
        }

    def fetch_recent_team_stats(self, team: str, limit: int = 5) -> dict[str, Any]:
        rows, payload_meta = self._context_rows_from_api(tournament=None)
        selected = [
            row
            for row in rows
            if team.strip().lower() in {_safe_lower(row.get("team_a")), _safe_lower(row.get("team_b"))}
        ][: max(1, limit)]
        if not selected:
            return {
                "provider": self.provider_name,
                "team": team,
                "source": payload_meta.get("source"),
                "matches_used": 0,
            }

        inferred_runs = []
        for row in selected:
            score = row.get("score")
            if isinstance(score, list) and score:
                inferred_runs.append(_safe_float(score[0].get("r"), 165.0))
        avg_runs = sum(inferred_runs) / len(inferred_runs) if inferred_runs else 165.0
        return {
            "provider": self.provider_name,
            "team": team,
            "source": payload_meta.get("source"),
            "matches_used": len(selected),
            "avg_runs_signal": round(avg_runs, 3),
        }

    def fetch_recent_player_stats(self, player: str, limit: int = 5) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "player": player,
            "source": "unavailable",
            "matches_used": 0,
            "note": "Player-level feed endpoint is not configured in this provider path yet.",
        }

    def enrich_match_context(self, match_row: dict[str, Any]) -> dict[str, Any]:
        context = self.fetch_match_context(
            match_id=match_row.get("match_id"),
            team_a=match_row.get("team_a"),
            team_b=match_row.get("team_b"),
            match_date=match_row.get("match_date"),
            tournament=match_row.get("tournament"),
        )

        features = context.get("features") or {}
        batting_index = _safe_float(features.get("batting_strength_index"), 30.0)
        bowling_index = _safe_float(features.get("bowling_strength_index"), 26.0)
        pressure_index = max(0.2, min(0.9, 0.45 + ((bowling_index - batting_index) / 80.0)))
        tempo_shift = max(-0.2, min(0.25, (batting_index - 30.0) / 110.0))
        injury_risk = max(0.02, min(0.22, 0.11 + (pressure_index - 0.5) * 0.1))

        return {
            "provider": self.provider_name,
            "updated_at": context.get("updated_at") or _utc_now_iso(),
            "tempo_shift": round(tempo_shift, 3),
            "pressure_index": round(pressure_index, 3),
            "injury_risk": round(injury_risk, 3),
            "source": context.get("source"),
            "freshness_seconds": context.get("freshness_seconds"),
        }

    def healthcheck(self) -> dict[str, Any]:
        matches_payload = self._request_json(
            endpoint="matches",
            params={"offset": 0},
            cache_key="cricapi_healthcheck",
        )
        live_overlay_payload = self._request_json(
            endpoint="currentMatches",
            params={"offset": 0},
            cache_key="cricapi_live_overlay_healthcheck",
        )
        source = str(matches_payload.get("source") or "unavailable")
        cache_stats = self.cache_store.stats()
        if source == "live":
            status = "ok"
        elif source.startswith("cache"):
            status = "degraded"
        else:
            status = "unavailable"
        return {
            "provider": self.provider_name,
            "status": status,
            "source": source,
            "updated_at": matches_payload.get("updated_at"),
            "freshness_seconds": matches_payload.get("freshness_seconds"),
            "healthy": status in {"ok", "degraded"},
            "live_overlay_source": str(live_overlay_payload.get("source") or "unavailable"),
            "live_overlay_updated_at": live_overlay_payload.get("updated_at"),
            "cache_entries": cache_stats.get("entries", 0),
            "cache_latest_updated_at": cache_stats.get("latest_updated_at"),
            "api_key_configured": bool(self.api_key),
        }
