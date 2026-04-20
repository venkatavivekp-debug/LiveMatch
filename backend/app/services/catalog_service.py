from __future__ import annotations

import logging
import re
from datetime import date
from functools import lru_cache
from typing import Any, Optional

import pandas as pd

from app.core.config import get_settings
from app.schemas.common import HeadToHeadMatchResponse, MatchResponse, SportResponse, TournamentResponse
from app.services.providers.factory import get_catalog_provider, get_live_provider

logger = logging.getLogger(__name__)


class CatalogService:
    @staticmethod
    def _norm(value: Optional[str]) -> str:
        return str(value or "").strip().lower()

    @staticmethod
    def _norm_key(value: Optional[str]) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())

    @classmethod
    def _tournament_match(cls, filter_value: Optional[str], tournament_value: Optional[str]) -> bool:
        if not filter_value:
            return True
        filter_key = cls._norm_key(filter_value)
        tournament_key = cls._norm_key(tournament_value)
        if not filter_key or not tournament_key:
            return False
        aliases = {
            "ipl": {"ipl", "indianpremierleague", "tataipl"},
            "icc_t20_wc": {"icct20wc", "iccmensworldcup", "t20worldcup", "worldcupt20"},
        }.get(filter_key, {filter_key})
        return any(alias and (alias in tournament_key or tournament_key in alias) for alias in aliases)

    @staticmethod
    @lru_cache(maxsize=1)
    def _cricket_results_lookup() -> dict[str, dict[str, Any]]:
        settings = get_settings()
        path = settings.data_processed_dir / "matches.csv"
        if not path.exists():
            return {}
        try:
            frame = pd.read_csv(path)
        except Exception:  # noqa: BLE001
            return {}
        if "match_id" not in frame.columns:
            return {}
        rows: dict[str, dict[str, Any]] = {}
        for _, row in frame.iterrows():
            match_id = str(row.get("match_id", "")).strip()
            if not match_id:
                continue
            rows[match_id] = row.to_dict()
        return rows

    @staticmethod
    @lru_cache(maxsize=1)
    def _football_results_lookup() -> dict[str, dict[str, Any]]:
        settings = get_settings()
        path = settings.data_processed_dir / "football_matches.csv"
        if not path.exists():
            return {}
        try:
            frame = pd.read_csv(path)
        except Exception:  # noqa: BLE001
            return {}
        if "match_id" not in frame.columns:
            return {}
        rows: dict[str, dict[str, Any]] = {}
        for _, row in frame.iterrows():
            match_id = str(row.get("match_id", "")).strip()
            if not match_id:
                continue
            rows[match_id] = row.to_dict()
        return rows

    @classmethod
    def _merge_live_rows(
        cls,
        rows: list[dict[str, Any]],
        sport: Optional[str],
        tournament: Optional[str],
        state: Optional[str],
        team: Optional[str],
        venue: Optional[str],
    ) -> list[dict[str, Any]]:
        sport_filter = (sport or "").strip().lower()
        if sport_filter and sport_filter != "cricket":
            return rows

        normalized_base_rows = []
        for row in rows:
            item = dict(row)
            item.setdefault("data_source", "historical")
            normalized_base_rows.append(item)

        state_filter = (state or "").strip().lower()
        if state_filter == "historical":
            return normalized_base_rows

        fetch_live = state_filter in {"", "live"}
        fetch_upcoming = state_filter in {"", "upcoming"}
        if not fetch_live and not fetch_upcoming:
            return normalized_base_rows

        live_provider = get_live_provider()
        live_rows: list[dict[str, Any]] = []
        provider_status: dict[str, Any] = {}
        provider_failed = False
        try:
            if state_filter == "upcoming":
                live_rows = live_provider.fetch_upcoming_matches(tournament=tournament, limit=30)
            elif state_filter == "live":
                live_rows = live_provider.fetch_live_matches(tournament=tournament, limit=30)
            else:
                if fetch_upcoming:
                    live_rows.extend(live_provider.fetch_upcoming_matches(tournament=tournament, limit=30))
                if fetch_live:
                    live_rows.extend(live_provider.fetch_live_matches(tournament=tournament, limit=30))
            provider_status = live_provider.healthcheck()
        except Exception:  # noqa: BLE001
            provider_failed = True
            provider_status = {"status": "unavailable"}

        if provider_failed and not live_rows:
            provider_status = {"status": "unavailable"}
        logger.info(
            "Catalog merge live rows state=%s provider=%s status=%s rows=%s failed=%s",
            state_filter or "all",
            str(provider_status.get("provider") or getattr(live_provider, "provider_name", "unknown")),
            str(provider_status.get("status") or "unknown"),
            len(live_rows),
            provider_failed,
        )
        if not live_rows:
            if state_filter in {"live", "upcoming"}:
                return []
            return normalized_base_rows

        merged: dict[str, dict[str, Any]] = {}
        if state_filter not in {"live", "upcoming"}:
            merged = {
                str(row["match_id"]): dict(row)
                for row in normalized_base_rows
                if row.get("match_id")
            }
        settings = get_settings()
        mock_allowed = settings.realtime_provider.strip().lower() in {"mock-realtime", "mock"}
        for row in live_rows:
            provider_source = str(row.get("provider_source") or row.get("source") or "").strip().lower()
            provider_name = str(row.get("provider") or "").strip().lower()
            if not mock_allowed and ("mock" in provider_source or "mock" in provider_name):
                continue
            if not str(row.get("match_id") or "").strip():
                continue
            if not str(row.get("team_a") or "").strip() or not str(row.get("team_b") or "").strip():
                continue
            if team:
                lookup = team.lower()
                if lookup not in str(row.get("team_a", "")).lower() and lookup not in str(row.get("team_b", "")).lower():
                    continue
            row = dict(row)
            row["venue"] = str(row.get("venue") or "Venue TBA").strip() or "Venue TBA"
            row["tournament"] = str(row.get("tournament") or tournament or "IPL").strip() or (tournament or "IPL")

            if venue and venue.lower() not in str(row.get("venue", "")).lower():
                continue
            if tournament and not cls._tournament_match(tournament, str(row.get("tournament", ""))):
                continue
            row_state = str(row.get("state", "")).lower()
            if state_filter in {"live", "upcoming"} and row_state != state_filter:
                continue
            if state_filter not in {"live", "upcoming"} and state_filter and state_filter not in row_state:
                continue

            match_id = str(row.get("match_id"))
            if not match_id:
                continue
            row.setdefault("data_source", "live-provider")
            merged[match_id] = row

        output = list(merged.values())
        output.sort(key=lambda row: (str(row.get("match_date") or ""), str(row.get("state") or "")), reverse=True)
        return output

    @staticmethod
    def list_sports() -> list[SportResponse]:
        rows = get_catalog_provider().list_sports()
        return [SportResponse(**row) for row in rows]

    @staticmethod
    def list_tournaments(sport: Optional[str] = None) -> list[TournamentResponse]:
        rows = get_catalog_provider().list_tournaments(sport=sport)
        return [TournamentResponse(**row) for row in rows]

    @classmethod
    def list_matches(
        cls,
        sport: Optional[str] = None,
        tournament: Optional[str] = None,
        team: Optional[str] = None,
        venue: Optional[str] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        state: Optional[str] = None,
    ) -> list[MatchResponse]:
        rows = get_catalog_provider().list_matches(
            sport=sport,
            tournament=tournament,
            team=team,
            venue=venue,
            date_from=date_from,
            date_to=date_to,
            state=state,
        )
        rows = cls._merge_live_rows(
            rows=rows,
            sport=sport,
            tournament=tournament,
            state=state,
            team=team,
            venue=venue,
        )
        return [MatchResponse(**row) for row in rows]

    @classmethod
    def get_match_by_id(cls, match_id: str) -> Optional[MatchResponse]:
        row = get_catalog_provider().get_match_by_id(match_id)
        if row is not None:
            payload = dict(row)
            payload.setdefault("data_source", "historical")
            return MatchResponse(**payload)

        for state in ("live", "upcoming"):
            rows = cls.list_matches(sport="cricket", state=state)
            for candidate in rows:
                if str(candidate.match_id) == str(match_id):
                    return candidate
        return None

    @staticmethod
    def top_players(
        sport: str,
        tournament: Optional[str] = None,
        team: Optional[str] = None,
        role: Optional[str] = None,
        limit: int = 5,
    ) -> list[dict]:
        return get_catalog_provider().get_top_players(
            sport=sport,
            tournament=tournament,
            team=team,
            role=role,
            limit=limit,
        )

    @staticmethod
    def active_provider_name() -> str:
        settings = get_settings()
        return settings.data_provider

    @classmethod
    def head_to_head_history(
        cls,
        *,
        sport: str,
        team_a: str,
        team_b: str,
        tournament: Optional[str] = None,
        limit: int = 5,
        include_evaluation: bool = True,
        k: int = 4,
    ) -> list[HeadToHeadMatchResponse]:
        sport_key = cls._norm(sport) or "cricket"
        left = cls._norm(team_a)
        right = cls._norm(team_b)
        if not left or not right:
            return []

        rows = cls.list_matches(
            sport=sport_key,
            tournament=tournament,
            state="historical",
        )
        selected: list[dict[str, Any]] = []
        for row in rows:
            payload = row.model_dump()
            row_left = cls._norm(payload.get("team_a"))
            row_right = cls._norm(payload.get("team_b"))
            if (row_left == left and row_right == right) or (row_left == right and row_right == left):
                selected.append(payload)
            if len(selected) >= max(1, min(limit, 15)):
                break

        cricket_lookup = cls._cricket_results_lookup() if sport_key == "cricket" else {}
        football_lookup = cls._football_results_lookup() if sport_key == "football" else {}

        history: list[HeadToHeadMatchResponse] = []
        for row in selected:
            match_id = str(row.get("match_id", ""))
            winner: Optional[str] = None
            actual_result: Optional[str] = None

            if sport_key == "cricket":
                raw = cricket_lookup.get(match_id) or {}
                first_team = str(raw.get("first_innings_team", "")).strip()
                second_team = str(raw.get("second_innings_team", "")).strip()
                first_total = raw.get("first_innings_total")
                second_total = raw.get("second_innings_total")
                winner = str(raw.get("winner", "")).strip() or None
                if first_team and second_team and pd.notna(first_total) and pd.notna(second_total):
                    actual_result = f"{first_team} {int(first_total)} vs {second_team} {int(second_total)}"
            else:
                raw = football_lookup.get(match_id) or {}
                home_goals = raw.get("home_goals")
                away_goals = raw.get("away_goals")
                if pd.notna(home_goals) and pd.notna(away_goals):
                    home = str(row.get("team_a") or "Home")
                    away = str(row.get("team_b") or "Away")
                    home_score = int(home_goals)
                    away_score = int(away_goals)
                    actual_result = f"{home} {home_score}-{away_score} {away}"
                    if home_score > away_score:
                        winner = home
                    elif away_score > home_score:
                        winner = away
                    else:
                        winner = "Draw"

            evaluation_payload = None
            if include_evaluation:
                from app.services.prediction_service import PredictionService

                try:
                    evaluated = PredictionService.evaluate_match(
                        match_payload={
                            "match_id": match_id,
                            "sport": row.get("sport"),
                            "tournament": row.get("tournament"),
                            "team_a": row.get("team_a"),
                            "team_b": row.get("team_b"),
                            "venue": row.get("venue"),
                            "match_date": str(row.get("match_date") or ""),
                            "state": "historical",
                        },
                        k=k,
                    )
                    prediction_rows = evaluated.get("prediction", {}).get("predictions", [])
                    predicted_heads: list[str] = []
                    for item in prediction_rows:
                        if isinstance(item, dict):
                            if item.get("score") is not None:
                                predicted_heads.append(str(item.get("score")))
                            elif item.get("scoreline") is not None:
                                predicted_heads.append(str(item.get("scoreline")))
                    eval_summary = evaluated.get("evaluation", {})
                    evaluation_payload = {
                        "predicted_heads": predicted_heads,
                        "best_match_to_actual": eval_summary.get("winner_scenario"),
                        "best_error": eval_summary.get("best_match_error"),
                        "in_range": eval_summary.get("interval_covered"),
                        "winner_correct": eval_summary.get("winner_correct"),
                    }
                except Exception:  # noqa: BLE001
                    evaluation_payload = None

            history.append(
                HeadToHeadMatchResponse(
                    match_id=match_id,
                    sport=str(row.get("sport") or sport_key),
                    tournament=str(row.get("tournament") or ""),
                    team_a=str(row.get("team_a") or ""),
                    team_b=str(row.get("team_b") or ""),
                    venue=str(row.get("venue") or ""),
                    match_date=row.get("match_date"),
                    winner=winner,
                    actual_result=actual_result,
                    evaluation=evaluation_payload,
                )
            )
        return history
