from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hashlib import md5
from typing import Any, Optional

from app.services.providers.base import RealtimeContextProvider


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MockRealtimeProvider(RealtimeContextProvider):
    """
    Deterministic local fallback provider.

    This keeps the backend stable when live APIs are unavailable and is also
    useful for tests where repeatability matters.
    """

    provider_name = "mock-realtime"

    @staticmethod
    def _stable_float(seed: str, low: float, high: float) -> float:
        digest = md5(seed.encode("utf-8")).hexdigest()[:8]
        value = int(digest, 16) / float(16**8)
        return low + (high - low) * value

    def _team_seed(self, team: str) -> str:
        return team.strip().lower() or "team"

    def fetch_live_matches(self, tournament: Optional[str] = None, limit: int = 20) -> list[dict[str, Any]]:
        now = datetime.now(timezone.utc)
        rows = [
            {
                "match_id": "mock_live_mi_csk",
                "sport": "cricket",
                "tournament": tournament or "IPL",
                "team_a": "Mumbai Indians",
                "team_b": "Chennai Super Kings",
                "venue": "Wankhede Stadium",
                "match_date": now.date().isoformat(),
                "state": "live",
                "provider": self.provider_name,
                "provider_source": "mock",
                "provider_updated_at": _utc_now_iso(),
                "freshness_seconds": 0,
            }
        ]
        return rows[: max(1, limit)]

    def fetch_upcoming_matches(self, tournament: Optional[str] = None, limit: int = 20) -> list[dict[str, Any]]:
        dt = datetime.now(timezone.utc) + timedelta(days=1)
        rows = [
            {
                "match_id": "mock_upcoming_rcb_kkr",
                "sport": "cricket",
                "tournament": tournament or "IPL",
                "team_a": "Royal Challengers Bengaluru",
                "team_b": "Kolkata Knight Riders",
                "venue": "M Chinnaswamy Stadium",
                "match_date": dt.date().isoformat(),
                "state": "upcoming",
                "provider": self.provider_name,
                "provider_source": "mock",
                "provider_updated_at": _utc_now_iso(),
                "freshness_seconds": 0,
            }
        ]
        return rows[: max(1, limit)]

    def fetch_match_context(
        self,
        match_id: Optional[str] = None,
        team_a: Optional[str] = None,
        team_b: Optional[str] = None,
        match_date: Optional[str] = None,
        tournament: Optional[str] = None,
    ) -> dict[str, Any]:
        seed = f"{match_id}|{team_a}|{team_b}|{datetime.utcnow().hour}"
        batting_runs = self._stable_float(seed + ":runs", 152.0, 196.0)
        wickets = self._stable_float(seed + ":wickets", 4.0, 8.0)
        bowling_runs = self._stable_float(seed + ":conceded", 156.0, 201.0)
        bowling_wk = self._stable_float(seed + ":bwk", 4.5, 8.8)

        team_a_last_5 = round(batting_runs, 3)
        team_a_last_10 = round(self._stable_float(seed + ":runs10", 148.0, 193.0), 3)
        team_b_last_5 = round(max(95.0, bowling_runs - 3.0), 3)
        team_b_last_10 = round(max(92.0, team_b_last_5 - 4.0), 3)
        team_a_chase_success = round(self._stable_float(seed + ":a_chase", 0.34, 0.71), 3)
        team_b_chase_success = round(self._stable_float(seed + ":b_chase", 0.3, 0.69), 3)
        team_a_defend_success = round(max(0.1, min(0.9, 1.0 - team_b_chase_success)), 3)
        team_b_defend_success = round(max(0.1, min(0.9, 1.0 - team_a_chase_success)), 3)
        venue_batting_first_adv = round(self._stable_float(seed + ":vf", -0.35, 0.35), 3)
        team_a_win_rate_vs_b = round(self._stable_float(seed + ":bwin", 0.32, 0.76), 3)
        batting_first_flag = round(self._stable_float(seed + ":toss", 0.0, 1.0), 3)
        features = {
            "team_a_avg_runs_last_5": team_a_last_5,
            "team_a_avg_runs_last_10": team_a_last_10,
            "team_a_avg_wickets_last_5": round(wickets, 3),
            "team_a_run_rate_trend": round(team_a_last_5 - team_a_last_10, 3),
            "team_b_avg_runs_last_5": team_b_last_5,
            "team_b_avg_runs_last_10": team_b_last_10,
            "team_b_avg_wickets_last_5": round(bowling_wk, 3),
            "team_b_run_rate_trend": round(team_b_last_5 - team_b_last_10, 3),
            "team_a_win_rate_vs_b": team_a_win_rate_vs_b,
            "avg_score_team_a_vs_b": round(self._stable_float(seed + ":h2h1", 154.0, 186.0), 3),
            "avg_score_team_b_vs_a": round(self._stable_float(seed + ":h2h2", 148.0, 182.0), 3),
            "venue_avg_score": round(self._stable_float(seed + ":vavg", 154.0, 185.0), 3),
            "venue_chase_success_rate": round(self._stable_float(seed + ":vch", 0.3, 0.66), 3),
            "venue_defend_bias": round(-venue_batting_first_adv, 3),
            "team_a_runs_vs_opponent_avg": round(team_a_last_5 - team_b_last_5, 3),
            "team_b_runs_vs_opponent_avg": round(team_b_last_5 - team_a_last_5, 3),
            "batting_first": batting_first_flag,
            "team_a_bats_first": 1.0 if batting_first_flag >= 0.5 else 0.0,
            "team_b_bats_first": 0.0 if batting_first_flag >= 0.5 else 1.0,
            "team_a_chase_success_rate": team_a_chase_success,
            "team_b_chase_success_rate": team_b_chase_success,
            "team_a_defend_success_rate": team_a_defend_success,
            "team_b_defend_success_rate": team_b_defend_success,
            "chase_defend_edge_team_a_first": round(team_b_chase_success - team_a_defend_success, 3),
            "chase_defend_edge_team_b_first": round(team_a_chase_success - team_b_defend_success, 3),
            "venue_batting_first_advantage": venue_batting_first_adv,
            "recent_form_diff": round(team_a_last_5 - team_b_last_5, 3),
            "recent_run_rate_diff": round((team_a_last_5 - team_a_last_10) - (team_b_last_5 - team_b_last_10), 3),
            "head_to_head_win_diff": round((2.0 * team_a_win_rate_vs_b) - 1.0, 3),
            "wickets_taken_diff": round(bowling_wk - wickets, 3),
        }
        return {
            "provider": self.provider_name,
            "source": "mock",
            "updated_at": _utc_now_iso(),
            "freshness_seconds": 0,
            "features": features,
            "summary": "Mock deterministic context for local development.",
        }

    def fetch_recent_team_stats(self, team: str, limit: int = 5) -> dict[str, Any]:
        seed = self._team_seed(team)
        return {
            "provider": self.provider_name,
            "team": team,
            "source": "mock",
            "matches_used": limit,
            "avg_runs_signal": round(self._stable_float(seed + ":runs", 150.0, 198.0), 3),
            "avg_wickets_signal": round(self._stable_float(seed + ":wk", 4.2, 8.6), 3),
        }

    def fetch_recent_player_stats(self, player: str, limit: int = 5) -> dict[str, Any]:
        seed = self._team_seed(player)
        return {
            "provider": self.provider_name,
            "player": player,
            "source": "mock",
            "matches_used": limit,
            "batting_signal": round(self._stable_float(seed + ":bat", 18.0, 68.0), 3),
            "bowling_signal": round(self._stable_float(seed + ":bowl", 12.0, 54.0), 3),
        }

    def healthcheck(self) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "status": "ok",
            "source": "mock",
            "updated_at": _utc_now_iso(),
            "freshness_seconds": 0,
            "api_key_configured": False,
        }

    def enrich_match_context(self, match_row: dict[str, Any]) -> dict[str, Any]:
        context = self.fetch_match_context(
            match_id=match_row.get("match_id"),
            team_a=match_row.get("team_a"),
            team_b=match_row.get("team_b"),
            match_date=match_row.get("match_date"),
            tournament=match_row.get("tournament"),
        )
        seed = f"{match_row.get('match_id','')}|{datetime.utcnow().hour}"
        tempo_shift = self._stable_float(seed + "tempo", -0.12, 0.18)
        pressure_index = self._stable_float(seed + "pressure", 0.32, 0.78)
        injury_risk = self._stable_float(seed + "injury", 0.02, 0.21)

        return {
            "provider": self.provider_name,
            "updated_at": context.get("updated_at", _utc_now_iso()),
            "tempo_shift": round(tempo_shift, 3),
            "pressure_index": round(pressure_index, 3),
            "injury_risk": round(injury_risk, 3),
            "source": context.get("source", "mock"),
            "freshness_seconds": context.get("freshness_seconds", 0),
        }
