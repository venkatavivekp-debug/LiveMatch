from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from app.services.providers.base import SportsDataProvider

SPORTS_CATALOG = [
    {
        "code": "cricket",
        "name": "Cricket",
        "description": "Ball-by-ball style probabilistic match forecasting",
    },
    {
        "code": "football",
        "name": "Football",
        "description": "Scoreline scenario forecasting with uncertainty spread",
    },
]

TOURNAMENT_NAME_MAP = {
    "IPL": "Indian Premier League",
    "ICC_T20_WC": "ICC Men's T20 World Cup",
    "EPL": "English Premier League",
    "UCL": "UEFA Champions League",
    "LALIGA": "La Liga",
}


class LocalDemoProvider(SportsDataProvider):
    def __init__(self, processed_dir: Path) -> None:
        self.processed_dir = processed_dir
        self._matches_cache: Optional[pd.DataFrame] = None
        self._cricket_players_cache: Optional[pd.DataFrame] = None
        self._football_players_cache: Optional[pd.DataFrame] = None

    @staticmethod
    def _clean_text(value: Any) -> str:
        text = " ".join(str(value or "").strip().split())
        return text

    @classmethod
    def _normalize_state(cls, value: Any) -> str:
        state = cls._clean_text(value).lower()
        if state in {"live", "upcoming", "historical"}:
            return state
        if state == "completed":
            return "historical"
        return "historical"

    @classmethod
    def _sanitize_matches(cls, frame: pd.DataFrame, sport: str) -> pd.DataFrame:
        required_cols = ["match_id", "tournament", "team_a", "team_b", "venue", "match_date", "state"]
        for col in required_cols:
            if col not in frame.columns:
                frame[col] = None

        frame = frame.copy()
        frame["sport"] = sport
        frame["match_id"] = frame["match_id"].map(cls._clean_text)
        frame["tournament"] = frame["tournament"].map(cls._clean_text)
        frame["team_a"] = frame["team_a"].map(cls._clean_text)
        frame["team_b"] = frame["team_b"].map(cls._clean_text)
        frame["venue"] = frame["venue"].map(cls._clean_text)
        frame["state"] = frame["state"].map(cls._normalize_state)
        frame["match_date"] = pd.to_datetime(frame["match_date"], errors="coerce").dt.date

        frame = frame[
            (frame["match_id"] != "")
            & (frame["tournament"] != "")
            & (frame["team_a"] != "")
            & (frame["team_b"] != "")
            & (frame["venue"] != "")
            & (frame["team_a"].str.lower() != frame["team_b"].str.lower())
        ]
        keep = ["match_id", "sport", "tournament", "team_a", "team_b", "venue", "match_date", "state"]
        return frame[keep].drop_duplicates(subset=["match_id"], keep="last").reset_index(drop=True)

    @classmethod
    def _sanitize_players(cls, frame: pd.DataFrame, sport: str) -> pd.DataFrame:
        if frame.empty:
            return frame
        frame = frame.copy()
        for col in ["player", "team", "tournament", "role"]:
            if col in frame.columns:
                frame[col] = frame[col].map(cls._clean_text)
        if "player" in frame.columns:
            frame = frame[frame["player"].astype(str).str.strip() != ""]
        if "team" in frame.columns:
            frame = frame[frame["team"].astype(str).str.strip() != ""]
        frame["sport"] = sport
        return frame.reset_index(drop=True)

    def list_sports(self) -> list[dict[str, Any]]:
        frame = self._matches_frame()
        tournament_counts = {
            sport["code"]: int(
                frame[frame["sport"].str.lower() == sport["code"]]["tournament"].dropna().nunique()
            )
            for sport in SPORTS_CATALOG
        }
        return [
            {
                **sport,
                "tournaments_supported": tournament_counts.get(sport["code"], 0),
            }
            for sport in SPORTS_CATALOG
        ]

    def list_tournaments(self, sport: Optional[str] = None) -> list[dict[str, Any]]:
        frame = self._matches_frame()
        if sport:
            frame = frame[frame["sport"].str.lower() == sport.lower()]
        if frame.empty:
            return []
        output = []
        for (sport_code, tournament_code), subset in frame.groupby(["sport", "tournament"], dropna=True):
            code = str(tournament_code)
            sport_key = str(sport_code).lower()
            seasons = sorted(
                {
                    str(dt.year)
                    for dt in pd.to_datetime(subset["match_date"], errors="coerce").dropna()
                    if hasattr(dt, "year")
                }
            )
            output.append(
                {
                    "code": code,
                    "name": TOURNAMENT_NAME_MAP.get(code, code.replace("_", " ")),
                    "sport": sport_key,
                    "category": "international" if "wc" in code.lower() or "world cup" in code.lower() else "league",
                    "matches_available": int(len(subset)),
                    "seasons": seasons,
                }
            )
        output.sort(key=lambda row: (row["sport"], row["code"]))
        return output

    def _load_cricket_matches(self) -> pd.DataFrame:
        path = self.processed_dir / "matches.csv"
        if path.exists():
            frame = pd.read_csv(path)
        else:
            frame = pd.DataFrame(columns=["match_id", "tournament", "team_a", "team_b", "venue", "match_date"])

        if "state" not in frame.columns:
            today = datetime.utcnow().date()
            match_dates = pd.to_datetime(frame.get("match_date"), errors="coerce").dt.date
            frame["state"] = match_dates.apply(
                lambda dt: "upcoming" if isinstance(dt, date) and dt >= today else "historical"
            )
        return self._sanitize_matches(frame, sport="cricket")

    def _load_football_matches(self) -> pd.DataFrame:
        path = self.processed_dir / "football_matches.csv"
        if path.exists():
            frame = pd.read_csv(path)
        else:
            frame = pd.DataFrame(columns=["match_id", "sport", "tournament", "team_a", "team_b", "venue", "match_date"])

        if "state" not in frame.columns:
            today = datetime.utcnow().date()
            match_dates = pd.to_datetime(frame.get("match_date"), errors="coerce").dt.date
            frame["state"] = match_dates.apply(
                lambda dt: "upcoming" if isinstance(dt, date) and dt >= today else "historical"
            )
        return self._sanitize_matches(frame, sport="football")

    def _matches_frame(self) -> pd.DataFrame:
        if self._matches_cache is None:
            cricket = self._load_cricket_matches()
            football = self._load_football_matches()
            frame = pd.concat([cricket, football], ignore_index=True)
            frame["state"] = frame["state"].fillna("historical").str.lower()
            self._matches_cache = frame
        return self._matches_cache.copy()

    def list_matches(
        self,
        sport: Optional[str] = None,
        tournament: Optional[str] = None,
        team: Optional[str] = None,
        venue: Optional[str] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        state: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        frame = self._matches_frame()

        if sport:
            frame = frame[frame["sport"].str.lower() == sport.lower()]
        if tournament:
            frame = frame[frame["tournament"].str.lower() == tournament.lower()]
        if team:
            lookup = team.lower()
            frame = frame[
                frame["team_a"].str.lower().str.contains(lookup, na=False)
                | frame["team_b"].str.lower().str.contains(lookup, na=False)
            ]
        if venue:
            frame = frame[frame["venue"].str.lower().str.contains(venue.lower(), na=False)]
        if date_from:
            frame = frame[frame["match_date"] >= date_from]
        if date_to:
            frame = frame[frame["match_date"] <= date_to]
        if state:
            frame = frame[frame["state"].str.lower() == state.lower()]

        frame = frame.sort_values(by=["match_date", "state"], ascending=[False, True]).reset_index(drop=True)
        frame["match_date"] = frame["match_date"].where(frame["match_date"].notna(), None)
        return frame.to_dict(orient="records")

    def get_match_by_id(self, match_id: str) -> Optional[dict[str, Any]]:
        frame = self._matches_frame()
        subset = frame[frame["match_id"] == match_id]
        if subset.empty:
            return None
        subset = subset.copy()
        subset["match_date"] = subset["match_date"].where(subset["match_date"].notna(), None)
        return subset.iloc[0].to_dict()

    def _load_cricket_players(self) -> pd.DataFrame:
        if self._cricket_players_cache is not None:
            return self._cricket_players_cache.copy()

        path = self.processed_dir / "player_form_latest.csv"
        if path.exists():
            frame = pd.read_csv(path)
        else:
            frame = pd.DataFrame(columns=["player", "team"])
        frame = self._sanitize_players(frame, sport="cricket")
        if "tournament" not in frame.columns:
            frame["tournament"] = "IPL"
        frame["tournament"] = frame["tournament"].map(self._clean_text).replace("", "IPL")
        self._cricket_players_cache = frame
        return frame.copy()

    def _load_football_players(self) -> pd.DataFrame:
        if self._football_players_cache is not None:
            return self._football_players_cache.copy()

        path = self.processed_dir / "football_player_form_latest.csv"
        if path.exists():
            frame = pd.read_csv(path)
        else:
            frame = pd.DataFrame(columns=["player", "team", "role"])
        frame = self._sanitize_players(frame, sport="football")
        self._football_players_cache = frame
        return frame.copy()

    def get_top_players(
        self,
        sport: str,
        tournament: Optional[str] = None,
        team: Optional[str] = None,
        role: Optional[str] = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        sport_code = sport.lower()
        if sport_code == "football":
            frame = self._load_football_players()
            if frame.empty:
                return []
            if tournament:
                frame = frame[frame["tournament"].str.lower() == tournament.lower()]
            if team:
                frame = frame[frame["team"].str.lower().str.contains(team.lower(), na=False)]
            if frame.empty:
                return []

            frame = frame.copy()
            frame["rank_score"] = (
                frame.get("impact_score", 50) * 0.45
                + frame.get("goals_last5", 0) * 8.5
                + frame.get("xg_per90", 0.2) * 28.0
                + frame.get("shot_conversion", 0.1) * 30.0
            )

            if role:
                role_lookup = role.lower()
                if role_lookup in {"goal_scorer", "scorer"}:
                    frame = frame[frame["role"].str.lower().isin(["goal_scorer", "standout"])]
                elif role_lookup in {"standout", "mvp"}:
                    frame = frame[frame["role"].str.lower().isin(["standout", "goal_scorer", "defender"])]
                elif role_lookup == "defender":
                    frame = frame[frame["role"].str.lower().isin(["defender", "standout"])]

            frame = frame.sort_values("rank_score", ascending=False).head(limit)
            return frame.to_dict(orient="records")

        frame = self._load_cricket_players()
        if frame.empty:
            return []
        if tournament:
            frame = frame[frame["tournament"].str.lower() == tournament.lower()]
        if team:
            frame = frame[frame["team"].str.lower().str.contains(team.lower(), na=False)]
        if frame.empty:
            return []

        frame = frame.copy()
        frame["batter_rank"] = (
            frame.get("batting_form", 25) * 0.68
            + frame.get("strike_rate", 120) * 0.08
            + frame.get("recent_runs", 20) * 0.24
        )
        frame["bowler_rank"] = (
            frame.get("bowling_form", 18) * 0.74
            + frame.get("recent_wickets", 0.8) * 9.0
            + (8.8 - frame.get("economy", 8.5)).clip(lower=0) * 6.2
        )
        frame["impact_rank"] = frame.get("impact_score", 20) * 1.0

        role_lookup = (role or "standout").lower()
        if role_lookup in {"batsman", "batter"}:
            frame["rank_score"] = frame["batter_rank"]
        elif role_lookup in {"bowler"}:
            frame["rank_score"] = frame["bowler_rank"]
        else:
            frame["rank_score"] = frame["impact_rank"]

        frame = frame.sort_values("rank_score", ascending=False).head(limit)
        return frame.to_dict(orient="records")
