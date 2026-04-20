from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Any, Optional


class SportsDataProvider(ABC):
    @abstractmethod
    def list_sports(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def list_tournaments(self, sport: Optional[str] = None) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def get_match_by_id(self, match_id: str) -> Optional[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_top_players(
        self,
        sport: str,
        tournament: Optional[str] = None,
        team: Optional[str] = None,
        role: Optional[str] = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError


class RealtimeContextProvider(ABC):
    provider_name: str = "unknown-live-provider"

    def fetch_upcoming_matches(
        self,
        tournament: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        return []

    def fetch_live_matches(
        self,
        tournament: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        return []

    def fetch_match_context(
        self,
        match_id: Optional[str] = None,
        team_a: Optional[str] = None,
        team_b: Optional[str] = None,
        match_date: Optional[str] = None,
        tournament: Optional[str] = None,
    ) -> dict[str, Any]:
        return {}

    def fetch_recent_team_stats(self, team: str, limit: int = 5) -> dict[str, Any]:
        return {}

    def fetch_recent_player_stats(self, player: str, limit: int = 5) -> dict[str, Any]:
        return {}

    def healthcheck(self) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "status": "unknown",
        }

    @abstractmethod
    def enrich_match_context(self, match_row: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError
