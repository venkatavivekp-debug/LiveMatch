from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class SportResponse(BaseModel):
    code: str
    name: str
    description: str
    tournaments_supported: int = Field(ge=0)


class TournamentResponse(BaseModel):
    code: str
    name: str
    sport: str
    category: str = "league"
    matches_available: Optional[int] = None
    seasons: list[str] = Field(default_factory=list)


class MatchResponse(BaseModel):
    match_id: str
    sport: str
    tournament: str
    team_a: str
    team_b: str
    venue: str
    match_date: Optional[date]
    state: str = "historical"
    start_time: Optional[str] = None
    data_source: Optional[str] = None


class HeadToHeadEvaluationResponse(BaseModel):
    predicted_heads: list[str] = Field(default_factory=list)
    best_match_to_actual: Optional[str] = None
    best_error: Optional[float] = None
    in_range: Optional[bool] = None
    winner_correct: Optional[bool] = None


class HeadToHeadMatchResponse(BaseModel):
    match_id: str
    sport: str
    tournament: str
    team_a: str
    team_b: str
    venue: str
    match_date: Optional[date]
    winner: Optional[str] = None
    actual_result: Optional[str] = None
    evaluation: Optional[HeadToHeadEvaluationResponse] = None
