from __future__ import annotations

from datetime import date
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status

from app.schemas.common import HeadToHeadMatchResponse, MatchResponse, SportResponse, TournamentResponse
from app.services.catalog_service import CatalogService

router = APIRouter(tags=["catalog"])

VALID_STATES = {"live", "upcoming", "historical", "completed"}


def _normalize_state(state: Optional[str]) -> Optional[str]:
    if state is None:
        return None
    key = str(state).strip().lower()
    if not key:
        return None
    if key not in VALID_STATES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="state must be one of: live, upcoming, historical, completed",
        )
    if key == "completed":
        return "historical"
    return key


@router.get("/sports", response_model=list[SportResponse])
def list_sports() -> list[SportResponse]:
    return CatalogService.list_sports()


@router.get("/tournaments", response_model=list[TournamentResponse])
def list_tournaments(sport: Optional[str] = Query(default=None)) -> list[TournamentResponse]:
    return CatalogService.list_tournaments(sport=sport)


@router.get("/matches", response_model=list[MatchResponse])
def list_matches(
    sport: Optional[str] = Query(default=None),
    tournament: Optional[str] = Query(default=None),
    team: Optional[str] = Query(default=None),
    venue: Optional[str] = Query(default=None),
    date_from: Optional[date] = Query(default=None),
    date_to: Optional[date] = Query(default=None),
    state: Optional[str] = Query(default=None),
) -> list[MatchResponse]:
    return CatalogService.list_matches(
        sport=sport,
        tournament=tournament,
        team=team,
        venue=venue,
        date_from=date_from,
        date_to=date_to,
        state=_normalize_state(state),
    )


@router.get("/matches/head-to-head", response_model=list[HeadToHeadMatchResponse])
def head_to_head_history(
    sport: str = Query(default="cricket"),
    team_a: str = Query(..., min_length=2),
    team_b: str = Query(..., min_length=2),
    tournament: Optional[str] = Query(default=None),
    limit: int = Query(default=5, ge=1, le=15),
    include_evaluation: bool = Query(default=True),
    k: int = Query(default=4, ge=2, le=7),
) -> list[HeadToHeadMatchResponse]:
    return CatalogService.head_to_head_history(
        sport=sport,
        team_a=team_a,
        team_b=team_b,
        tournament=tournament,
        limit=limit,
        include_evaluation=include_evaluation,
        k=k,
    )
