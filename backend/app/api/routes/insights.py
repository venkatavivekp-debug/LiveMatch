from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status
from app.schemas.analytics import (
    LiveInsightsResponse,
    ModelStatusResponse,
    RefreshLiveDataRequest,
    RefreshLiveDataResponse,
    SystemStatusResponse,
    TopPlayerResponse,
)
from app.services.insights_service import InsightsService
from app.services.prediction_service import PredictionService

router = APIRouter(tags=["insights"])
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


@router.get("/insights/live", response_model=LiveInsightsResponse)
def get_live_insights(
    sport: str = Query(default="cricket"),
    tournament: Optional[str] = Query(default=None),
    state: Optional[str] = Query(default=None),
    limit: int = Query(default=4, ge=1, le=12),
) -> LiveInsightsResponse:
    payload = InsightsService.live_insights(
        sport=sport,
        tournament=tournament,
        state=_normalize_state(state),
        limit=limit,
    )
    return LiveInsightsResponse(**payload)


@router.get("/players/top", response_model=list[TopPlayerResponse])
def get_top_players(
    sport: str = Query(default="cricket"),
    tournament: Optional[str] = Query(default=None),
    team: Optional[str] = Query(default=None),
    role: Optional[str] = Query(default=None),
    limit: int = Query(default=5, ge=1, le=15),
) -> list[TopPlayerResponse]:
    rows = InsightsService.top_players(
        sport=sport,
        tournament=tournament,
        team=team,
        role=role,
        limit=limit,
    )
    return [TopPlayerResponse(**row) for row in rows]


@router.get("/model/status", response_model=ModelStatusResponse)
def get_model_status() -> ModelStatusResponse:
    return ModelStatusResponse(**PredictionService.model_status())


@router.get("/system/status", response_model=SystemStatusResponse)
def get_system_status() -> SystemStatusResponse:
    payload = PredictionService.system_status()
    return SystemStatusResponse(**payload)


@router.post("/admin/refresh-live-data", response_model=RefreshLiveDataResponse)
def refresh_live_data(
    payload: RefreshLiveDataRequest,
) -> RefreshLiveDataResponse:
    result = PredictionService.refresh_live_data(sport=payload.sport, tournament=payload.tournament)
    return RefreshLiveDataResponse(**result)
