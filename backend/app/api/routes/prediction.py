from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, status
from app.core.config import get_settings
from app.schemas.prediction import (
    BatchPredictionItem,
    BatchPredictionRequest,
    BatchPredictionResponse,
    MatchEvaluationResponse,
    PredictionRequest,
    PredictionResponse,
    ScenarioPrediction,
    UncertaintySummary,
)
from app.services.catalog_service import CatalogService
from app.services.prediction_service import PredictionService

router = APIRouter(tags=["predictions"])
VALID_STATES = {"live", "upcoming", "historical", "completed"}


def _normalize_state(state: str) -> str:
    key = str(state or "").strip().lower() or "upcoming"
    if key not in VALID_STATES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="state must be one of: live, upcoming, historical, completed",
        )
    if key == "completed":
        return "historical"
    return key


def _resolve_match_payload(match_payload: dict[str, Any]) -> dict[str, Any]:
    payload = match_payload.copy()

    if payload.get("match_id"):
        match_record = CatalogService.get_match_by_id(str(payload["match_id"]))
        if match_record is not None:
            resolved = match_record.model_dump()
            for field in ["sport", "tournament", "team_a", "team_b", "venue", "match_date", "state"]:
                if not payload.get(field):
                    payload[field] = resolved.get(field)

                if field == "sport" and str(payload.get(field, "")).lower() != str(resolved.get(field, "")).lower():
                    payload[field] = resolved.get(field)

            if payload.get("match_date") is not None:
                payload["match_date"] = str(payload["match_date"])

    required = ["team_a", "team_b", "venue"]
    missing = [field for field in required if not payload.get(field)]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Missing required match fields: {', '.join(missing)}",
        )

    payload["sport"] = str(payload.get("sport") or "cricket").lower()
    payload["state"] = str(payload.get("state") or "upcoming").lower()
    payload["tournament"] = str(payload.get("tournament") or ("IPL" if payload["sport"] == "cricket" else "EPL"))
    return payload


@router.post("/predict", response_model=PredictionResponse)
def predict_match(
    payload: PredictionRequest,
) -> PredictionResponse:
    resolved_match_payload = _resolve_match_payload(payload.match.model_dump())
    response = PredictionService.predict(match_payload=resolved_match_payload, k=payload.k)
    return PredictionResponse(**response)


@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(
    payload: BatchPredictionRequest,
) -> BatchPredictionResponse:
    settings = get_settings()
    if len(payload.requests) > settings.max_batch_predictions:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Batch size {len(payload.requests)} exceeds configured limit "
                f"{settings.max_batch_predictions}."
            ),
        )

    items: list[BatchPredictionItem] = []
    success = 0
    for idx, request_payload in enumerate(payload.requests):
        try:
            resolved = _resolve_match_payload(request_payload.match.model_dump())
            prediction = PredictionService.predict(match_payload=resolved, k=request_payload.k)
            items.append(BatchPredictionItem(index=idx, prediction=PredictionResponse(**prediction)))
            success += 1
        except Exception as exc:  # noqa: BLE001
            items.append(BatchPredictionItem(index=idx, error=str(exc)))

    failed = len(payload.requests) - success
    return BatchPredictionResponse(
        total=len(payload.requests),
        success=success,
        failed=failed,
        items=items,
    )


def _build_query_match_payload(
    match_id: Optional[str],
    sport: str,
    tournament: Optional[str],
    team_a: Optional[str],
    team_b: Optional[str],
    venue: Optional[str],
    match_date: Optional[str],
    state: str,
) -> dict[str, Any]:
    return {
        "match_id": match_id,
        "sport": sport,
        "tournament": tournament,
        "team_a": team_a,
        "team_b": team_b,
        "venue": venue,
        "match_date": match_date,
        "state": _normalize_state(state),
    }


@router.get("/forecast/scenarios", response_model=list[ScenarioPrediction])
def forecast_scenarios(
    match_id: Optional[str] = Query(default=None),
    sport: str = Query(default="cricket"),
    tournament: Optional[str] = Query(default=None),
    team_a: Optional[str] = Query(default=None),
    team_b: Optional[str] = Query(default=None),
    venue: Optional[str] = Query(default=None),
    match_date: Optional[str] = Query(default=None),
    state: str = Query(default="upcoming"),
    k: int = Query(default=3, ge=2, le=7),
) -> list[ScenarioPrediction]:
    resolved = _resolve_match_payload(
        _build_query_match_payload(match_id, sport, tournament, team_a, team_b, venue, match_date, state)
    )
    output = PredictionService.predict(match_payload=resolved, k=k)
    return [ScenarioPrediction(**row) for row in output["predictions"]]


@router.get("/forecast/uncertainty", response_model=UncertaintySummary)
def forecast_uncertainty(
    match_id: Optional[str] = Query(default=None),
    sport: str = Query(default="cricket"),
    tournament: Optional[str] = Query(default=None),
    team_a: Optional[str] = Query(default=None),
    team_b: Optional[str] = Query(default=None),
    venue: Optional[str] = Query(default=None),
    match_date: Optional[str] = Query(default=None),
    state: str = Query(default="upcoming"),
    k: int = Query(default=3, ge=2, le=7),
) -> UncertaintySummary:
    resolved = _resolve_match_payload(
        _build_query_match_payload(match_id, sport, tournament, team_a, team_b, venue, match_date, state)
    )
    output = PredictionService.predict(match_payload=resolved, k=k)
    return UncertaintySummary(**output["uncertainty"])


@router.get("/matches/{match_id}/evaluation", response_model=MatchEvaluationResponse)
def evaluate_match(
    match_id: str,
    k: int = Query(default=4, ge=2, le=7),
) -> MatchEvaluationResponse:
    resolved = _resolve_match_payload(_build_query_match_payload(match_id, "cricket", None, None, None, None, None, "historical"))
    output = PredictionService.evaluate_match(match_payload=resolved, k=k)
    return MatchEvaluationResponse(**output)
