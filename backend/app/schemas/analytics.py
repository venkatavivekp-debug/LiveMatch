from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict

from app.schemas.prediction import ExplanationFactor


class LiveInsightCard(BaseModel):
    match_id: str
    sport: str
    tournament: str
    team_a: str
    team_b: str
    state: str
    summary: str
    reasons: list[ExplanationFactor]


class LiveInsightsResponse(BaseModel):
    provider: str
    as_of: str
    cards: list[LiveInsightCard]


class TopPlayerResponse(BaseModel):
    name: str
    team: str
    sport: str
    tournament: str
    role: str
    impact_score: float
    confidence: float
    reasons: list[ExplanationFactor]


class ModelStatusResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model: Optional[str] = None
    data: Optional[str] = None
    provider: Optional[str] = None
    heads: Optional[int] = None
    last_update: Optional[str] = None
    healthy: Optional[bool] = None
    model_mode: Optional[str] = None
    runtime_mode: str
    data_mode: Optional[str] = None
    checkpoint_exists: bool
    torch_available: bool
    num_heads: int
    model_version: Optional[str] = None
    checkpoint_updated_at: Optional[str] = None
    active_experiment_id: Optional[str] = None
    latest_run_id: Optional[str] = None
    encoder_type: Optional[str] = None
    encoder_config: Optional[dict] = None
    calibration: Optional[dict] = None
    latest_evaluation_summary: Optional[dict] = None
    provider_status: Optional[dict[str, Any]] = None
    last_live_sync_time: Optional[str] = None
    live_sync_status: Optional[dict[str, Any]] = None
    artifact_paths: Optional[dict[str, str]] = None
    sports_supported: list[str]
    active_provider: str
    training_job_mode: Optional[str] = None
    notes: Optional[str] = None


class RefreshLiveDataRequest(BaseModel):
    sport: str = "cricket"
    tournament: Optional[str] = None


class RefreshLiveDataResponse(BaseModel):
    status: str
    provider: str
    sport: str
    tournament: Optional[str] = None
    refreshed_at: str
    live_matches: Optional[int] = None
    upcoming_matches: Optional[int] = None
    message: Optional[str] = None
    error: Optional[str] = None
    provider_health: Optional[dict[str, Any]] = None


class SystemStatusResponse(BaseModel):
    generated_at: str
    model: ModelStatusResponse
    provider: Optional[dict[str, Any]] = None
    live_sync: Optional[dict[str, Any]] = None
