from __future__ import annotations

from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, Field


class MatchContextRequest(BaseModel):
    match_id: Optional[str] = None
    sport: str = Field(default="cricket", min_length=3, max_length=32)
    tournament: str = "IPL"
    team_a: Optional[str] = Field(default=None, min_length=2, max_length=80)
    team_b: Optional[str] = Field(default=None, min_length=2, max_length=80)
    venue: Optional[str] = Field(default=None, min_length=2, max_length=120)
    match_date: Optional[str] = None
    state: str = Field(default="upcoming", min_length=4, max_length=24)


class PredictionRequest(BaseModel):
    match: MatchContextRequest
    k: int = Field(default=3, ge=2, le=7)


class ExplanationFactor(BaseModel):
    feature: str
    value: Optional[Union[float, str, int]] = None
    baseline: Optional[Union[float, str, int]] = None
    delta: Optional[float] = None
    unit: Optional[str] = None
    impact: str
    explanation: str


class ScenarioPrediction(BaseModel):
    label: Optional[str] = None
    story: Optional[str] = None
    score: Optional[int] = None
    team_a_score: Optional[int] = None
    team_b_score: Optional[int] = None
    winner: Optional[str] = None
    scoreline: Optional[str] = None
    home_goals: Optional[int] = None
    away_goals: Optional[int] = None
    likely_result: Optional[str] = None
    team_a_first: Optional[Dict[str, Any]] = None
    team_b_first: Optional[Dict[str, Any]] = None
    scenario: str
    reason: list[ExplanationFactor]
    confidence: float
    scenario_probability: Optional[float] = None


class PlayerPrediction(BaseModel):
    name: str
    role: str
    team: Optional[str] = None
    reason: list[ExplanationFactor]
    confidence: float


class MatchMetadataResponse(BaseModel):
    sport: str
    tournament: str
    team_a: str
    team_b: str
    venue: str
    match_date: Optional[str]
    state: str = "upcoming"


class UncertaintySummary(BaseModel):
    spread: float
    interval_low: float
    interval_high: float
    mean_prediction: float
    std_prediction: Optional[float] = None


class ForecastSummary(BaseModel):
    favored_team: Optional[str] = None
    favored_team_confidence: Optional[float] = None
    win_probability: Optional[float] = None
    predicted_band_low: Optional[float] = None
    predicted_band_high: Optional[float] = None
    expected_score_range: Optional[str] = None
    key_risk: Optional[str] = None
    risk_level: Optional[str] = None
    risk_explanation: Optional[str] = None
    final_summary: Optional[str] = None


class PerformanceSummary(BaseModel):
    reliability: Optional[str] = None
    accuracy: Optional[float] = None
    avg_error: Optional[float] = None
    in_range_pct: Optional[float] = None
    samples: Optional[int] = None
    interpretation: Optional[str] = None


class PredictionResponse(BaseModel):
    match: MatchMetadataResponse
    predictions: list[ScenarioPrediction]
    scenarios: Optional[list[ScenarioPrediction]] = None
    best_player: PlayerPrediction
    best_bowler: Optional[PlayerPrediction] = None
    man_of_the_match: PlayerPrediction
    players: Optional[Dict[str, list[PlayerPrediction]]] = None
    match_insight: Optional[str] = None
    forecast_summary: Optional[ForecastSummary] = None
    performance_summary: Optional[PerformanceSummary] = None
    uncertainty: UncertaintySummary
    metadata: Dict[str, Any]


class MatchEvaluationSummary(BaseModel):
    available: bool
    target_type: str
    actual_value: Optional[float] = None
    actual_first_innings_team: Optional[str] = None
    actual_score_summary: Optional[str] = None
    actual_winner: Optional[str] = None
    predicted_winner: Optional[str] = None
    winner_correct: Optional[bool] = None
    winner_index: Optional[int] = None
    winner_scenario: Optional[str] = None
    best_matching_scenario: Optional[str] = None
    best_matching_branch: Optional[str] = None
    best_head_value: Optional[float] = None
    predicted_heads: list[str] = Field(default_factory=list)
    best_match_error: Optional[float] = None
    best_match_error_method: Optional[str] = None
    team_a_score_error: Optional[float] = None
    team_b_score_error: Optional[float] = None
    center_prediction: Optional[float] = None
    center_error: Optional[float] = None
    interval_low: Optional[float] = None
    interval_high: Optional[float] = None
    interval_covered: Optional[bool] = None
    evaluation_summary: Optional[str] = None
    source: Optional[str] = None
    message: Optional[str] = None


class MatchEvaluationResponse(BaseModel):
    match_id: str
    prediction: PredictionResponse
    evaluation: MatchEvaluationSummary


class BatchPredictionRequest(BaseModel):
    requests: list[PredictionRequest] = Field(min_length=1, max_length=25)


class BatchPredictionItem(BaseModel):
    index: int
    prediction: Optional[PredictionResponse] = None
    error: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    total: int
    success: int
    failed: int
    items: list[BatchPredictionItem]
