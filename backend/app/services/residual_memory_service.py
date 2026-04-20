from __future__ import annotations

import uuid
from datetime import datetime
from statistics import mean
from typing import Any

from sqlalchemy.orm import Session

from app.db.models import PredictionResidualRecord
from app.db.session import SessionLocal


class ResidualMemoryService:
    @staticmethod
    def _norm(value: Any) -> str:
        return str(value or "").strip().lower()

    @staticmethod
    def _pair_match(row: PredictionResidualRecord, team: str, opponent: str) -> bool:
        left = ResidualMemoryService._norm(row.team)
        right = ResidualMemoryService._norm(row.opponent)
        return (left == team and right == opponent) or (left == opponent and right == team)

    @staticmethod
    def _mean(values: list[float], default: float = 0.0) -> float:
        return float(mean(values)) if values else float(default)

    @classmethod
    def inference_context(cls, match_payload: dict[str, Any], lookback: int = 320) -> dict[str, Any]:
        sport = cls._norm(match_payload.get("sport") or "cricket")
        team = cls._norm(match_payload.get("team_a"))
        opponent = cls._norm(match_payload.get("team_b"))
        venue = cls._norm(match_payload.get("venue"))
        tournament = cls._norm(match_payload.get("tournament"))
        state = cls._norm(match_payload.get("state"))

        db = SessionLocal()
        try:
            try:
                rows = (
                    db.query(PredictionResidualRecord)
                    .filter(PredictionResidualRecord.sport == sport)
                    .order_by(PredictionResidualRecord.created_at.desc())
                    .limit(max(40, lookback))
                    .all()
                )
            except Exception:
                return {
                    "samples": 0,
                    "venue_bias_residual": 0.0,
                    "team_pair_residual": 0.0,
                    "recent_underprediction_rate": 0.0,
                    "recent_overprediction_rate": 0.0,
                    "combined_bias": 0.0,
                    "residual_shift_score": 0.0,
                    "source": "unavailable",
                }
        finally:
            db.close()

        if not rows:
            return {
                "samples": 0,
                "venue_bias_residual": 0.0,
                "team_pair_residual": 0.0,
                "recent_underprediction_rate": 0.0,
                "recent_overprediction_rate": 0.0,
                "combined_bias": 0.0,
                "residual_shift_score": 0.0,
                "source": "empty",
            }

        all_residuals = [float(row.residual_value) for row in rows]
        global_mean = cls._mean(all_residuals)

        venue_rows = [row for row in rows if cls._norm(row.venue) == venue]
        pair_rows = [row for row in rows if cls._pair_match(row, team=team, opponent=opponent)]
        tournament_rows = [row for row in rows if cls._norm(row.tournament) == tournament]
        state_rows = [row for row in rows if cls._norm(row.match_state) == state]

        venue_bias = cls._mean([float(row.residual_value) for row in venue_rows], default=global_mean)
        pair_bias = cls._mean([float(row.residual_value) for row in pair_rows], default=global_mean)
        tournament_bias = cls._mean([float(row.residual_value) for row in tournament_rows], default=global_mean)
        state_bias = cls._mean([float(row.residual_value) for row in state_rows], default=global_mean)

        combined_bias = (
            (0.4 * pair_bias)
            + (0.25 * venue_bias)
            + (0.2 * tournament_bias)
            + (0.15 * state_bias)
        )
        recent = rows[: min(30, len(rows))]
        under = [row for row in recent if float(row.residual_value) > 0.0]
        over = [row for row in recent if float(row.residual_value) < 0.0]

        recent_mean = cls._mean([float(row.residual_value) for row in recent], default=global_mean)
        residual_shift_score = abs(recent_mean - global_mean)

        return {
            "samples": int(len(rows)),
            "venue_bias_residual": round(float(venue_bias), 3),
            "team_pair_residual": round(float(pair_bias), 3),
            "tournament_bias_residual": round(float(tournament_bias), 3),
            "state_bias_residual": round(float(state_bias), 3),
            "recent_underprediction_rate": round(float(len(under) / max(1, len(recent))), 3),
            "recent_overprediction_rate": round(float(len(over) / max(1, len(recent))), 3),
            "combined_bias": round(float(combined_bias), 3),
            "residual_shift_score": round(float(residual_shift_score), 3),
            "source": "db",
            "updated_at": datetime.utcnow().isoformat(),
        }

    @classmethod
    def record(
        cls,
        *,
        match_payload: dict[str, Any],
        prediction_output: dict[str, Any],
        evaluation_summary: dict[str, Any],
    ) -> str | None:
        actual_value = evaluation_summary.get("actual_value")
        best_head_value = evaluation_summary.get("best_head_value")
        best_error = evaluation_summary.get("best_match_error")
        if not isinstance(actual_value, (int, float)):
            return None
        if not isinstance(best_head_value, (int, float)):
            return None
        if not isinstance(best_error, (int, float)):
            return None

        uncertainty = prediction_output.get("uncertainty") or {}
        metadata = prediction_output.get("metadata") or {}
        predictions = prediction_output.get("predictions") or []
        winner_index = evaluation_summary.get("winner_index")
        winner_scenario = None
        if isinstance(winner_index, int) and 0 <= winner_index < len(predictions):
            winner_scenario = predictions[winner_index].get("scenario")

        predicted_mean = float(uncertainty.get("mean_prediction") or 0.0)
        interval_low = float(uncertainty.get("interval_low") or predicted_mean)
        interval_high = float(uncertainty.get("interval_high") or predicted_mean)
        residual = float(actual_value) - predicted_mean

        db = SessionLocal()
        try:
            match_id = str(match_payload.get("match_id") or "")
            try:
                if match_id:
                    existing = (
                        db.query(PredictionResidualRecord)
                        .filter(PredictionResidualRecord.match_id == match_id)
                        .order_by(PredictionResidualRecord.created_at.desc())
                        .first()
                    )
                    if (
                        existing is not None
                        and abs(float(existing.predicted_mean) - predicted_mean) < 0.01
                        and abs(float(existing.actual_value) - float(actual_value)) < 0.01
                    ):
                        return existing.record_id

                record_id = f"res_{datetime.utcnow():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
                row = PredictionResidualRecord(
                    record_id=record_id,
                    match_id=match_id or None,
                    sport=str(match_payload.get("sport") or "cricket").lower(),
                    tournament=str(match_payload.get("tournament") or "") or None,
                    team=str(match_payload.get("team_a") or "") or None,
                    opponent=str(match_payload.get("team_b") or "") or None,
                    venue=str(match_payload.get("venue") or "") or None,
                    match_state=str(match_payload.get("state") or "") or None,
                    scenario=str(winner_scenario or "") or None,
                    predicted_mean=predicted_mean,
                    interval_low=interval_low,
                    interval_high=interval_high,
                    actual_value=float(actual_value),
                    best_head_value=float(best_head_value),
                    error_value=float(best_error),
                    residual_value=float(residual),
                    data_mode=str(metadata.get("data_mode") or "") or None,
                    metadata_json={
                        "model_mode": metadata.get("model_mode"),
                        "num_heads": metadata.get("num_heads"),
                        "scenario_probabilities": metadata.get("scenario_probabilities"),
                        "evaluation_summary": evaluation_summary,
                    },
                )
                db.add(row)
                db.commit()
                return record_id
            except Exception:
                db.rollback()
                return None
        finally:
            db.close()
