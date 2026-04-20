from __future__ import annotations

import importlib.util
import json
import logging
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import get_settings
from app.db.models import PredictionResidualRecord
from app.db.repositories.research_repository import ResearchRepository
from app.db.session import SessionLocal
from app.services.name_resolver_service import get_player_name_resolver
from app.services.providers.factory import get_live_provider
from app.services.residual_memory_service import ResidualMemoryService

logger = logging.getLogger(__name__)


class PredictionService:
    @staticmethod
    def _to_float(value: Any) -> float | None:
        try:
            casted = float(value)
        except (TypeError, ValueError):
            return None
        if np.isnan(casted):
            return None
        return casted

    @staticmethod
    def _finalize_prediction_contract(response: dict[str, Any]) -> dict[str, Any]:
        predictions = response.get("predictions")
        if isinstance(predictions, list):
            labels = ["Low", "Baseline", "High", "Aggressive"]
            winner_votes: list[str] = []
            for row in predictions:
                if not isinstance(row, dict):
                    continue
                winner = str(row.get("winner") or "").strip()
                if winner:
                    winner_votes.append(winner.lower())

            winner_agreement = None
            if winner_votes:
                winner_counts: dict[str, int] = {}
                for winner in winner_votes:
                    winner_counts[winner] = winner_counts.get(winner, 0) + 1
                winner_agreement = max(winner_counts.values()) / len(winner_votes)

            for idx, row in enumerate(predictions):
                if isinstance(row, dict):
                    scenario = str(row.get("scenario") or "").strip()
                    if not scenario:
                        scenario = labels[min(idx, len(labels) - 1)]
                    row["scenario"] = scenario
                    row["label"] = str(row.get("label") or scenario)

                    probability = PredictionService._to_float(row.get("scenario_probability"))
                    if probability is not None:
                        probability = float(np.clip(probability, 0.0, 1.0))
                        row["scenario_probability"] = round(probability, 2)

                    confidence = PredictionService._to_float(row.get("confidence"))
                    if confidence is None and probability is not None:
                        confidence = probability
                    if confidence is None and winner_agreement is not None:
                        confidence = winner_agreement
                    if confidence is not None:
                        row["confidence"] = round(float(np.clip(confidence, 0.0, 1.0)), 2)
            response["scenarios"] = predictions
        return response

    @classmethod
    def _build_forecast_summary(cls, response: dict[str, Any]) -> dict[str, Any] | None:
        predictions = response.get("predictions")
        match = response.get("match") if isinstance(response.get("match"), dict) else {}
        metadata = response.get("metadata") if isinstance(response.get("metadata"), dict) else {}
        if not isinstance(predictions, list) or not predictions:
            return None

        team_a = str(match.get("team_a") or "").strip()
        team_b = str(match.get("team_b") or "").strip()
        sport = str(match.get("sport") or "cricket").strip().lower()
        uncertainty = response.get("uncertainty") if isinstance(response.get("uncertainty"), dict) else {}
        spread = cls._to_float(uncertainty.get("spread")) or 0.0

        def _canonical_team(winner_raw: Any) -> str | None:
            raw = str(winner_raw or "").strip()
            if not raw:
                return None
            lower = raw.lower()
            if team_a and lower == team_a.lower():
                return team_a
            if team_b and lower == team_b.lower():
                return team_b
            return None

        scenario_weight_by_team: dict[str, float] = {}
        winner_vote_count: dict[str, int] = {}
        winner_vote_total = 0
        scenario_scores: list[float] = []
        for row in predictions:
            if not isinstance(row, dict):
                continue
            canonical = _canonical_team(row.get("winner"))
            if not canonical:
                continue
            weight = cls._to_float(row.get("scenario_probability"))
            if weight is None or weight <= 0:
                weight = cls._to_float(row.get("confidence"))
            if weight is None or weight <= 0:
                weight = 1.0
            scenario_weight_by_team[canonical] = scenario_weight_by_team.get(canonical, 0.0) + float(weight)
            winner_vote_count[canonical] = winner_vote_count.get(canonical, 0) + 1
            winner_vote_total += 1
            score = cls._to_float(row.get("score"))
            if score is None:
                team_a_score = cls._to_float(row.get("team_a_score"))
                team_b_score = cls._to_float(row.get("team_b_score"))
                if team_a_score is not None and team_b_score is not None:
                    score = (team_a_score + team_b_score) / 2.0
            if score is not None:
                scenario_scores.append(float(score))

        candidate_teams = [team for team in [team_a, team_b] if team]
        if candidate_teams:
            favored_team = max(candidate_teams, key=lambda team: scenario_weight_by_team.get(team, 0.0))
        elif scenario_weight_by_team:
            favored_team = max(scenario_weight_by_team, key=scenario_weight_by_team.get)
        else:
            favored_team = None
        interval_low = cls._to_float(uncertainty.get("interval_low"))
        interval_high = cls._to_float(uncertainty.get("interval_high"))
        if interval_low is None or interval_high is None:
            derived_scores: list[float] = []
            for row in predictions:
                if not isinstance(row, dict):
                    continue
                score = cls._to_float(row.get("score"))
                if score is None:
                    home = cls._to_float(row.get("home_goals"))
                    away = cls._to_float(row.get("away_goals"))
                    if home is not None and away is not None:
                        score = home + away
                if score is not None:
                    derived_scores.append(score)
            if derived_scores:
                interval_low = min(derived_scores)
                interval_high = max(derived_scores)
                if spread <= 0.0:
                    spread = float(interval_high - interval_low)

        if spread <= 0.0 and scenario_scores:
            spread = float(max(scenario_scores) - min(scenario_scores))

        batting_order_flip = False
        for row in predictions:
            if not isinstance(row, dict):
                continue
            team_a_first = row.get("team_a_first")
            team_b_first = row.get("team_b_first")
            if not isinstance(team_a_first, dict) or not isinstance(team_b_first, dict):
                continue
            a_winner = str(team_a_first.get("winner") or "").strip()
            b_winner = str(team_b_first.get("winner") or "").strip()
            if a_winner and b_winner and a_winner != b_winner:
                batting_order_flip = True
                break

        ensemble_uncertainty = metadata.get("ensemble_uncertainty") if isinstance(metadata.get("ensemble_uncertainty"), dict) else {}
        std_team_a = cls._to_float(ensemble_uncertainty.get("std_team_a"))
        std_team_b = cls._to_float(ensemble_uncertainty.get("std_team_b"))
        std_winner_prob = cls._to_float(ensemble_uncertainty.get("std_winner_prob"))
        disagreement_hint = cls._to_float(metadata.get("ensemble_disagreement_score"))
        ensemble_variance_component: float
        if std_team_a is not None and std_team_b is not None and std_winner_prob is not None:
            score_std_component = float(np.clip((std_team_a + std_team_b) / 28.0, 0.0, 1.0))
            winner_std_component = float(np.clip(std_winner_prob / 0.22, 0.0, 1.0))
            ensemble_variance_component = float(np.clip((score_std_component + winner_std_component) / 2.0, 0.0, 1.0))
        elif disagreement_hint is not None:
            ensemble_variance_component = float(np.clip(disagreement_hint, 0.0, 1.0))
        else:
            ensemble_variance_component = float(np.clip(float(np.var(np.asarray(scenario_scores, dtype=float))) / 140.0, 0.0, 1.0)) if scenario_scores else 0.0

        spread_threshold = 30.0 if sport == "cricket" else 3.0
        spread_component = float(np.clip((spread or 0.0) / spread_threshold, 0.0, 1.0))

        if scenario_weight_by_team:
            winner_weight_total = sum(scenario_weight_by_team.values())
            top_winner_weight = max(scenario_weight_by_team.values()) if winner_weight_total > 0 else 0.0
            winner_disagreement_component = float(np.clip(1.0 - (top_winner_weight / winner_weight_total), 0.0, 1.0))
        elif winner_vote_total > 0:
            top_vote = max(winner_vote_count.values())
            winner_disagreement_component = float(np.clip(1.0 - (float(top_vote) / float(winner_vote_total)), 0.0, 1.0))
        else:
            winner_disagreement_component = 0.5

        uncertainty_score = float(
            np.clip(
                (spread_component + winner_disagreement_component + ensemble_variance_component) / 3.0,
                0.0,
                1.0,
            )
        )
        favored_conf = float(np.clip(1.0 - uncertainty_score, 0.05, 0.95))

        win_probability = 0.5
        if favored_team and team_a and team_b:
            w_a = scenario_weight_by_team.get(team_a, 0.0)
            w_b = scenario_weight_by_team.get(team_b, 0.0)
            denom = w_a + w_b
            if denom > 0:
                win_probability = float(scenario_weight_by_team.get(favored_team, 0.0) / denom)
            else:
                win_probability = 0.5
        elif favored_team and scenario_weight_by_team.get(favored_team):
            win_probability = 1.0

        if (
            winner_disagreement_component >= 0.34
            or spread_component >= 0.62
            or ensemble_variance_component >= 0.62
        ):
            risk_level = "High"
            risk_explanation = "High uncertainty from wide scenario spread and unstable outcome signals."
        elif (
            winner_disagreement_component >= 0.16
            or spread_component >= 0.34
            or ensemble_variance_component >= 0.34
        ):
            risk_level = "Medium"
            risk_explanation = "Moderate uncertainty with visible variation across plausible outcomes."
        else:
            risk_level = "Low"
            risk_explanation = "Low uncertainty with tight score distribution and stable winner signals."

        if batting_order_flip:
            risk_explanation = risk_explanation.rstrip(".") + " Batting-order sensitivity remains important."

        risk_components = {
            "winner_disagreement": winner_disagreement_component,
            "spread": spread_component,
            "ensemble_variance": ensemble_variance_component,
        }
        dominant_risk = max(risk_components, key=risk_components.get)
        if dominant_risk == "winner_disagreement":
            key_risk = "Scenario winners disagree across plausible match paths"
        elif dominant_risk == "spread":
            key_risk = "Scenario scores span a wide range of outcomes"
        else:
            key_risk = "Ensemble members disagree on match direction"
        if batting_order_flip and dominant_risk != "winner_disagreement":
            key_risk = "Batting-order branches can still flip the likely winner"

        ranked_rows = sorted(
            [row for row in predictions if isinstance(row, dict)],
            key=lambda row: float(cls._to_float(row.get("scenario_probability")) or cls._to_float(row.get("confidence")) or 0.0),
            reverse=True,
        )
        primary = ranked_rows[0] if ranked_rows else {}
        if favored_team:
            favored_primary = next(
                (
                    row
                    for row in ranked_rows
                    if str(row.get("winner") or "").strip().lower() == str(favored_team).strip().lower()
                ),
                None,
            )
            if favored_primary is not None:
                primary = favored_primary
        primary_reasons = primary.get("reason") if isinstance(primary.get("reason"), list) else []
        top_reason = ""
        if primary_reasons:
            preferred = None
            for item in primary_reasons:
                if not isinstance(item, dict):
                    continue
                explanation = str(item.get("explanation") or "").strip()
                if not explanation:
                    continue
                if favored_team and favored_team.lower() in explanation.lower():
                    preferred = explanation
                    break
                if preferred is None:
                    preferred = explanation
            top_reason = preferred or ""

        edge = "slightly"
        if win_probability >= 0.74:
            edge = "strongly"
        elif win_probability >= 0.6:
            edge = "moderately"

        if not top_reason:
            top_reason = "recent form and matchup signals"
        other_team = ""
        if favored_team and team_a and team_b:
            other_team = team_b if favored_team.strip().lower() == team_a.strip().lower() else team_a
        if favored_team and other_team and other_team.lower() in top_reason.lower() and favored_team.lower() not in top_reason.lower():
            top_reason = ""

        primary_reason = top_reason.rstrip(".")
        if not primary_reason:
            if batting_order_flip:
                primary_reason = "stronger batting-order flexibility in key phases"
            elif uncertainty_score >= 0.45:
                primary_reason = "better control in volatile match phases"
            else:
                primary_reason = "more stable recent performance signals"
        if favored_team:
            favored_lower = favored_team.strip().lower()
            reason_lower = primary_reason.strip().lower()
            if favored_lower and reason_lower.startswith(favored_lower):
                primary_reason = primary_reason.strip()[len(favored_team) :].strip()
            for verb_prefix in [
                "is ",
                "are ",
                "has ",
                "have ",
                "show ",
                "shows ",
                "hold ",
                "holds ",
                "carry ",
                "carries ",
                "enter ",
                "enters ",
                "remain ",
                "remains ",
            ]:
                if primary_reason.lower().startswith(verb_prefix):
                    primary_reason = primary_reason[len(verb_prefix) :].strip()
                    break
            if not primary_reason:
                primary_reason = "more stable recent performance signals"

        if batting_order_flip:
            secondary_factor = "batting-order shifts can still flip the result"
        elif winner_disagreement_component >= 0.16:
            secondary_factor = "scenario disagreement keeps both sides live"
        elif risk_level == "High":
            secondary_factor = "wide score spread keeps both outcomes in play"
        else:
            secondary_factor = "small phase swings can still shift momentum"

        if favored_team:
            final_summary = (
                f"{favored_team} are {edge} favored due to {primary_reason}, "
                f"while {secondary_factor}."
            )
        else:
            final_summary = (
                f"No clear favorite: {primary_reason}, "
                f"while {secondary_factor}."
            )

        score_range = None
        if interval_low is not None and interval_high is not None:
            score_range = f"{round(float(interval_low))}-{round(float(interval_high))}"

        return {
            "favored_team": favored_team,
            "favored_team_confidence": round(float(np.clip(favored_conf, 0.0, 1.0)), 2),
            "win_probability": round(float(np.clip(win_probability, 0.0, 1.0)), 2),
            "predicted_band_low": round(float(interval_low), 2) if interval_low is not None else None,
            "predicted_band_high": round(float(interval_high), 2) if interval_high is not None else None,
            "expected_score_range": score_range,
            "key_risk": key_risk,
            "risk_level": risk_level,
            "risk_explanation": risk_explanation,
            "final_summary": final_summary,
        }

    @classmethod
    def _inject_forecast_summary(cls, response: dict[str, Any]) -> dict[str, Any]:
        summary = cls._build_forecast_summary(response)
        if summary is not None:
            response["forecast_summary"] = summary
        return response

    @staticmethod
    def _performance_summary_from_metrics(metrics: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(metrics, dict):
            return None
        accuracy = metrics.get("winner_accuracy")
        avg_error = (
            metrics.get("best_scenario_error")
            if metrics.get("best_scenario_error") is not None
            else metrics.get("mae_total")
        )
        in_range = (
            metrics.get("coverage")
            if metrics.get("coverage") is not None
            else metrics.get("conformal_coverage")
        )
        samples = metrics.get("samples")
        if accuracy is None and avg_error is None and in_range is None:
            return None
        accuracy_value = float(accuracy) if isinstance(accuracy, (int, float)) else None
        avg_error_value = float(avg_error) if isinstance(avg_error, (int, float)) else None
        in_range_value = float(in_range) if isinstance(in_range, (int, float)) else None
        reliability = PredictionService._performance_reliability(
            accuracy=accuracy_value,
            avg_error=avg_error_value,
            in_range_pct=in_range_value,
        )
        return {
            "reliability": reliability,
            "accuracy": round(float(accuracy), 2) if isinstance(accuracy, (int, float)) else None,
            "avg_error": round(float(avg_error), 2) if isinstance(avg_error, (int, float)) else None,
            "in_range_pct": round(float(in_range), 2) if isinstance(in_range, (int, float)) else None,
            "samples": int(float(samples)) if isinstance(samples, (int, float)) else None,
            "interpretation": PredictionService._performance_interpretation(
                accuracy=accuracy_value,
                avg_error=avg_error_value,
                in_range_pct=in_range_value,
            ),
        }

    @staticmethod
    def _performance_reliability(
        *,
        accuracy: float | None,
        avg_error: float | None,
        in_range_pct: float | None,
    ) -> str:
        if accuracy is None or avg_error is None or in_range_pct is None:
            return "Unknown"
        if accuracy >= 0.68 and avg_error <= 10.0 and in_range_pct >= 0.72:
            return "High"
        if accuracy >= 0.55 and avg_error <= 16.0 and in_range_pct >= 0.6:
            return "Moderate"
        return "Low"

    @staticmethod
    def _performance_interpretation(
        *,
        accuracy: float | None,
        avg_error: float | None,
        in_range_pct: float | None,
    ) -> str:
        reliability = PredictionService._performance_reliability(
            accuracy=accuracy,
            avg_error=avg_error,
            in_range_pct=in_range_pct,
        )
        if reliability == "High":
            return "Model shows strong reliability across winner prediction and score range coverage."
        if reliability == "Moderate":
            return "Model shows moderate reliability with reasonable range coverage."
        if reliability == "Low":
            return "Model is volatile on this sample and should be used with caution."
        if accuracy is not None and in_range_pct is not None and avg_error is not None:
            if accuracy >= 0.65 and in_range_pct >= 0.70 and avg_error <= 10.0:
                return "Model shows strong reliability across both score prediction and winner selection."
            if in_range_pct >= 0.65 and accuracy < 0.60:
                return "Model performs well in estimating score ranges but is moderately accurate on match winners."
            if accuracy >= 0.62 and avg_error <= 12.0:
                return "Model is reasonably reliable on both winner picks and score estimates."
        return "Model output is probabilistic; use scenarios as ranges, not certainties."

    @classmethod
    def _performance_summary_for_sport(cls, sport: str) -> dict[str, Any]:
        normalized_sport = str(sport or "cricket").strip().lower()
        db = SessionLocal()
        try:
            rows = (
                db.query(PredictionResidualRecord)
                .filter(PredictionResidualRecord.sport == normalized_sport)
                .order_by(PredictionResidualRecord.created_at.desc())
                .limit(500)
                .all()
            )
        except Exception:  # noqa: BLE001
            rows = []
        finally:
            db.close()

        if rows:
            errors = [float(row.error_value) for row in rows if isinstance(row.error_value, (int, float))]
            winner_flags: list[float] = []
            coverage_flags: list[float] = []
            for row in rows:
                payload = row.metadata_json if isinstance(row.metadata_json, dict) else {}
                eval_payload = payload.get("evaluation_summary") if isinstance(payload.get("evaluation_summary"), dict) else {}
                winner_correct = eval_payload.get("winner_correct")
                interval_covered = eval_payload.get("interval_covered")
                if isinstance(winner_correct, bool):
                    winner_flags.append(1.0 if winner_correct else 0.0)
                if isinstance(interval_covered, bool):
                    coverage_flags.append(1.0 if interval_covered else 0.0)
            summary = {
                "reliability": cls._performance_reliability(
                    accuracy=float(np.mean(winner_flags)) if winner_flags else None,
                    avg_error=float(np.mean(errors)) if errors else None,
                    in_range_pct=float(np.mean(coverage_flags)) if coverage_flags else None,
                ),
                "accuracy": round(float(np.mean(winner_flags)), 2) if winner_flags else None,
                "avg_error": round(float(np.mean(errors)), 2) if errors else None,
                "in_range_pct": round(float(np.mean(coverage_flags)), 2) if coverage_flags else None,
                "samples": int(len(rows)),
            }
            summary["interpretation"] = cls._performance_interpretation(
                accuracy=float(summary["accuracy"]) if isinstance(summary["accuracy"], (int, float)) else None,
                avg_error=float(summary["avg_error"]) if isinstance(summary["avg_error"], (int, float)) else None,
                in_range_pct=float(summary["in_range_pct"]) if isinstance(summary["in_range_pct"], (int, float)) else None,
            )
            if any(value is not None for key, value in summary.items() if key != "samples"):
                return summary

        artifacts = cls._status_artifacts()
        latest_eval = artifacts.get("latest_evaluation_summary")
        metrics = latest_eval.get("metrics") if isinstance(latest_eval, dict) else None
        from_metrics = cls._performance_summary_from_metrics(metrics if isinstance(metrics, dict) else None)
        if from_metrics is not None:
            return from_metrics
        return {
            "reliability": "Unknown",
            "accuracy": None,
            "avg_error": None,
            "in_range_pct": None,
            "samples": 0,
            "interpretation": cls._performance_interpretation(
                accuracy=None,
                avg_error=None,
                in_range_pct=None,
            ),
        }

    @classmethod
    def _inject_performance_summary(cls, response: dict[str, Any], match_payload: dict[str, Any]) -> dict[str, Any]:
        sport = str((response.get("match") or {}).get("sport") or match_payload.get("sport") or "cricket")
        response["performance_summary"] = cls._performance_summary_for_sport(sport)
        return response

    @staticmethod
    def _evaluation_summary_text(
        *,
        available: bool,
        winner_correct: bool | None,
        interval_covered: bool | None,
    ) -> str:
        if not available:
            return "Actual result not available for evaluation."
        if winner_correct is True and interval_covered is True:
            return "Model got the winner right and stayed inside the predicted range."
        if winner_correct is True and interval_covered is False:
            return "Model got the winner right but missed the predicted range."
        if winner_correct is False and interval_covered is True:
            return "Model missed the winner despite a close score fit."
        if winner_correct is False and interval_covered is False:
            return "Model missed both winner and predicted range."
        return "Evaluation is partially available for this match."

    @staticmethod
    def _resolve_actual_for_match(match_payload: dict[str, Any]) -> dict[str, Any]:
        sport = str(match_payload.get("sport") or "cricket").lower()
        if sport != "cricket":
            return {"available": False, "target": None, "target_type": None, "winner": None}

        settings = get_settings()
        match_id = str(match_payload.get("match_id") or "").strip()
        if not match_id:
            return {"available": False, "target": None, "target_type": "first_innings_score", "winner": None}

        paths = [
            settings.data_processed_dir / "matches.csv",
            settings.data_processed_dir / "match_feature_lookup.csv",
            settings.data_processed_dir / "model_features.csv",
        ]
        for path in paths:
            if not path.exists():
                continue
            try:
                import pandas as pd

                frame = pd.read_csv(path)
            except Exception:  # noqa: BLE001
                continue
            if "match_id" not in frame.columns:
                continue
            subset = frame[frame["match_id"] == match_id]
            if subset.empty:
                continue
            row = subset.iloc[0]
            def _to_float(value: Any) -> float | None:
                try:
                    casted = float(value)
                except (TypeError, ValueError):
                    return None
                if np.isnan(casted):
                    return None
                return casted
            for col in ["actual_first_innings_score", "target_score", "first_innings_total"]:
                if col in subset.columns:
                    value = row.get(col)
                    try:
                        target = float(value)
                    except (TypeError, ValueError):
                        continue
                    if np.isnan(target):
                        continue
                    return {
                        "available": True,
                        "target": target,
                        "target_type": "first_innings_score",
                        "source": str(path),
                        "winner": str(row.get("winner") or "").strip() or None,
                        "first_innings_team": str(row.get("first_innings_team") or "").strip() or None,
                        "second_innings_team": str(row.get("second_innings_team") or "").strip() or None,
                        "second_innings_total": _to_float(row.get("second_innings_total")),
                        "first_innings_total": _to_float(row.get("first_innings_total")) or target,
                    }
        return {"available": False, "target": None, "target_type": "first_innings_score", "winner": None}

    @staticmethod
    def _scenario_numeric_value(scenario: dict[str, Any], first_innings_team: str | None = None) -> float | None:
        if first_innings_team:
            for key in ["team_a_first", "team_b_first"]:
                branch = scenario.get(key)
                if not isinstance(branch, dict):
                    continue
                batting_team = str(branch.get("batting_team") or "").strip().lower()
                if batting_team and batting_team == first_innings_team.strip().lower():
                    if isinstance(branch.get("batting_score"), (int, float)):
                        return float(branch["batting_score"])
        if isinstance(scenario.get("score"), (int, float)):
            return float(scenario["score"])
        if isinstance(scenario.get("home_goals"), (int, float)) and isinstance(scenario.get("away_goals"), (int, float)):
            return float(scenario["home_goals"]) + float(scenario["away_goals"])
        return None

    @staticmethod
    def _actual_score_summary(actual: dict[str, Any]) -> str | None:
        first_team = str(actual.get("first_innings_team") or "").strip()
        second_team = str(actual.get("second_innings_team") or "").strip()
        first_total = actual.get("first_innings_total")
        second_total = actual.get("second_innings_total")
        if not first_team or not second_team:
            return None
        if not isinstance(first_total, (int, float)) or not isinstance(second_total, (int, float)):
            return None
        return f"{first_team} {int(round(float(first_total)))} vs {second_team} {int(round(float(second_total)))}"

    @staticmethod
    def _scenario_branch_details(
        scenario: dict[str, Any],
        first_innings_team: str | None,
    ) -> dict[str, Any] | None:
        if not isinstance(first_innings_team, str) or not first_innings_team.strip():
            return None
        target_team = first_innings_team.strip().lower()
        for key in ["team_a_first", "team_b_first"]:
            branch = scenario.get(key)
            if not isinstance(branch, dict):
                continue
            batting_team = str(branch.get("batting_team") or "").strip().lower()
            if batting_team != target_team:
                continue
            batting_score = branch.get("batting_score")
            chase_score = branch.get("chase_score")
            if not isinstance(batting_score, (int, float)) or not isinstance(chase_score, (int, float)):
                continue
            details = {
                "branch": key,
                "batting_team": str(branch.get("batting_team") or "").strip() or None,
                "batting_score": float(batting_score),
                "chase_score": float(chase_score),
                "winner": str(branch.get("winner") or "").strip() or None,
            }
            if key == "team_a_first":
                details["team_a_score"] = float(batting_score)
                details["team_b_score"] = float(chase_score)
            else:
                details["team_a_score"] = float(chase_score)
                details["team_b_score"] = float(batting_score)
            return details
        return None

    @staticmethod
    def _scenario_branch_scores(
        scenario: dict[str, Any],
        first_innings_team: str | None,
    ) -> tuple[float | None, float | None]:
        details = PredictionService._scenario_branch_details(scenario, first_innings_team)
        if details is not None:
            return float(details["batting_score"]), float(details["chase_score"])
        return None, None

    @classmethod
    def _evaluate_prediction(
        cls,
        prediction_output: dict[str, Any],
        actual_value: float,
        actual_second_innings: float | None = None,
        actual_winner: str | None = None,
        first_innings_team: str | None = None,
        actual_score_summary: str | None = None,
    ) -> dict[str, Any]:
        predictions = prediction_output.get("predictions") or []
        scenario_values: list[tuple[int, float]] = []
        scenario_pair_errors: list[tuple[int, float]] = []
        scenario_branch_details: dict[int, dict[str, Any]] = {}
        predicted_heads: list[str] = []
        for idx, row in enumerate(predictions):
            if isinstance(row, dict):
                if row.get("score") is not None:
                    predicted_heads.append(str(row.get("score")))
                elif row.get("scoreline") is not None:
                    predicted_heads.append(str(row.get("scoreline")))
            value = cls._scenario_numeric_value(
                row if isinstance(row, dict) else {},
                first_innings_team=first_innings_team,
            )
            if value is None:
                continue
            scenario_values.append((idx, value))
            if isinstance(actual_second_innings, (int, float)) and isinstance(row, dict):
                branch_details = cls._scenario_branch_details(
                    scenario=row,
                    first_innings_team=first_innings_team,
                )
                if branch_details is not None:
                    scenario_branch_details[idx] = branch_details
                    pred_first = float(branch_details["batting_score"])
                    pred_second = float(branch_details["chase_score"])
                else:
                    pred_first, pred_second = None, None
                if isinstance(pred_first, (int, float)) and isinstance(pred_second, (int, float)):
                    pair_rmse = float(
                        np.sqrt(
                            (
                                (float(pred_first) - float(actual_value)) ** 2
                                + (float(pred_second) - float(actual_second_innings)) ** 2
                            )
                            / 2.0
                        )
                    )
                    scenario_pair_errors.append((idx, pair_rmse))

        if not scenario_values:
            return {
                "actual_value": float(actual_value),
                "target_type": "first_innings_score",
                "available": False,
                "message": "No numeric prediction heads available for evaluation.",
                "evaluation_summary": "Evaluation unavailable: no numeric scenario output.",
            }

        if scenario_pair_errors:
            winner_idx, best_error = min(scenario_pair_errors, key=lambda item: float(item[1]))
            winner_value = dict(scenario_values).get(winner_idx, float(actual_value))
        else:
            winner_idx, winner_value = min(
                scenario_values,
                key=lambda item: abs(float(item[1]) - float(actual_value)),
            )
            best_error = abs(float(actual_value) - float(winner_value))
        uncertainty = prediction_output.get("uncertainty") or {}
        interval_low = float(uncertainty.get("interval_low") or winner_value)
        interval_high = float(uncertainty.get("interval_high") or winner_value)
        center_prediction = float(uncertainty.get("mean_prediction") or np.mean([value for _, value in scenario_values]))
        scenario_row = predictions[winner_idx] if 0 <= winner_idx < len(predictions) else {}
        predicted_winner = None
        selected_branch = scenario_branch_details.get(winner_idx)
        if isinstance(scenario_row, dict):
            a_first = scenario_row.get("team_a_first")
            b_first = scenario_row.get("team_b_first")
            if isinstance(selected_branch, dict):
                predicted_winner = selected_branch.get("winner")
            if predicted_winner is None:
                if isinstance(a_first, dict):
                    predicted_winner = str(a_first.get("winner") or "").strip() or None
                elif isinstance(scenario_row.get("likely_result"), str):
                    likely = str(scenario_row.get("likely_result") or "").strip()
                    if likely.lower().endswith(" win"):
                        predicted_winner = likely[:-4].strip() or None
                    elif likely.lower() == "draw":
                        predicted_winner = "Draw"

        winner_correct = None
        if isinstance(actual_winner, str) and actual_winner.strip() and isinstance(predicted_winner, str):
            winner_correct = predicted_winner.strip().lower() == actual_winner.strip().lower()

        team_a_error = None
        team_b_error = None
        if isinstance(actual_second_innings, (int, float)) and isinstance(selected_branch, dict):
            pred_team_a = selected_branch.get("team_a_score")
            pred_team_b = selected_branch.get("team_b_score")
            if isinstance(pred_team_a, (int, float)) and isinstance(pred_team_b, (int, float)):
                actual_first = float(actual_value)
                actual_second = float(actual_second_innings)
                if str(selected_branch.get("branch")) == "team_a_first":
                    actual_team_a = actual_first
                    actual_team_b = actual_second
                else:
                    actual_team_b = actual_first
                    actual_team_a = actual_second
                team_a_error = round(abs(float(pred_team_a) - float(actual_team_a)), 3)
                team_b_error = round(abs(float(pred_team_b) - float(actual_team_b)), 3)

        interval_covered = bool(interval_low <= float(actual_value) <= interval_high)
        return {
            "available": True,
            "actual_value": round(float(actual_value), 3),
            "target_type": "first_innings_score",
            "actual_first_innings_team": first_innings_team,
            "winner_index": int(winner_idx),
            "winner_scenario": scenario_row.get("scenario"),
            "best_matching_scenario": scenario_row.get("scenario"),
            "best_matching_branch": selected_branch.get("branch") if isinstance(selected_branch, dict) else None,
            "best_head_value": round(float(winner_value), 3),
            "best_match_error": round(float(best_error), 3),
            "best_match_error_method": "pair_rmse" if scenario_pair_errors else "first_innings_abs",
            "center_prediction": round(center_prediction, 3),
            "center_error": round(abs(float(actual_value) - center_prediction), 3),
            "interval_low": round(interval_low, 3),
            "interval_high": round(interval_high, 3),
            "interval_covered": interval_covered,
            "actual_winner": actual_winner,
            "predicted_winner": predicted_winner,
            "winner_correct": winner_correct,
            "team_a_score_error": team_a_error,
            "team_b_score_error": team_b_error,
            "actual_score_summary": actual_score_summary,
            "predicted_heads": predicted_heads,
            "evaluation_summary": cls._evaluation_summary_text(
                available=True,
                winner_correct=winner_correct,
                interval_covered=interval_covered,
            ),
        }

    @staticmethod
    def _player_name_resolver():
        settings = get_settings()
        return get_player_name_resolver(str(settings.data_processed_dir))

    @classmethod
    def _normalize_player_name(
        cls,
        payload: dict[str, Any] | None,
        *,
        sport: str,
        team_fallback: str | None,
        seed: str,
    ) -> dict[str, Any] | None:
        if payload is None:
            return None
        block = dict(payload)
        resolver = cls._player_name_resolver()
        resolved_name, replaced = resolver.resolve(
            name=block.get("name"),
            sport=sport,
            team=str(block.get("team") or team_fallback or ""),
            role=str(block.get("role") or ""),
            seed=seed,
        )
        block["name"] = resolved_name
        return block

    @classmethod
    def _normalize_player_list(
        cls,
        players: list[dict[str, Any]] | None,
        *,
        sport: str,
        default_team: str,
        seed_prefix: str,
    ) -> list[dict[str, Any]]:
        if not isinstance(players, list):
            return []
        normalized: list[dict[str, Any]] = []
        for idx, payload in enumerate(players):
            row = cls._normalize_player_name(
                payload if isinstance(payload, dict) else None,
                sport=sport,
                team_fallback=default_team,
                seed=f"{seed_prefix}|{idx}",
            )
            if row is not None:
                normalized.append(row)
        return normalized

    @classmethod
    def _normalize_prediction_names(cls, response: dict[str, Any], match_payload: dict[str, Any]) -> dict[str, Any]:
        sport = str((response.get("match") or {}).get("sport") or match_payload.get("sport") or "cricket")
        team_a = str((response.get("match") or {}).get("team_a") or match_payload.get("team_a") or "")
        team_b = str((response.get("match") or {}).get("team_b") or match_payload.get("team_b") or "")

        response["best_player"] = cls._normalize_player_name(
            response.get("best_player"),
            sport=sport,
            team_fallback=team_a,
            seed=f"{sport}|best_player|{team_a}|{team_b}",
        )
        response["best_bowler"] = cls._normalize_player_name(
            response.get("best_bowler"),
            sport=sport,
            team_fallback=team_b,
            seed=f"{sport}|best_bowler|{team_a}|{team_b}",
        )
        response["man_of_the_match"] = cls._normalize_player_name(
            response.get("man_of_the_match"),
            sport=sport,
            team_fallback=team_a,
            seed=f"{sport}|mom|{team_a}|{team_b}",
        )

        players = response.get("players")
        if isinstance(players, dict):
            normalized_players = dict(players)
            normalized_players["top_batsmen"] = cls._normalize_player_list(
                players.get("top_batsmen"),
                sport=sport,
                default_team=team_a,
                seed_prefix=f"{sport}|top_batsmen|{team_a}|{team_b}",
            )
            normalized_players["top_bowlers"] = cls._normalize_player_list(
                players.get("top_bowlers"),
                sport=sport,
                default_team=team_b,
                seed_prefix=f"{sport}|top_bowlers|{team_a}|{team_b}",
            )
            normalized_players["top_match_impact"] = cls._normalize_player_list(
                players.get("top_match_impact"),
                sport=sport,
                default_team=team_a,
                seed_prefix=f"{sport}|top_match_impact|{team_a}|{team_b}",
            )
            normalized_players["top_goal_scorers"] = cls._normalize_player_list(
                players.get("top_goal_scorers"),
                sport=sport,
                default_team=team_a,
                seed_prefix=f"{sport}|top_goal_scorers|{team_a}|{team_b}",
            )
            normalized_players["top_standout"] = cls._normalize_player_list(
                players.get("top_standout"),
                sport=sport,
                default_team=team_b,
                seed_prefix=f"{sport}|top_standout|{team_a}|{team_b}",
            )
            response["players"] = normalized_players

        return response

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None
        if isinstance(payload, dict):
            return payload
        return None

    @staticmethod
    def _active_artifact_snapshot() -> dict[str, Any]:
        db = SessionLocal()
        try:
            row = ResearchRepository.latest_active_model_artifact(db=db)
            if row is None:
                return {}
            return {
                "model_version": row.version,
                "active_experiment_id": row.experiment_id,
                "artifact_id": row.artifact_id,
                "checkpoint_path": row.checkpoint_path,
            }
        except Exception:  # noqa: BLE001
            return {}
        finally:
            db.close()

    @staticmethod
    def _status_artifacts() -> dict[str, Any]:
        settings = get_settings()
        checkpoint = settings.checkpoint_path
        checkpoint_updated_at = None
        if checkpoint.exists():
            checkpoint_updated_at = datetime.fromtimestamp(
                checkpoint.stat().st_mtime,
                tz=timezone.utc,
            ).isoformat()

        latest_eval = PredictionService._read_json(settings.latest_evaluation_summary_path)
        if latest_eval is None:
            raw_eval = PredictionService._read_json(settings.ml_artifacts_dir / "evaluation_metrics.json")
            if raw_eval is not None:
                latest_eval = {
                    "generated_at": None,
                    "source": "ml.evaluate",
                    "metrics": raw_eval,
                }
        training_run = PredictionService._read_json(settings.training_run_path)
        artifact_db = PredictionService._active_artifact_snapshot()

        model_version = artifact_db.get("model_version")
        if model_version is None and training_run is not None:
            run_id = str(training_run.get("run_id") or "")
            if run_id:
                model_version = run_id

        return {
            "model_version": model_version,
            "active_experiment_id": artifact_db.get("active_experiment_id"),
            "latest_run_id": (training_run or {}).get("run_id"),
            "checkpoint_updated_at": checkpoint_updated_at,
            "encoder_type": (training_run or {}).get("encoder_type"),
            "patch_encoder": (training_run or {}).get("patch_encoder"),
            "conformal_calibration": (training_run or {}).get("conformal_calibration"),
            "latest_evaluation_summary": latest_eval,
        }

    @staticmethod
    def _read_live_sync_status() -> dict[str, Any] | None:
        settings = get_settings()
        return PredictionService._read_json(settings.live_sync_status_path)

    @staticmethod
    def _write_live_sync_status(payload: dict[str, Any]) -> None:
        settings = get_settings()
        settings.live_sync_status_path.parent.mkdir(parents=True, exist_ok=True)
        settings.live_sync_status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _provider_health_snapshot() -> dict[str, Any]:
        provider = get_live_provider()
        try:
            payload = provider.healthcheck()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Live provider healthcheck failed: %s", exc)
            return {
                "provider": getattr(provider, "provider_name", "unknown"),
                "status": "unavailable",
                "error": str(exc),
                "source": "error",
            }
        return payload if isinstance(payload, dict) else {"provider": "unknown", "status": "unavailable"}

    @staticmethod
    def _freshness_summary(freshness_seconds: Any) -> str:
        if freshness_seconds is None:
            return "unknown"
        try:
            seconds = int(float(freshness_seconds))
        except (TypeError, ValueError):
            return "unknown"
        if seconds < 60:
            return "just now"
        if seconds < 3600:
            return f"{seconds // 60} min ago"
        return f"{seconds // 3600} h ago"

    @classmethod
    def _resolve_live_context(cls, match_payload: dict[str, Any]) -> dict[str, Any]:
        sport = str(match_payload.get("sport", "cricket")).lower()
        if sport != "cricket":
            return {
                "data_mode": "HISTORICAL",
                "provider_used": "sport-local",
                "last_refresh_time": None,
                "freshness_seconds": None,
                "freshness_summary": "historical",
                "live_context": {},
            }

        provider = get_live_provider()
        try:
            context = provider.fetch_match_context(
                match_id=match_payload.get("match_id"),
                team_a=match_payload.get("team_a"),
                team_b=match_payload.get("team_b"),
                match_date=match_payload.get("match_date"),
                tournament=match_payload.get("tournament"),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Live context fetch failed; historical fallback applied: %s", exc)
            return {
                "data_mode": "HISTORICAL",
                "provider_used": getattr(provider, "provider_name", "unknown"),
                "last_refresh_time": None,
                "freshness_seconds": None,
                "freshness_summary": "historical",
                "live_context": {},
                "error": str(exc),
            }

        context = context if isinstance(context, dict) else {}
        source = str(context.get("source") or "historical").lower()
        provider_name = str(context.get("provider") or getattr(provider, "provider_name", "unknown")).lower()
        raw_features = context.get("features")
        live_features: dict[str, float] = {}
        if isinstance(raw_features, dict):
            for key, value in raw_features.items():
                if not isinstance(value, (int, float)):
                    continue
                value_float = float(value)
                if not np.isfinite(value_float):
                    continue
                live_features[str(key)] = value_float

        if "mock" in source or "mock" in provider_name:
            live_features = {}
            data_mode = "HISTORICAL"
            source = "historical"
        elif live_features:
            if source == "live":
                data_mode = "LIVE"
            elif source.startswith("cache"):
                data_mode = "HYBRID"
            else:
                data_mode = "HYBRID"
        else:
            data_mode = "HISTORICAL"

        freshness_seconds = context.get("freshness_seconds")
        return {
            "data_mode": data_mode,
            "provider_used": str(context.get("provider") or getattr(provider, "provider_name", "unknown")),
            "last_refresh_time": context.get("updated_at"),
            "freshness_seconds": freshness_seconds,
            "freshness_summary": cls._freshness_summary(freshness_seconds),
            "live_context": live_features,
            "source": source,
            "live_summary": context.get("summary"),
        }

    @classmethod
    def _inject_data_metadata(
        cls,
        response: dict[str, Any],
        live_resolution: dict[str, Any],
    ) -> dict[str, Any]:
        metadata = dict(response.get("metadata") or {})
        model_mode = str(metadata.get("model_mode") or "unknown").lower()
        data_mode = str(live_resolution.get("data_mode") or "HISTORICAL").upper()
        if data_mode == "HISTORICAL" and "fallback" in model_mode:
            data_mode = "FALLBACK"

        metadata.update(
            {
                "data_mode": data_mode,
                "provider_used": live_resolution.get("provider_used"),
                "last_refresh_time": live_resolution.get("last_refresh_time"),
                "freshness_seconds": live_resolution.get("freshness_seconds"),
                "freshness_summary": live_resolution.get("freshness_summary"),
                "live_context_available": bool(live_resolution.get("live_context")),
            }
        )
        if live_resolution.get("live_summary"):
            metadata["live_summary"] = live_resolution.get("live_summary")
        response["metadata"] = metadata
        return response

    @staticmethod
    def _inject_residual_metadata(response: dict[str, Any], residual_context: dict[str, Any]) -> dict[str, Any]:
        metadata = dict(response.get("metadata") or {})
        metadata["residual_context"] = residual_context
        response["metadata"] = metadata
        return response

    @staticmethod
    def _scenario_probabilities(values: list[float], temperature: float = 12.0) -> list[float]:
        if not values:
            return []
        arr = np.asarray(values, dtype=float)
        centered = arr - np.mean(arr)
        logits = np.exp(-np.abs(centered) / max(0.5, float(temperature)))
        weights = logits / np.sum(logits)
        return [float(round(w, 6)) for w in weights]

    @staticmethod
    def _confidence_curve(num_rows: int, probabilities: list[float], model_base: float) -> list[float]:
        if num_rows <= 0:
            return []
        midpoint = (num_rows - 1) / 2.0
        rows: list[float] = []
        for idx in range(num_rows):
            distance_penalty = 0.15 * (abs(idx - midpoint) / max(1.0, midpoint))
            prob_boost = (probabilities[idx] if idx < len(probabilities) else 0.0) * 0.26
            confidence = model_base - distance_penalty + prob_boost
            rows.append(float(round(np.clip(confidence, 0.35, 0.93), 2)))
        return rows

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_predictor() -> Any:
        settings = get_settings()
        repo_root = settings.repo_root
        if str(repo_root) not in sys.path:
            sys.path.append(str(repo_root))

        try:
            from ml.inference import LiveMatchPredictor

            predictor = LiveMatchPredictor(
                checkpoint_path=settings.checkpoint_path,
                manifest_path=settings.manifest_path,
                num_heads=settings.predictor_num_heads,
            )
            mode = "TRAINED MODEL" if getattr(predictor, "model_loaded", False) else "FALLBACK"
            logger.info("Loaded ML predictor from %s", settings.checkpoint_path)
            logger.info("Running in %s mode", mode)
            return predictor
        except Exception as exc:  # noqa: BLE001
            logger.warning("Running in FALLBACK mode (predictor initialization failed): %s", exc)
            return None

    @classmethod
    def runtime_mode(cls) -> str:
        predictor = cls._load_predictor()
        if predictor is None:
            return "FALLBACK"
        if getattr(predictor, "model_loaded", False):
            return "TRAINED MODEL"
        return "FALLBACK"

    @classmethod
    def predict(cls, match_payload: dict[str, Any], k: int) -> dict[str, Any]:
        live_resolution = cls._resolve_live_context(match_payload)
        enriched_payload = dict(match_payload)
        if live_resolution.get("live_context"):
            enriched_payload["live_context"] = live_resolution.get("live_context")
            enriched_payload["live_recency_weight"] = get_settings().live_feature_recency_weight
        residual_context = ResidualMemoryService.inference_context(enriched_payload)
        enriched_payload["residual_context"] = residual_context

        predictor = cls._load_predictor()
        if predictor is None:
            fallback = cls._heuristic_fallback(enriched_payload, k)
            fallback = cls._normalize_prediction_names(fallback, enriched_payload)
            fallback = cls._inject_data_metadata(fallback, live_resolution)
            fallback = cls._inject_residual_metadata(fallback, residual_context)
            fallback = cls._finalize_prediction_contract(fallback)
            fallback = cls._inject_forecast_summary(fallback)
            fallback = cls._inject_performance_summary(fallback, enriched_payload)
            cls._record_residual_if_available(enriched_payload, fallback)
            return fallback

        try:
            response = predictor.predict(match_payload=enriched_payload, k=k)
            response = cls._normalize_prediction_names(response, enriched_payload)
            response = cls._inject_data_metadata(response, live_resolution)
            response = cls._inject_residual_metadata(response, residual_context)
            response = cls._finalize_prediction_contract(response)
            response = cls._inject_forecast_summary(response)
            response = cls._inject_performance_summary(response, enriched_payload)
            cls._record_residual_if_available(enriched_payload, response)
            return response
        except Exception as exc:  # noqa: BLE001
            logger.exception("Predictor runtime failure. Falling back to data-driven fallback mode: %s", exc)
            fallback = cls._heuristic_fallback(enriched_payload, k)
            fallback = cls._normalize_prediction_names(fallback, enriched_payload)
            fallback = cls._inject_data_metadata(fallback, live_resolution)
            fallback = cls._inject_residual_metadata(fallback, residual_context)
            fallback = cls._finalize_prediction_contract(fallback)
            fallback = cls._inject_forecast_summary(fallback)
            fallback = cls._inject_performance_summary(fallback, enriched_payload)
            cls._record_residual_if_available(enriched_payload, fallback)
            return fallback

    @classmethod
    def model_status(cls) -> dict[str, Any]:
        settings = get_settings()
        predictor = cls._load_predictor()
        runtime_mode = cls.runtime_mode()
        artifact_status = cls._status_artifacts()
        provider_status = cls._provider_health_snapshot()
        live_sync_status = cls._read_live_sync_status() or {}
        provider_health = str(provider_status.get("status") or "unavailable").lower()
        if provider_health == "ok":
            data_mode = "LIVE"
        elif provider_health == "degraded":
            data_mode = "HYBRID"
        elif live_sync_status:
            data_mode = "HISTORICAL"
        else:
            data_mode = "FALLBACK"

        model_mode = "TRAINED" if runtime_mode == "TRAINED MODEL" else "FALLBACK"
        provider_name = str(provider_status.get("provider") or "unknown")
        last_update = (
            provider_status.get("updated_at")
            or live_sync_status.get("refreshed_at")
            or artifact_status.get("checkpoint_updated_at")
        )
        healthy = bool(
            model_mode in {"TRAINED", "FALLBACK"}
            and provider_health in {"ok", "degraded", "unavailable"}
        )

        artifact_paths = {
            "checkpoint_path": str(settings.checkpoint_path),
            "manifest_path": str(settings.manifest_path),
            "training_run_path": str(settings.training_run_path),
            "latest_evaluation_summary_path": str(settings.latest_evaluation_summary_path),
            "live_cache_path": str(settings.live_cache_path),
            "live_sync_status_path": str(settings.live_sync_status_path),
        }
        return {
            "model": model_mode.lower(),
            "data": data_mode.lower(),
            "provider": provider_name,
            "heads": int(getattr(predictor, "num_heads", settings.predictor_num_heads)),
            "last_update": last_update,
            "healthy": healthy,
            "model_mode": model_mode,
            "runtime_mode": runtime_mode,
            "data_mode": data_mode,
            "checkpoint_exists": settings.checkpoint_path.exists(),
            "torch_available": importlib.util.find_spec("torch") is not None,
            "num_heads": int(getattr(predictor, "num_heads", settings.predictor_num_heads)),
            "model_version": artifact_status.get("model_version"),
            "checkpoint_updated_at": artifact_status.get("checkpoint_updated_at"),
            "active_experiment_id": artifact_status.get("active_experiment_id"),
            "latest_run_id": artifact_status.get("latest_run_id"),
            "encoder_type": str(
                getattr(predictor, "encoder_type", artifact_status.get("encoder_type") or "mlp")
            ),
            "encoder_config": getattr(
                predictor,
                "patch_encoder_config",
                artifact_status.get("patch_encoder") or {},
            ),
            "calibration": getattr(
                predictor,
                "conformal_calibration",
                artifact_status.get("conformal_calibration"),
            ),
            "latest_evaluation_summary": artifact_status.get("latest_evaluation_summary"),
            "provider_status": provider_status,
            "last_live_sync_time": live_sync_status.get("refreshed_at"),
            "live_sync_status": live_sync_status,
            "artifact_paths": artifact_paths,
            "sports_supported": ["cricket", "football"],
            "active_provider": f"{settings.data_provider} + {settings.realtime_provider}",
            "training_job_mode": settings.training_job_mode,
            "notes": "Fallback remains active when trained artifacts are unavailable.",
        }

    @classmethod
    def system_status(cls) -> dict[str, Any]:
        model_status = cls.model_status()
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": model_status,
            "provider": model_status.get("provider_status"),
            "live_sync": model_status.get("live_sync_status"),
        }

    @classmethod
    def refresh_live_data(cls, sport: str = "cricket", tournament: str | None = None) -> dict[str, Any]:
        provider = get_live_provider()
        refreshed_at = datetime.now(timezone.utc).isoformat()
        if sport.strip().lower() != "cricket":
            payload = {
                "status": "ignored",
                "provider": getattr(provider, "provider_name", "unknown"),
                "sport": sport,
                "tournament": tournament,
                "refreshed_at": refreshed_at,
                "message": "Live refresh currently implemented for cricket provider path.",
            }
            cls._write_live_sync_status(payload)
            return payload

        try:
            live_rows = provider.fetch_live_matches(tournament=tournament, limit=40)
            upcoming_rows = provider.fetch_upcoming_matches(tournament=tournament, limit=40)
            health = provider.healthcheck()
            payload = {
                "status": "ok",
                "provider": getattr(provider, "provider_name", "unknown"),
                "sport": "cricket",
                "tournament": tournament,
                "refreshed_at": refreshed_at,
                "live_matches": len(live_rows),
                "upcoming_matches": len(upcoming_rows),
                "provider_health": health,
            }
            cls._write_live_sync_status(payload)
            return payload
        except Exception as exc:  # noqa: BLE001
            payload = {
                "status": "failed",
                "provider": getattr(provider, "provider_name", "unknown"),
                "sport": "cricket",
                "tournament": tournament,
                "refreshed_at": refreshed_at,
                "error": str(exc),
            }
            cls._write_live_sync_status(payload)
            return payload

    @classmethod
    def _record_residual_if_available(cls, match_payload: dict[str, Any], prediction_output: dict[str, Any]) -> None:
        actual = cls._resolve_actual_for_match(match_payload)
        if not actual.get("available") or not isinstance(actual.get("target"), (int, float)):
            return
        summary = cls._evaluate_prediction(
            prediction_output=prediction_output,
            actual_value=float(actual["target"]),
            actual_second_innings=actual.get("second_innings_total"),
            actual_winner=actual.get("winner"),
            first_innings_team=actual.get("first_innings_team"),
            actual_score_summary=cls._actual_score_summary(actual),
        )
        summary["target_type"] = actual.get("target_type", "first_innings_score")
        try:
            ResidualMemoryService.record(
                match_payload=match_payload,
                prediction_output=prediction_output,
                evaluation_summary=summary,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Residual memory record failed: %s", exc)

    @classmethod
    def evaluate_match(cls, match_payload: dict[str, Any], k: int) -> dict[str, Any]:
        prediction = cls.predict(match_payload=match_payload, k=k)
        actual = cls._resolve_actual_for_match(match_payload)
        if actual.get("available") and isinstance(actual.get("target"), (int, float)):
            evaluation = cls._evaluate_prediction(
                prediction_output=prediction,
                actual_value=float(actual["target"]),
                actual_second_innings=actual.get("second_innings_total"),
                actual_winner=actual.get("winner"),
                first_innings_team=actual.get("first_innings_team"),
                actual_score_summary=cls._actual_score_summary(actual),
            )
            evaluation["target_type"] = actual.get("target_type", "first_innings_score")
            evaluation["source"] = actual.get("source")
        else:
            evaluation = {
                "available": False,
                "target_type": actual.get("target_type", "first_innings_score"),
                "message": "Actual result not available for this match.",
                "evaluation_summary": cls._evaluation_summary_text(
                    available=False,
                    winner_correct=None,
                    interval_covered=None,
                ),
            }
        return {
            "match_id": str(match_payload.get("match_id") or ""),
            "prediction": prediction,
            "evaluation": evaluation,
        }

    @staticmethod
    @lru_cache(maxsize=1)
    def _cricket_history_frame():
        settings = get_settings()
        path = settings.data_processed_dir / "matches.csv"
        if not path.exists():
            return None
        try:
            import pandas as pd

            frame = pd.read_csv(path)
        except Exception:  # noqa: BLE001
            return None
        required = {
            "match_id",
            "tournament",
            "team_a",
            "team_b",
            "venue",
            "winner",
            "first_innings_team",
            "first_innings_total",
            "second_innings_team",
            "second_innings_total",
        }
        if not required.issubset(frame.columns):
            return None
        return frame

    @staticmethod
    @lru_cache(maxsize=1)
    def _football_history_frame():
        settings = get_settings()
        path = settings.data_processed_dir / "football_matches.csv"
        if not path.exists():
            return None
        try:
            import pandas as pd

            frame = pd.read_csv(path)
        except Exception:  # noqa: BLE001
            return None
        required = {"match_id", "tournament", "team_a", "team_b", "home_goals", "away_goals"}
        if not required.issubset(frame.columns):
            return None
        return frame

    @staticmethod
    def _safe_mean(values: Any, default: float) -> float:
        try:
            arr = np.asarray(values, dtype=float)
        except Exception:  # noqa: BLE001
            return float(default)
        if arr.size == 0:
            return float(default)
        return float(np.nanmean(arr))

    @staticmethod
    def _enforce_min_gap(values: list[float], min_gap: float) -> list[float]:
        if not values:
            return values
        ordered = sorted(float(v) for v in values)
        for idx in range(1, len(ordered)):
            if ordered[idx] - ordered[idx - 1] < min_gap:
                ordered[idx] = ordered[idx - 1] + min_gap
        return ordered

    @classmethod
    def _quantile_heads(
        cls,
        samples: Any,
        *,
        k: int,
        default_center: float,
        low: float,
        high: float,
        min_gap: float,
    ) -> list[float]:
        try:
            arr = np.asarray(samples, dtype=float)
            arr = arr[~np.isnan(arr)]
        except Exception:  # noqa: BLE001
            arr = np.asarray([], dtype=float)
        if arr.size == 0:
            arr = np.asarray([default_center - 12.0, default_center, default_center + 12.0], dtype=float)
        if k <= 1:
            quantiles = np.asarray([0.5], dtype=float)
        elif k == 2:
            quantiles = np.asarray([0.28, 0.78], dtype=float)
        elif k == 3:
            quantiles = np.asarray([0.16, 0.5, 0.86], dtype=float)
        elif k == 4:
            quantiles = np.asarray([0.1, 0.42, 0.74, 0.93], dtype=float)
        else:
            quantiles = np.linspace(0.08, 0.93, max(2, k))
        heads = [float(x) for x in np.quantile(arr, quantiles)]
        heads = cls._enforce_min_gap(heads, min_gap=min_gap)
        clipped = np.clip(np.asarray(heads, dtype=float), low, high)
        return [float(round(x)) for x in clipped.tolist()]

    @staticmethod
    def _reason_signal(
        *,
        feature: str,
        value: float | str,
        baseline: float | str | None,
        impact: str,
        unit: str | None,
        label: str,
    ) -> dict[str, Any]:
        delta = None
        if isinstance(value, (int, float)) and isinstance(baseline, (int, float)):
            delta = round(float(value) - float(baseline), 2)
        return {
            "feature": feature,
            "value": round(float(value), 2) if isinstance(value, (int, float)) else value,
            "baseline": round(float(baseline), 2) if isinstance(baseline, (int, float)) else baseline,
            "delta": delta,
            "unit": unit,
            "impact": impact,
            "explanation": label,
        }

    @classmethod
    def _fallback_players(
        cls,
        *,
        sport: str,
        tournament: str,
        team_a: str,
        team_b: str,
    ) -> dict[str, Any]:
        from app.services.catalog_service import CatalogService

        def _as_player(row: dict[str, Any], role_default: str) -> dict[str, Any]:
            name = str(row.get("player") or row.get("name") or "").strip() or "Unavailable"
            role = str(row.get("role") or role_default)
            team = str(row.get("team") or team_a)
            score = float(row.get("rank_score", row.get("impact_score", 0.0)) or 0.0)
            recent_runs = float(row.get("recent_runs", 0.0) or 0.0)
            recent_wickets = float(row.get("recent_wickets", 0.0) or 0.0)
            goals_last5 = float(row.get("goals_last5", 0.0) or 0.0)
            xg_per90 = float(row.get("xg_per90", 0.0) or 0.0)
            reasons: list[dict[str, Any]] = []
            role_key = role.lower()
            if role_key in {"batsman", "goal_scorer"}:
                reasons.append(
                    cls._reason_signal(
                        feature="recent_scoring",
                        value=max(recent_runs, goals_last5),
                        baseline=0.0,
                        impact="positive",
                        unit=None,
                        label="Consistent recent scoring output",
                    )
                )
                reasons.append(
                    cls._reason_signal(
                        feature="shot_quality",
                        value=xg_per90 if xg_per90 > 0 else score,
                        baseline=0.0,
                        impact="positive",
                        unit=None,
                        label="Strong chance creation and conversion profile",
                    )
                )
            elif role_key == "bowler":
                reasons.append(
                    cls._reason_signal(
                        feature="recent_wickets",
                        value=recent_wickets,
                        baseline=0.0,
                        impact="positive",
                        unit="wickets",
                        label="Frequent wicket-taking spells",
                    )
                )
                reasons.append(
                    cls._reason_signal(
                        feature="bowling_control",
                        value=float(row.get("economy", 0.0) or 0.0),
                        baseline=0.0,
                        impact="positive",
                        unit="economy",
                        label="Controls scoring rate in pressure overs",
                    )
                )
            else:
                reasons.append(
                    cls._reason_signal(
                        feature="impact_score",
                        value=score,
                        baseline=0.0,
                        impact="positive",
                        unit=None,
                        label="High match impact across phases",
                    )
                )
                reasons.append(
                    cls._reason_signal(
                        feature="form_signal",
                        value=float(row.get("win_rate", row.get("form_points_last5", 0.0)) or 0.0),
                        baseline=0.0,
                        impact="positive",
                        unit=None,
                        label="Reliable contribution in recent matches",
                    )
                )
            return {
                "name": name,
                "role": role,
                "team": team,
                "reason": reasons[:2],
                "confidence": float(round(np.clip(score / 100.0, 0.35, 0.9), 2)),
            }

        if sport == "football":
            try:
                scorers = CatalogService.top_players(
                    sport="football",
                    tournament=tournament,
                    role="goal_scorer",
                    limit=3,
                )
            except Exception:  # noqa: BLE001
                scorers = []
            try:
                standout = CatalogService.top_players(
                    sport="football",
                    tournament=tournament,
                    role="standout",
                    limit=3,
                )
            except Exception:  # noqa: BLE001
                standout = []
            scorers_players = [_as_player(row, "goal_scorer") for row in scorers]
            standout_players = [_as_player(row, "standout") for row in standout]
            if not scorers_players and standout_players:
                scorers_players = standout_players[:1]
            if not standout_players and scorers_players:
                standout_players = scorers_players[:1]
            best_player = scorers_players[0] if scorers_players else _as_player({}, "goal_scorer")
            mom = standout_players[0] if standout_players else best_player
            return {
                "best_player": best_player,
                "best_bowler": None,
                "man_of_the_match": mom,
                "players": {
                    "top_goal_scorers": scorers_players[:3],
                    "top_standout": standout_players[:3],
                    "top_match_impact": standout_players[:3] or scorers_players[:3],
                    "top_batsmen": [],
                    "top_bowlers": [],
                },
            }

        settings = get_settings()
        player_path = settings.data_processed_dir / "player_form_latest.csv"
        batsmen: list[dict[str, Any]] = []
        bowlers: list[dict[str, Any]] = []
        impact: list[dict[str, Any]] = []
        if player_path.exists():
            try:
                import pandas as pd

                frame = pd.read_csv(player_path)
                frame = frame[frame["team"].astype(str).str.lower().isin([team_a.lower(), team_b.lower()])].copy()
                if not frame.empty:
                    frame["batsman_rank"] = (
                        pd.to_numeric(frame.get("recent_runs"), errors="coerce").fillna(0.0) * 0.45
                        + pd.to_numeric(frame.get("batting_form"), errors="coerce").fillna(0.0) * 0.4
                        + pd.to_numeric(frame.get("strike_rate"), errors="coerce").fillna(0.0) * 0.08
                    )
                    frame["bowler_rank"] = (
                        pd.to_numeric(frame.get("recent_wickets"), errors="coerce").fillna(0.0) * 10.0
                        + pd.to_numeric(frame.get("bowling_form"), errors="coerce").fillna(0.0) * 0.45
                        - pd.to_numeric(frame.get("economy"), errors="coerce").fillna(9.0) * 1.1
                    )
                    frame["impact_rank"] = pd.to_numeric(frame.get("impact_score"), errors="coerce").fillna(0.0)
                    batsmen = (
                        frame.sort_values("batsman_rank", ascending=False)
                        .head(3)
                        .rename(columns={"player": "name"})
                        .to_dict(orient="records")
                    )
                    bowler_frame = frame[
                        (pd.to_numeric(frame.get("recent_wickets"), errors="coerce").fillna(0.0) >= 0.5)
                        | (pd.to_numeric(frame.get("avg_wickets"), errors="coerce").fillna(0.0) >= 0.6)
                    ]
                    if bowler_frame.empty:
                        bowler_frame = frame.copy()
                    bowlers = (
                        bowler_frame.sort_values("bowler_rank", ascending=False)
                        .head(3)
                        .rename(columns={"player": "name"})
                        .to_dict(orient="records")
                    )
                    impact = (
                        frame.sort_values("impact_rank", ascending=False)
                        .head(3)
                        .rename(columns={"player": "name"})
                        .to_dict(orient="records")
                    )
            except Exception:  # noqa: BLE001
                batsmen = []
                bowlers = []
                impact = []

        if not batsmen:
            try:
                batsmen = CatalogService.top_players(
                    sport="cricket",
                    tournament=tournament,
                    team=team_a,
                    role="batsman",
                    limit=3,
                )
            except Exception:  # noqa: BLE001
                batsmen = []
        if not bowlers:
            try:
                bowlers = CatalogService.top_players(
                    sport="cricket",
                    tournament=tournament,
                    team=team_b,
                    role="bowler",
                    limit=3,
                )
            except Exception:  # noqa: BLE001
                bowlers = []
        if not impact:
            try:
                impact = CatalogService.top_players(
                    sport="cricket",
                    tournament=tournament,
                    role="standout",
                    limit=3,
                )
            except Exception:  # noqa: BLE001
                impact = []
        batsmen_players = [_as_player(row, "batsman") for row in batsmen]
        bowlers_players = [_as_player(row, "bowler") for row in bowlers]
        impact_players = [_as_player(row, "all-round impact") for row in impact]
        best_player = batsmen_players[0] if batsmen_players else (impact_players[0] if impact_players else _as_player({}, "batsman"))
        best_bowler = bowlers_players[0] if bowlers_players else None
        if best_bowler and best_bowler.get("name") == best_player.get("name"):
            replacement = next((row for row in bowlers_players if row.get("name") != best_player.get("name")), None)
            best_bowler = replacement or best_bowler
        mom = impact_players[0] if impact_players else best_player
        return {
            "best_player": best_player,
            "best_bowler": best_bowler,
            "man_of_the_match": mom,
            "players": {
                "top_batsmen": batsmen_players[:3],
                "top_bowlers": bowlers_players[:3],
                "top_match_impact": impact_players[:3] or [best_player],
                "top_goal_scorers": [],
                "top_standout": [],
            },
        }

    @classmethod
    def _fallback_cricket_prediction(
        cls,
        *,
        match_payload: dict[str, Any],
        k: int,
        anomaly_score: float,
        odd_variant_flag: bool,
        residual_shift: float,
    ) -> dict[str, Any]:
        team_a = str(match_payload.get("team_a", "Team A"))
        team_b = str(match_payload.get("team_b", "Team B"))
        tournament = str(match_payload.get("tournament", "IPL"))
        venue = str(match_payload.get("venue", "Unknown Venue"))

        frame = cls._cricket_history_frame()
        if frame is not None:
            frame = frame.copy()
        if frame is None or frame.empty:
            frame = None

        global_first = 170.0
        global_second = 167.0
        venue_first = global_first
        venue_second = global_second
        scores_a = cls._quantile_heads([], k=k, default_center=global_first, low=95, high=270, min_gap=7.0)
        scores_b = cls._quantile_heads([], k=k, default_center=global_first, low=95, high=270, min_gap=7.0)

        if frame is not None:
            numeric_cols = ["first_innings_total", "second_innings_total"]
            for col in numeric_cols:
                frame[col] = frame[col].astype(float)
            tournament_rows = frame[frame["tournament"].astype(str).str.lower() == tournament.lower()].copy()
            if tournament_rows.empty:
                tournament_rows = frame.copy()

            pair_mask = (
                (tournament_rows["team_a"].astype(str).str.lower() == team_a.lower())
                & (tournament_rows["team_b"].astype(str).str.lower() == team_b.lower())
            ) | (
                (tournament_rows["team_a"].astype(str).str.lower() == team_b.lower())
                & (tournament_rows["team_b"].astype(str).str.lower() == team_a.lower())
            )
            pair_rows = tournament_rows[pair_mask].copy()
            venue_rows = tournament_rows[tournament_rows["venue"].astype(str).str.lower() == venue.lower()].copy()

            global_first = cls._safe_mean(tournament_rows["first_innings_total"], 170.0)
            global_second = cls._safe_mean(tournament_rows["second_innings_total"], 167.0)
            venue_first = cls._safe_mean(venue_rows["first_innings_total"], global_first)
            venue_second = cls._safe_mean(venue_rows["second_innings_total"], global_second)

            a_first = pair_rows[pair_rows["first_innings_team"].astype(str).str.lower() == team_a.lower()]
            if len(a_first) < max(3, k):
                a_first = tournament_rows[tournament_rows["first_innings_team"].astype(str).str.lower() == team_a.lower()]
            b_first = pair_rows[pair_rows["first_innings_team"].astype(str).str.lower() == team_b.lower()]
            if len(b_first) < max(3, k):
                b_first = tournament_rows[tournament_rows["first_innings_team"].astype(str).str.lower() == team_b.lower()]

            scores_a = cls._quantile_heads(
                a_first["first_innings_total"] if not a_first.empty else tournament_rows["first_innings_total"],
                k=k,
                default_center=global_first,
                low=95,
                high=270,
                min_gap=7.0,
            )
            scores_b = cls._quantile_heads(
                b_first["first_innings_total"] if not b_first.empty else tournament_rows["first_innings_total"],
                k=k,
                default_center=global_first,
                low=95,
                high=270,
                min_gap=7.0,
            )

            def _chase_estimate(target: float, chasing_team: str, bowling_team: str) -> int:
                chase_rows = tournament_rows[tournament_rows["second_innings_team"].astype(str).str.lower() == chasing_team.lower()]
                concede_rows = tournament_rows[tournament_rows["first_innings_team"].astype(str).str.lower() == bowling_team.lower()]
                pair_chase = pair_rows[pair_rows["second_innings_team"].astype(str).str.lower() == chasing_team.lower()]
                values: list[tuple[float, float]] = []
                if not chase_rows.empty:
                    values.append((cls._safe_mean(chase_rows["second_innings_total"], global_second), 0.45))
                if not concede_rows.empty:
                    values.append((cls._safe_mean(concede_rows["second_innings_total"], global_second), 0.35))
                if not pair_chase.empty:
                    values.append((cls._safe_mean(pair_chase["second_innings_total"], global_second), 0.2))
                if values:
                    denom = sum(weight for _, weight in values)
                    base = sum(value * weight for value, weight in values) / max(denom, 1e-6)
                else:
                    base = global_second
                projected = base + (0.58 * (target - global_first)) + (0.22 * (venue_second - global_second))
                return int(np.clip(round(projected), 85, 280))
        else:
            def _chase_estimate(target: float, chasing_team: str, bowling_team: str) -> int:  # noqa: ARG001
                return int(np.clip(round(global_second + (0.58 * (target - global_first))), 85, 280))

        labels = ["Low", "Baseline", "High", "Aggressive"]
        scenario_scores = [float(round((a + b) / 2.0)) for a, b in zip(scores_a, scores_b)]
        probabilities = cls._scenario_probabilities(scenario_scores, temperature=11.0)
        confidences = cls._confidence_curve(len(scenario_scores), probabilities, model_base=0.52)

        prediction_rows: list[dict[str, Any]] = []
        for idx, (a_score, b_score, center_score) in enumerate(zip(scores_a, scores_b, scenario_scores)):
            b_chase = _chase_estimate(float(a_score), team_b, team_a)
            a_chase = _chase_estimate(float(b_score), team_a, team_b)
            label = labels[min(idx, len(labels) - 1)]
            if label == "Low":
                primary_reason = "Recent bowling pressure points to a lower-scoring script"
                secondary_reason = "Venue trend keeps first-innings totals in check"
            elif label == "High":
                primary_reason = "Recent batting form supports higher first-innings scoring"
                secondary_reason = "Venue trend supports above-average totals"
            elif label == "Aggressive":
                primary_reason = "Form spread widens the match outcome range"
                secondary_reason = "Chase-defend split creates a volatile finish"
            else:
                primary_reason = "Recent team form is closely matched"
                secondary_reason = "Venue trend is near league baseline"
            prediction_rows.append(
                {
                    "score": int(center_score),
                    "scenario": label,
                    "team_a_first": {
                        "batting_team": team_a,
                        "bowling_team": team_b,
                        "batting_score": int(a_score),
                        "chase_score": int(b_chase),
                        "winner": team_b if b_chase >= int(a_score) else team_a,
                    },
                    "team_b_first": {
                        "batting_team": team_b,
                        "bowling_team": team_a,
                        "batting_score": int(b_score),
                        "chase_score": int(a_chase),
                        "winner": team_a if a_chase >= int(b_score) else team_b,
                    },
                    "reason": [
                        cls._reason_signal(
                            feature="match_form_signal",
                            value=float(a_score),
                            baseline=global_first,
                            impact="positive" if a_score >= global_first else "negative",
                            unit="runs",
                            label=primary_reason,
                        ),
                        cls._reason_signal(
                            feature="venue_trend",
                            value=float(b_score),
                            baseline=global_first,
                            impact="positive" if b_score >= global_first else "negative",
                            unit="runs",
                            label=secondary_reason,
                        ),
                        cls._reason_signal(
                            feature="branch_outcome_gap",
                            value=venue_first,
                            baseline=global_first,
                            impact="positive" if venue_first >= global_first else "negative",
                            unit="runs",
                            label="Both batting-order branches remain competitive",
                        ),
                    ],
                    "scenario_probability": round(probabilities[idx], 2),
                    "confidence": round(confidences[idx], 2),
                }
            )

        players = cls._fallback_players(
            sport="cricket",
            tournament=tournament,
            team_a=team_a,
            team_b=team_b,
        )
        spread = float(max(scenario_scores) - min(scenario_scores)) if scenario_scores else 0.0
        return {
            "match": {
                "sport": "cricket",
                "tournament": tournament,
                "team_a": team_a,
                "team_b": team_b,
                "venue": venue,
                "match_date": match_payload.get("match_date"),
                "state": str(match_payload.get("state", "upcoming")),
            },
            "predictions": prediction_rows,
            "best_player": players["best_player"],
            "best_bowler": players["best_bowler"],
            "man_of_the_match": players["man_of_the_match"],
            "players": players["players"],
            "uncertainty": {
                "spread": round(spread, 3),
                "interval_low": float(min(scenario_scores) if scenario_scores else global_first - 8),
                "interval_high": float(max(scenario_scores) if scenario_scores else global_first + 8),
                "mean_prediction": round(float(np.mean(scenario_scores)) if scenario_scores else global_first, 3),
                "std_prediction": round(float(np.std(scenario_scores)) if scenario_scores else 0.0, 3),
            },
            "metadata": {
                "model_mode": "fallback",
                "num_heads": len(prediction_rows),
                "timemcl_style": {
                    "shared_encoder": True,
                    "winner_takes_all": True,
                    "diversity_regularization": True,
                    "multi_hypothesis": True,
                },
                "calibration": {"enabled": False, "method": "fallback_none"},
                "scenario_probabilities": [round(float(x), 2) for x in probabilities],
                "anomaly_score": round(anomaly_score, 3),
                "odd_variant_flag": odd_variant_flag,
                "residual_shift_score": round(residual_shift, 3),
            },
        }

    @classmethod
    def _fallback_football_prediction(
        cls,
        *,
        match_payload: dict[str, Any],
        k: int,
        anomaly_score: float,
        odd_variant_flag: bool,
        residual_shift: float,
    ) -> dict[str, Any]:
        team_a = str(match_payload.get("team_a", "Home Team"))
        team_b = str(match_payload.get("team_b", "Away Team"))
        tournament = str(match_payload.get("tournament", "EPL"))
        venue = str(match_payload.get("venue", "Unknown Venue"))
        state = str(match_payload.get("state", "upcoming"))

        frame = cls._football_history_frame()
        if frame is not None:
            frame = frame.copy()
        if frame is None or frame.empty:
            frame = None

        totals: list[float] = []
        diffs: list[float] = []
        expected_a = 1.35
        expected_b = 1.2

        if frame is not None:
            for col in ["home_goals", "away_goals"]:
                frame[col] = frame[col].astype(float)
            tournament_rows = frame[frame["tournament"].astype(str).str.lower() == tournament.lower()].copy()
            if tournament_rows.empty:
                tournament_rows = frame.copy()

            team_a_home = tournament_rows[tournament_rows["team_a"].astype(str).str.lower() == team_a.lower()]
            team_a_away = tournament_rows[tournament_rows["team_b"].astype(str).str.lower() == team_a.lower()]
            team_b_home = tournament_rows[tournament_rows["team_a"].astype(str).str.lower() == team_b.lower()]
            team_b_away = tournament_rows[tournament_rows["team_b"].astype(str).str.lower() == team_b.lower()]

            a_for_values = np.concatenate(
                [
                    team_a_home["home_goals"].to_numpy(dtype=float) if not team_a_home.empty else np.asarray([], dtype=float),
                    team_a_away["away_goals"].to_numpy(dtype=float) if not team_a_away.empty else np.asarray([], dtype=float),
                ]
            )
            b_for_values = np.concatenate(
                [
                    team_b_home["home_goals"].to_numpy(dtype=float) if not team_b_home.empty else np.asarray([], dtype=float),
                    team_b_away["away_goals"].to_numpy(dtype=float) if not team_b_away.empty else np.asarray([], dtype=float),
                ]
            )
            expected_a = cls._safe_mean(a_for_values, expected_a)
            expected_b = cls._safe_mean(b_for_values, expected_b)

            pair_mask = (
                (tournament_rows["team_a"].astype(str).str.lower() == team_a.lower())
                & (tournament_rows["team_b"].astype(str).str.lower() == team_b.lower())
            ) | (
                (tournament_rows["team_a"].astype(str).str.lower() == team_b.lower())
                & (tournament_rows["team_b"].astype(str).str.lower() == team_a.lower())
            )
            pair_rows = tournament_rows[pair_mask].copy()

            if not pair_rows.empty:
                for _, row in pair_rows.iterrows():
                    if str(row.get("team_a", "")).strip().lower() == team_a.lower():
                        a_goals = float(row.get("home_goals", 0.0))
                        b_goals = float(row.get("away_goals", 0.0))
                    else:
                        a_goals = float(row.get("away_goals", 0.0))
                        b_goals = float(row.get("home_goals", 0.0))
                    totals.append(a_goals + b_goals)
                    diffs.append(a_goals - b_goals)

            if not totals:
                totals = (tournament_rows["home_goals"] + tournament_rows["away_goals"]).astype(float).tolist()
                diffs = (tournament_rows["home_goals"] - tournament_rows["away_goals"]).astype(float).tolist()

        total_heads = cls._quantile_heads(
            totals,
            k=k,
            default_center=expected_a + expected_b,
            low=0.2,
            high=6.0,
            min_gap=0.45,
        )
        diff_heads = cls._quantile_heads(
            diffs,
            k=k,
            default_center=expected_a - expected_b,
            low=-4.0,
            high=4.0,
            min_gap=0.35,
        )

        labels = ["Low", "Baseline", "High", "Aggressive"]
        probabilities = cls._scenario_probabilities([float(x) for x in total_heads], temperature=1.5)
        confidences = cls._confidence_curve(len(total_heads), probabilities, model_base=0.55)
        prediction_rows: list[dict[str, Any]] = []
        for idx, (total_goals, diff_goals) in enumerate(zip(total_heads, diff_heads)):
            home_goals = int(max(0, round((float(total_goals) + float(diff_goals)) / 2.0)))
            away_goals = int(max(0, round(float(total_goals) - home_goals)))
            label = labels[min(idx, len(labels) - 1)]
            if home_goals > away_goals:
                likely = f"{team_a} win"
            elif away_goals > home_goals:
                likely = f"{team_b} win"
            else:
                likely = "Draw"
            if label == "Low":
                reason_primary = "Recent defensive form keeps expected goals lower"
            elif label == "High":
                reason_primary = "Recent attacking form pushes totals upward"
            elif label == "Aggressive":
                reason_primary = "High volatility in chance conversion widens outcomes"
            else:
                reason_primary = "Both teams project a balanced chance profile"
            prediction_rows.append(
                {
                    "scoreline": f"{home_goals}-{away_goals}",
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "scenario": label,
                    "likely_result": likely,
                    "reason": [
                        cls._reason_signal(
                            feature="attacking_signal",
                            value=expected_a,
                            baseline=1.25,
                            impact="positive" if expected_a >= 1.25 else "negative",
                            unit="goals",
                            label=reason_primary,
                        ),
                        cls._reason_signal(
                            feature="opponent_response",
                            value=expected_b,
                            baseline=1.15,
                            impact="positive" if expected_b >= 1.15 else "negative",
                            unit="goals",
                            label="Opponent response profile remains competitive",
                        ),
                    ],
                    "scenario_probability": round(probabilities[idx], 2),
                    "confidence": round(confidences[idx], 2),
                }
            )

        players = cls._fallback_players(
            sport="football",
            tournament=tournament,
            team_a=team_a,
            team_b=team_b,
        )
        totals_numeric = [float(item["home_goals"]) + float(item["away_goals"]) for item in prediction_rows]
        return {
            "match": {
                "sport": "football",
                "tournament": tournament,
                "team_a": team_a,
                "team_b": team_b,
                "venue": venue,
                "match_date": match_payload.get("match_date"),
                "state": state,
            },
            "predictions": prediction_rows,
            "best_player": players["best_player"],
            "best_bowler": None,
            "man_of_the_match": players["man_of_the_match"],
            "players": players["players"],
            "uncertainty": {
                "spread": round(float(max(totals_numeric) - min(totals_numeric)) if totals_numeric else 0.0, 3),
                "interval_low": float(min(totals_numeric) if totals_numeric else 0.0),
                "interval_high": float(max(totals_numeric) if totals_numeric else 0.0),
                "mean_prediction": round(float(np.mean(totals_numeric)) if totals_numeric else 0.0, 3),
                "std_prediction": round(float(np.std(totals_numeric)) if totals_numeric else 0.0, 3),
            },
            "metadata": {
                "model_mode": "fallback",
                "num_heads": len(prediction_rows),
                "timemcl_style": {
                    "shared_encoder": True,
                    "winner_takes_all": True,
                    "diversity_regularization": True,
                    "multi_hypothesis": True,
                },
                "calibration": {"enabled": False, "method": "fallback_none"},
                "scenario_probabilities": [round(float(x), 2) for x in probabilities],
                "anomaly_score": round(anomaly_score, 3),
                "odd_variant_flag": odd_variant_flag,
                "residual_shift_score": round(residual_shift, 3),
            },
        }

    @classmethod
    def _heuristic_fallback(cls, match_payload: dict[str, Any], k: int) -> dict[str, Any]:
        sport = str(match_payload.get("sport", "cricket")).lower()
        residual_context = (
            match_payload.get("residual_context")
            if isinstance(match_payload.get("residual_context"), dict)
            else {}
        )
        combined_bias = float(residual_context.get("combined_bias", 0.0))
        residual_shift = float(residual_context.get("residual_shift_score", 0.0))
        anomaly_score = float(np.clip((abs(combined_bias) / 20.0) + (residual_shift / 18.0), 0.0, 1.0))
        odd_variant_flag = anomaly_score >= 0.62

        if sport == "football":
            return cls._fallback_football_prediction(
                match_payload=match_payload,
                k=k,
                anomaly_score=anomaly_score,
                odd_variant_flag=odd_variant_flag,
                residual_shift=residual_shift,
            )
        return cls._fallback_cricket_prediction(
            match_payload=match_payload,
            k=k,
            anomaly_score=anomaly_score,
            odd_variant_flag=odd_variant_flag,
            residual_shift=residual_shift,
        )
