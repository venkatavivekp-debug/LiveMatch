from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

try:
    import torch

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

from ml.calibration import evaluate_conformal_interval
from ml.config import MLConfig

if TORCH_AVAILABLE:
    from ml.model import TimeMCLModel
else:
    TimeMCLModel = Any  # type: ignore[assignment]


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def coverage(actual: np.ndarray, predictions: np.ndarray) -> float:
    lower = predictions.min(axis=1)
    upper = predictions.max(axis=1)
    return float(np.mean((actual >= lower) & (actual <= upper)))


def best_match_error(actual: np.ndarray, predictions: np.ndarray) -> float:
    errors = np.abs(predictions - actual[:, None])
    return float(np.mean(np.min(errors, axis=1)))


def diversity_score(predictions: np.ndarray) -> float:
    if predictions.shape[1] < 2:
        return 0.0
    pairwise = []
    for i in range(predictions.shape[1]):
        for j in range(i + 1, predictions.shape[1]):
            pairwise.append(np.abs(predictions[:, i] - predictions[:, j]))
    return float(np.mean(np.stack(pairwise, axis=1)))


def interval_width(predictions: np.ndarray) -> float:
    return float(np.mean(predictions.max(axis=1) - predictions.min(axis=1)))


def scenario_spread_metric(predictions: np.ndarray) -> float:
    return float(np.mean(np.std(predictions, axis=1)))


def crps_ensemble(actual: np.ndarray, predictions: np.ndarray) -> float:
    term1 = np.mean(np.abs(predictions - actual[:, None]), axis=1)
    pairwise = np.abs(predictions[:, :, None] - predictions[:, None, :])
    term2 = 0.5 * np.mean(pairwise, axis=(1, 2))
    return float(np.mean(term1 - term2))


def calibration_summary(
    actual: np.ndarray,
    predictions: np.ndarray,
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
) -> dict[str, float]:
    summary: dict[str, float] = {}
    sorted_preds = np.sort(predictions, axis=1)
    for quantile in quantiles:
        q_values = np.quantile(sorted_preds, quantile, axis=1)
        observed = np.mean(actual <= q_values)
        summary[f"quantile_{quantile:.1f}_observed"] = float(observed)
        summary[f"quantile_{quantile:.1f}_error"] = float(abs(observed - quantile))
    summary["calibration_mae"] = float(
        np.mean([value for key, value in summary.items() if key.endswith("_error")])
    )
    return summary


def winner_accuracy(actual_winner_team_a: np.ndarray, winner_probability: np.ndarray) -> float:
    predicted = (winner_probability >= 0.5).astype(np.float32)
    return float(np.mean(predicted == actual_winner_team_a.astype(np.float32)))


def best_match_error_multi(actual_scores: np.ndarray, predicted_scores: np.ndarray) -> float:
    deltas = predicted_scores - actual_scores[:, None, :]
    head_error = np.sqrt(np.mean(deltas**2, axis=2))
    return float(np.mean(np.min(head_error, axis=1)))


def coverage_multi(actual_scores: np.ndarray, predicted_scores: np.ndarray) -> float:
    lower = predicted_scores.min(axis=1)
    upper = predicted_scores.max(axis=1)
    in_range = (actual_scores >= lower) & (actual_scores <= upper)
    return float(np.mean(np.all(in_range, axis=1)))


def interval_width_multi(predicted_scores: np.ndarray) -> float:
    spread = predicted_scores.max(axis=1) - predicted_scores.min(axis=1)
    return float(np.mean(np.mean(spread, axis=1)))


def diversity_score_multi(
    predicted_scores: np.ndarray,
    winner_probability: np.ndarray | None = None,
) -> float:
    if predicted_scores.shape[1] < 2:
        return 0.0
    pairwise: list[np.ndarray] = []
    for i in range(predicted_scores.shape[1]):
        for j in range(i + 1, predicted_scores.shape[1]):
            score_dist = np.linalg.norm(predicted_scores[:, i, :] - predicted_scores[:, j, :], axis=1)
            if winner_probability is not None:
                score_dist = score_dist + (2.5 * np.abs(winner_probability[:, i] - winner_probability[:, j]))
            pairwise.append(score_dist)
    return float(np.mean(np.stack(pairwise, axis=1)))


def football_scoreline_hit_rate(
    actual_home: np.ndarray,
    actual_away: np.ndarray,
    predicted_home: np.ndarray,
    predicted_away: np.ndarray,
) -> float:
    hits = (actual_home[:, None] == predicted_home) & (actual_away[:, None] == predicted_away)
    return float(np.mean(np.any(hits, axis=1)))


def _prepare_eval_frame(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    required_cols = feature_columns + [
        "target_team_a_score",
        "target_team_b_score",
        "target_winner_team_a",
    ]
    return frame.dropna(subset=required_cols).copy()


def _predict_from_checkpoint(
    checkpoint: dict[str, Any],
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    feature_columns: list[str] = checkpoint["feature_columns"]
    x = frame[feature_columns].to_numpy(dtype=np.float32)
    mean = np.array(checkpoint["scaler"]["mean"], dtype=np.float32)
    std = np.array(checkpoint["scaler"]["std"], dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    x_scaled = (x - mean) / std
    model = TimeMCLModel(
        input_dim=len(feature_columns),
        num_heads=int(checkpoint["num_heads"]),
        hidden_dims=tuple(checkpoint["hidden_dims"]),
        dropout=float(checkpoint["dropout"]),
        encoder_type=str(checkpoint.get("encoder_type", "mlp")),
        patch_length=int((checkpoint.get("patch_encoder") or {}).get("patch_length", 4)),
        patch_stride=int((checkpoint.get("patch_encoder") or {}).get("patch_stride", 2)),
        patch_model_dim=int((checkpoint.get("patch_encoder") or {}).get("patch_model_dim", 64)),
        patch_layers=int((checkpoint.get("patch_encoder") or {}).get("patch_layers", 2)),
        patch_attention_heads=int((checkpoint.get("patch_encoder") or {}).get("patch_attention_heads", 4)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        score_predictions, winner_logits = model(torch.tensor(x_scaled, dtype=torch.float32))
        return score_predictions.numpy(), torch.sigmoid(winner_logits).numpy()


def run_evaluation(
    checkpoint_path: Path,
    features_path: Path,
    ensemble_manifest_path: Path | None = None,
) -> dict[str, float | dict[str, float]]:
    frame = pd.read_csv(features_path)
    checkpoints: list[dict[str, Any]] = []
    source_label = "single"
    active_manifest: dict[str, Any] | None = None

    if ensemble_manifest_path and ensemble_manifest_path.exists():
        try:
            active_manifest = json.loads(ensemble_manifest_path.read_text(encoding="utf-8"))
            if isinstance(active_manifest, dict):
                member_rows = active_manifest.get("members")
                if isinstance(member_rows, list):
                    for row in member_rows:
                        if not isinstance(row, dict):
                            continue
                        member_path = row.get("checkpoint_path")
                        if not isinstance(member_path, str):
                            continue
                        path = Path(member_path)
                        if not path.is_absolute():
                            path = (ensemble_manifest_path.parent / path).resolve()
                        if path.exists():
                            checkpoints.append(torch.load(path, map_location="cpu"))
        except Exception:  # noqa: BLE001
            checkpoints = []
            active_manifest = None

    if not checkpoints:
        checkpoints = [torch.load(checkpoint_path, map_location="cpu")]
        source_label = "single"
    else:
        source_label = "ensemble"

    feature_columns: list[str] = checkpoints[0]["feature_columns"]
    eval_frame = _prepare_eval_frame(frame=frame, feature_columns=feature_columns)
    actual_scores = eval_frame[["target_team_a_score", "target_team_b_score"]].to_numpy(dtype=np.float32)
    actual_winner_a = eval_frame["target_winner_team_a"].to_numpy(dtype=np.float32)

    score_heads_rows: list[np.ndarray] = []
    winner_head_rows: list[np.ndarray] = []
    for checkpoint in checkpoints:
        score_preds, winner_probs = _predict_from_checkpoint(checkpoint=checkpoint, frame=eval_frame)
        score_heads_rows.append(score_preds)
        winner_head_rows.append(winner_probs)

    pred_scores = np.concatenate(score_heads_rows, axis=1)
    pred_winner_probs = np.concatenate(winner_head_rows, axis=1)
    center_scores = np.mean(pred_scores, axis=1)
    center_totals = np.mean(center_scores, axis=1)
    actual_totals = np.mean(actual_scores, axis=1)
    winner_center = np.mean(pred_winner_probs, axis=1)
    metrics: dict[str, float | dict[str, float]] = {
        "mae_team_a_score": mae(actual_scores[:, 0], center_scores[:, 0]),
        "mae_team_b_score": mae(actual_scores[:, 1], center_scores[:, 1]),
        "mae_total": mae(actual_totals, center_totals),
        "rmse_team_a_score": rmse(actual_scores[:, 0], center_scores[:, 0]),
        "rmse_team_b_score": rmse(actual_scores[:, 1], center_scores[:, 1]),
        "rmse_total": rmse(actual_totals, center_totals),
        "winner_accuracy": winner_accuracy(actual_winner_a, winner_center),
        "winner_brier": float(np.mean((winner_center - actual_winner_a) ** 2)),
        "best_scenario_error": best_match_error_multi(actual_scores, pred_scores),
        "coverage": coverage_multi(actual_scores, pred_scores),
        "diversity": diversity_score_multi(pred_scores, pred_winner_probs),
        "ensemble_disagreement": diversity_score_multi(pred_scores, pred_winner_probs),
        "interval_width": interval_width_multi(pred_scores),
        "crps_team_a": crps_ensemble(actual_scores[:, 0], pred_scores[:, :, 0]),
        "crps_team_b": crps_ensemble(actual_scores[:, 1], pred_scores[:, :, 1]),
        "samples": float(len(eval_frame)),
        "calibration_team_a": calibration_summary(actual_scores[:, 0], pred_scores[:, :, 0]),
        "calibration_team_b": calibration_summary(actual_scores[:, 1], pred_scores[:, :, 1]),
        "ensemble_size_used": float(len(checkpoints)),
    }

    conformal = checkpoints[0].get("conformal_calibration")
    if isinstance(conformal, dict) and conformal.get("enabled", False):
        team_a_conformal = evaluate_conformal_interval(
            actual_scores[:, 0],
            pred_scores[:, :, 0],
            conformal,
        )
        team_b_conformal = evaluate_conformal_interval(
            actual_scores[:, 1],
            pred_scores[:, :, 1],
            conformal,
        )
        metrics["conformal_coverage"] = float((team_a_conformal["coverage"] + team_b_conformal["coverage"]) / 2.0)
        metrics["conformal_interval_width"] = float(
            (team_a_conformal["interval_width"] + team_b_conformal["interval_width"]) / 2.0
        )
        metrics["conformal_alpha"] = float(conformal.get("alpha", 0.1))
    if source_label == "ensemble" and isinstance(active_manifest, dict):
        metrics["ensemble_members_available"] = float(len(active_manifest.get("members") or []))
    return metrics


def summarize_metrics_for_status(metrics: dict[str, float | dict[str, float]]) -> dict[str, float]:
    keys = [
        "mae_team_a_score",
        "mae_team_b_score",
        "mae_total",
        "rmse_team_a_score",
        "rmse_team_b_score",
        "rmse_total",
        "winner_accuracy",
        "winner_brier",
        "best_scenario_error",
        "coverage",
        "conformal_coverage",
        "diversity",
        "ensemble_disagreement",
        "interval_width",
        "conformal_interval_width",
    ]
    return {
        key: float(metrics[key])
        for key in keys
        if key in metrics and isinstance(metrics[key], (int, float))
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TimeMCL-inspired LiveMatch model")
    parser.add_argument("--checkpoint", type=Path, default=MLConfig().artifacts_dir / "time_mcl.pt")
    parser.add_argument(
        "--ensemble-manifest",
        type=Path,
        default=MLConfig().ensemble_dir / "ensemble_manifest.json",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=MLConfig().processed_data_dir / "model_features.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=MLConfig().artifacts_dir / "evaluation" / "evaluation_metrics.json",
    )
    parser.add_argument(
        "--latest-out",
        type=Path,
        default=MLConfig().artifacts_dir / "evaluation" / "latest_evaluation_summary.json",
    )
    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        raise SystemExit(
            "Torch is not installed. Running in fallback mode (no trained model available). "
            "Install optional torch via scripts/setup_ml.sh before evaluation."
        )

    metrics = run_evaluation(
        checkpoint_path=args.checkpoint,
        features_path=args.features,
        ensemble_manifest_path=args.ensemble_manifest,
    )
    payload = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "checkpoint_path": str(args.checkpoint),
        "ensemble_manifest_path": str(args.ensemble_manifest),
        "features_path": str(args.features),
        "metrics": metrics,
        "status_summary": summarize_metrics_for_status(metrics),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.latest_out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    args.latest_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
