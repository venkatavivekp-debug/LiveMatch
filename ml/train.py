from __future__ import annotations

import argparse
import json
import random
import shutil
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]
    DataLoader = object  # type: ignore[assignment]
    TensorDataset = object  # type: ignore[assignment]
    TORCH_AVAILABLE = False

from ml.config import FEATURE_COLUMNS, MLConfig, ensure_directories
from ml.calibration import evaluate_conformal_interval, fit_split_conformal_interval
from ml.evaluate import (
    best_match_error_multi,
    coverage_multi,
    crps_ensemble,
    diversity_score_multi,
    interval_width_multi,
    mae,
    rmse,
    summarize_metrics_for_status,
    winner_accuracy,
)
from ml.features import run_feature_pipeline

if TORCH_AVAILABLE:
    from ml.model import TimeMCLModel, wta_diverse_loss
else:
    TimeMCLModel = Any  # type: ignore[assignment]

    def wta_diverse_loss(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Torch is not installed. Install torch to train the model.")


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE and torch is not None:
        torch.manual_seed(seed)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_training_frame(config: MLConfig) -> pd.DataFrame:
    features_path = config.processed_data_dir / "model_features.csv"
    if not features_path.exists():
        run_feature_pipeline(config=config, bootstrap_if_missing=False)

    frame = pd.read_csv(features_path)
    required_cols = FEATURE_COLUMNS + [
        "target_team_a_score",
        "target_team_b_score",
        "target_winner_team_a",
        "match_date",
    ]
    missing = [col for col in required_cols if col not in frame.columns]
    if missing:
        run_feature_pipeline(config=config, bootstrap_if_missing=False)
        frame = pd.read_csv(features_path)
        missing = [col for col in required_cols if col not in frame.columns]
        if missing:
            raise RuntimeError(
                "Missing required training columns after feature rebuild: "
                + ", ".join(missing)
            )

    frame = frame.dropna(subset=required_cols)
    frame["match_date"] = pd.to_datetime(frame["match_date"], errors="coerce")
    frame = frame.dropna(subset=["match_date"]).sort_values("match_date")
    return frame.reset_index(drop=True)


def time_split(
    frame: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    years = frame["match_date"].dt.year
    if years.min() <= 2010 and years.max() >= 2024:
        train_frame = frame[years <= 2021].copy()
        val_frame = frame[years == 2022].copy()
        test_frame = frame[years >= 2023].copy()
        if not train_frame.empty and not val_frame.empty and not test_frame.empty:
            return train_frame, val_frame, test_frame

    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"train/val/test ratios must sum to 1.0, got {ratio_sum}")

    n = len(frame)
    if n < 12:
        raise ValueError("Need at least 12 rows for train/val/test split.")

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_end = max(1, min(train_end, n - 2))
    val_end = max(train_end + 1, min(val_end, n - 1))

    train_frame = frame.iloc[:train_end].copy()
    val_frame = frame.iloc[train_end:val_end].copy()
    test_frame = frame.iloc[val_end:].copy()

    if train_frame.empty or val_frame.empty or test_frame.empty:
        raise ValueError(
            "Time split produced empty partition. "
            f"sizes={(len(train_frame), len(val_frame), len(test_frame))}"
        )

    return train_frame, val_frame, test_frame


def standardize(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)

    x_train_scaled = (x_train - mean) / std
    x_val_scaled = (x_val - mean) / std
    x_test_scaled = (x_test - mean) / std
    return x_train_scaled, x_val_scaled, x_test_scaled, mean, std


def evaluate_predictions(
    y_scores: np.ndarray,
    y_winner: np.ndarray,
    score_preds: np.ndarray,
    winner_probs: np.ndarray,
) -> dict[str, float]:
    center_scores = score_preds.mean(axis=1)
    winner_center = winner_probs.mean(axis=1)
    return {
        "mae_team_a_score": mae(y_scores[:, 0], center_scores[:, 0]),
        "mae_team_b_score": mae(y_scores[:, 1], center_scores[:, 1]),
        "mae_total": mae(np.mean(y_scores, axis=1), np.mean(center_scores, axis=1)),
        "rmse_team_a_score": rmse(y_scores[:, 0], center_scores[:, 0]),
        "rmse_team_b_score": rmse(y_scores[:, 1], center_scores[:, 1]),
        "rmse_total": rmse(np.mean(y_scores, axis=1), np.mean(center_scores, axis=1)),
        "winner_accuracy": winner_accuracy(y_winner, winner_center),
        "winner_brier": float(np.mean((winner_center - y_winner) ** 2)),
        "coverage": coverage_multi(y_scores, score_preds),
        "best_scenario_error": best_match_error_multi(y_scores, score_preds),
        "diversity": diversity_score_multi(score_preds, winner_probs),
        "ensemble_disagreement": diversity_score_multi(score_preds, winner_probs),
        "interval_width": interval_width_multi(score_preds),
        "crps_team_a": crps_ensemble(y_scores[:, 0], score_preds[:, :, 0]),
        "crps_team_b": crps_ensemble(y_scores[:, 1], score_preds[:, :, 1]),
    }


def evaluate_model(
    model: nn.Module,
    x_eval: np.ndarray,
    y_scores: np.ndarray,
    y_winner: np.ndarray,
) -> dict[str, float]:
    score_preds, winner_probs = predict_heads(model=model, features=x_eval)
    return evaluate_predictions(y_scores, y_winner, score_preds, winner_probs)


def predict_heads(model: nn.Module, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        score_preds, winner_logits = model(torch.tensor(features, dtype=torch.float32))
        score_np = score_preds.cpu().numpy()
        winner_np = torch.sigmoid(winner_logits).cpu().numpy()
    return score_np, winner_np


def train_model(
    frame: pd.DataFrame,
    config: MLConfig,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    soft_wta_temperature: float,
    encoder_type: str,
    patch_length: int,
    patch_stride: int,
    patch_model_dim: int,
    patch_layers: int,
    patch_attention_heads: int,
    conformal_alpha: float,
    member_seed: int,
    bootstrap_train: bool,
) -> tuple[TimeMCLModel, dict[str, Any]]:
    set_seeds(member_seed)
    train_frame, val_frame, test_frame = time_split(
        frame=frame,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
    )
    if bootstrap_train:
        train_frame = (
            train_frame.sample(
                n=len(train_frame),
                replace=True,
                random_state=member_seed,
            )
            .sort_values("match_date")
            .reset_index(drop=True)
        )

    x_train = train_frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_train_scores = train_frame[["target_team_a_score", "target_team_b_score"]].to_numpy(dtype=np.float32)
    y_train_winner = train_frame["target_winner_team_a"].to_numpy(dtype=np.float32)
    x_val = val_frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_val_scores = val_frame[["target_team_a_score", "target_team_b_score"]].to_numpy(dtype=np.float32)
    y_val_winner = val_frame["target_winner_team_a"].to_numpy(dtype=np.float32)
    x_test = test_frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_test_scores = test_frame[["target_team_a_score", "target_team_b_score"]].to_numpy(dtype=np.float32)
    y_test_winner = test_frame["target_winner_team_a"].to_numpy(dtype=np.float32)

    x_train_scaled, x_val_scaled, x_test_scaled, mean, std = standardize(x_train, x_val, x_test)

    train_dataset = TensorDataset(
        torch.tensor(x_train_scaled, dtype=torch.float32),
        torch.tensor(y_train_scores, dtype=torch.float32),
        torch.tensor(y_train_winner, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = TimeMCLModel(
        input_dim=len(FEATURE_COLUMNS),
        num_heads=config.num_heads,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
        encoder_type=encoder_type,  # PatchTST-style option, TimeMCL heads unchanged.
        patch_length=patch_length,
        patch_stride=patch_stride,
        patch_model_dim=patch_model_dim,
        patch_layers=patch_layers,
        patch_attention_heads=patch_attention_heads,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    history: list[dict[str, float]] = []
    best_val = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_primary = 0.0
        running_winner = 0.0
        running_diversity = 0.0
        running_entropy = 0.0

        for batch_x, batch_y_scores, batch_y_winner in train_loader:
            optimizer.zero_grad(set_to_none=True)
            score_predictions, winner_logits = model(batch_x)
            loss_output = wta_diverse_loss(
                score_predictions,
                batch_y_scores,
                winner_logits=winner_logits,
                winner_target=batch_y_winner,
                diversity_margin=config.diversity_margin,
                diversity_weight=config.diversity_weight,
                winner_loss_weight=config.winner_loss_weight,
                soft_wta_temperature=soft_wta_temperature,
            )
            loss_output.loss.backward()
            optimizer.step()

            running_loss += float(loss_output.loss.item())
            running_primary += float(loss_output.primary_loss.item())
            running_winner += float(loss_output.winner_loss.item())
            running_diversity += float(loss_output.diversity_penalty.item())
            running_entropy += float(loss_output.winner_soft_entropy.item())

        val_metrics = evaluate_model(model, x_val_scaled, y_val_scores, y_val_winner)
        epoch_metrics = {
            "epoch": float(epoch),
            "train_loss": running_loss / max(1, len(train_loader)),
            "train_primary_loss": running_primary / max(1, len(train_loader)),
            "train_winner_loss": running_winner / max(1, len(train_loader)),
            "train_diversity_penalty": running_diversity / max(1, len(train_loader)),
            "train_winner_soft_entropy": running_entropy / max(1, len(train_loader)),
            **val_metrics,
        }
        history.append(epoch_metrics)

        if epoch_metrics["best_scenario_error"] < best_val:
            best_val = epoch_metrics["best_scenario_error"]
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"epoch={epoch:03d} "
                f"train_loss={epoch_metrics['train_loss']:.3f} "
                f"val_best={epoch_metrics['best_scenario_error']:.3f} "
                f"val_rmse_total={epoch_metrics['rmse_total']:.3f} "
                f"val_wacc={epoch_metrics['winner_accuracy']:.3f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    val_score_preds, val_winner_probs = predict_heads(model=model, features=x_val_scaled)
    test_score_preds, test_winner_probs = predict_heads(model=model, features=x_test_scaled)
    test_metrics = evaluate_predictions(
        y_scores=y_test_scores,
        y_winner=y_test_winner,
        score_preds=test_score_preds,
        winner_probs=test_winner_probs,
    )

    conformal_a = fit_split_conformal_interval(
        actual=y_val_scores[:, 0],
        predictions=val_score_preds[:, :, 0],
        alpha=conformal_alpha,
    )
    conformal_b = fit_split_conformal_interval(
        actual=y_val_scores[:, 1],
        predictions=val_score_preds[:, :, 1],
        alpha=conformal_alpha,
    )
    conformal_test_a = evaluate_conformal_interval(
        actual=y_test_scores[:, 0],
        predictions=test_score_preds[:, :, 0],
        calibration=conformal_a,
    )
    conformal_test_b = evaluate_conformal_interval(
        actual=y_test_scores[:, 1],
        predictions=test_score_preds[:, :, 1],
        calibration=conformal_b,
    )
    test_metrics["conformal_coverage"] = float(
        (conformal_test_a["coverage"] + conformal_test_b["coverage"]) / 2.0
    )
    test_metrics["conformal_interval_width"] = float(
        (conformal_test_a["interval_width"] + conformal_test_b["interval_width"]) / 2.0
    )
    conformal_calibration = {
        "enabled": True,
        "alpha": conformal_alpha,
        "team_a": {
            **conformal_a,
            "test_coverage": conformal_test_a["coverage"],
            "test_interval_width": conformal_test_a["interval_width"],
        },
        "team_b": {
            **conformal_b,
            "test_coverage": conformal_test_b["coverage"],
            "test_interval_width": conformal_test_b["interval_width"],
        },
    }

    metadata: dict[str, Any] = {
        "run_id": f"run_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}",
        "generated_at": utc_now_iso(),
        "experiment_tag": config.experiment_tag,
        "member_seed": member_seed,
        "bootstrap_train": bool(bootstrap_train),
        "feature_columns": FEATURE_COLUMNS,
        "scaler": {"mean": mean.tolist(), "std": std.tolist()},
        "num_heads": config.num_heads,
        "hidden_dims": list(config.hidden_dims),
        "dropout": config.dropout,
        "target_spec": {
            "score_outputs": ["team_a_score", "team_b_score"],
            "winner_output": "team_a_win_probability",
        },
        "encoder_type": encoder_type,
        "patch_encoder": {
            "patch_length": patch_length,
            "patch_stride": patch_stride,
            "patch_model_dim": patch_model_dim,
            "patch_layers": patch_layers,
            "patch_attention_heads": patch_attention_heads,
        },
        "history": history,
        "soft_wta_temperature": soft_wta_temperature,
        "conformal_calibration": conformal_calibration,
        "train_samples": int(len(train_frame)),
        "val_samples": int(len(val_frame)),
        "test_samples": int(len(test_frame)),
        "split_ratios": {
            "train": config.train_ratio,
            "val": config.val_ratio,
            "test": config.test_ratio,
        },
        "best_epoch": int(best_epoch),
        "test_metrics": test_metrics,
        "status_summary": summarize_metrics_for_status(test_metrics),
        "data_window": {
            "train_start": train_frame["match_date"].min().isoformat(),
            "train_end": train_frame["match_date"].max().isoformat(),
            "test_start": test_frame["match_date"].min().isoformat(),
            "test_end": test_frame["match_date"].max().isoformat(),
        },
        "training_params": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "diversity_margin": config.diversity_margin,
            "diversity_weight": config.diversity_weight,
            "winner_loss_weight": config.winner_loss_weight,
            "conformal_alpha": conformal_alpha,
        },
    }
    return model, metadata


def _aggregate_metrics(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not rows:
        return {"mean": {}, "std": {}}
    keys = sorted({key for row in rows for key in row.keys()})
    mean_block: dict[str, float] = {}
    std_block: dict[str, float] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        if not values:
            continue
        mean_block[key] = float(np.mean(values))
        std_block[key] = float(np.std(values))
    return {"mean": mean_block, "std": std_block}


def train_ensemble(
    *,
    frame: pd.DataFrame,
    config: MLConfig,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    soft_wta_temperature: float,
    encoder_type: str,
    patch_length: int,
    patch_stride: int,
    patch_model_dim: int,
    patch_layers: int,
    patch_attention_heads: int,
    conformal_alpha: float,
    ensemble_size: int,
    bootstrap_train: bool,
) -> tuple[list[tuple[TimeMCLModel, dict[str, Any]]], dict[str, Any]]:
    runs: list[tuple[TimeMCLModel, dict[str, Any]]] = []
    member_metrics: list[dict[str, float]] = []
    for member_idx in range(max(1, ensemble_size)):
        seed = int(config.random_seed + (member_idx * 97))
        model, metadata = train_model(
            frame=frame,
            config=config,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            soft_wta_temperature=soft_wta_temperature,
            encoder_type=encoder_type,
            patch_length=patch_length,
            patch_stride=patch_stride,
            patch_model_dim=patch_model_dim,
            patch_layers=patch_layers,
            patch_attention_heads=patch_attention_heads,
            conformal_alpha=conformal_alpha,
            member_seed=seed,
            bootstrap_train=bootstrap_train,
        )
        metadata["member_id"] = member_idx
        runs.append((model, metadata))
        member_metrics.append(metadata.get("test_metrics", {}))
        print(
            f"member={member_idx:02d} seed={seed} "
            f"test_best={metadata.get('test_metrics', {}).get('best_scenario_error', float('nan')):.3f} "
            f"test_wacc={metadata.get('test_metrics', {}).get('winner_accuracy', float('nan')):.3f}"
        )

    summary = {
        "generated_at": utc_now_iso(),
        "ensemble_size": len(runs),
        "bootstrap_train": bool(bootstrap_train),
        "metrics": _aggregate_metrics(member_metrics),
    }
    return runs, summary


def save_artifacts(model: TimeMCLModel, metadata: dict[str, Any], config: MLConfig) -> None:
    ensure_directories(config)
    evaluation_dir = config.artifacts_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state": model.state_dict(),
        **metadata,
    }
    checkpoint_path = config.artifacts_dir / "time_mcl.pt"
    torch.save(checkpoint, checkpoint_path)

    history_path = config.artifacts_dir / "training_metrics.json"
    history_path.write_text(json.dumps(metadata.get("history", []), indent=2), encoding="utf-8")

    run_path = config.artifacts_dir / "training_run.json"
    run_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    latest_eval_path = evaluation_dir / "latest_evaluation_summary.json"
    run_eval_path = evaluation_dir / f"{metadata.get('run_id', 'run')}_metrics.json"
    latest_eval_payload = {
        "generated_at": metadata.get("generated_at", utc_now_iso()),
        "source": "ml.train:test_split",
        "run_id": metadata.get("run_id"),
        "checkpoint_path": str(checkpoint_path),
        "encoder_type": metadata.get("encoder_type", "mlp"),
        "metrics": metadata.get("test_metrics", {}),
        "status_summary": metadata.get("status_summary", {}),
        "conformal_calibration": metadata.get("conformal_calibration", {}),
    }
    latest_eval_path.write_text(json.dumps(latest_eval_payload, indent=2), encoding="utf-8")
    run_eval_path.write_text(
        json.dumps(metadata.get("test_metrics", {}), indent=2),
        encoding="utf-8",
    )

    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved training history to {history_path}")
    print(f"Saved training run metadata to {run_path}")
    print(f"Saved latest evaluation summary to {latest_eval_path}")
    print(f"Saved run evaluation metrics to {run_eval_path}")


def save_ensemble_artifacts(
    *,
    runs: list[tuple[TimeMCLModel, dict[str, Any]]],
    ensemble_summary: dict[str, Any],
    config: MLConfig,
) -> None:
    ensure_directories(config)
    evaluation_dir = config.artifacts_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    ensemble_dir = config.ensemble_dir
    if ensemble_dir.exists():
        shutil.rmtree(ensemble_dir)
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    members_payload: list[dict[str, Any]] = []
    best_idx = 0
    best_error = float("inf")

    for idx, (model, metadata) in enumerate(runs):
        checkpoint_payload = {"model_state": model.state_dict(), **metadata}
        member_filename = f"member_{idx:02d}.pt"
        member_path = ensemble_dir / member_filename
        torch.save(checkpoint_payload, member_path)

        test_metrics = metadata.get("test_metrics", {})
        best_metric = float(test_metrics.get("best_scenario_error", float("inf")))
        if best_metric < best_error:
            best_error = best_metric
            best_idx = idx

        members_payload.append(
            {
                "member_id": idx,
                "seed": metadata.get("member_seed"),
                "checkpoint_path": str(member_path),
                "num_heads": metadata.get("num_heads"),
                "encoder_type": metadata.get("encoder_type"),
                "feature_columns": metadata.get("feature_columns", FEATURE_COLUMNS),
                "test_metrics": test_metrics,
            }
        )

    best_model, best_metadata = runs[best_idx]
    best_checkpoint = {"model_state": best_model.state_dict(), **best_metadata}
    checkpoint_path = config.artifacts_dir / "time_mcl.pt"
    torch.save(best_checkpoint, checkpoint_path)

    ensemble_manifest = {
        "generated_at": ensemble_summary.get("generated_at", utc_now_iso()),
        "run_id": f"ensemble_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}",
        "experiment_tag": config.experiment_tag,
        "ensemble_size": len(runs),
        "best_member_id": best_idx,
        "best_checkpoint_path": str(checkpoint_path),
        "metrics": ensemble_summary.get("metrics", {}),
        "members": members_payload,
        "bootstrap_train": ensemble_summary.get("bootstrap_train", False),
    }
    manifest_path = ensemble_dir / "ensemble_manifest.json"
    manifest_path.write_text(json.dumps(ensemble_manifest, indent=2), encoding="utf-8")

    run_path = config.artifacts_dir / "training_run.json"
    run_payload = {
        "generated_at": ensemble_manifest["generated_at"],
        "run_id": ensemble_manifest["run_id"],
        "mode": "ensemble",
        "ensemble_size": len(runs),
        "best_member_id": best_idx,
        "best_checkpoint_path": str(checkpoint_path),
        "ensemble_manifest_path": str(manifest_path),
        "metrics": ensemble_manifest["metrics"],
        "members": members_payload,
    }
    run_path.write_text(json.dumps(run_payload, indent=2), encoding="utf-8")

    history_rows = [metadata.get("history", []) for _, metadata in runs]
    history_path = config.artifacts_dir / "training_metrics.json"
    history_path.write_text(json.dumps(history_rows, indent=2), encoding="utf-8")

    latest_eval_path = evaluation_dir / "latest_evaluation_summary.json"
    latest_eval_payload = {
        "generated_at": ensemble_manifest["generated_at"],
        "source": "ml.train:ensemble_test_split",
        "run_id": ensemble_manifest["run_id"],
        "checkpoint_path": str(checkpoint_path),
        "ensemble_manifest_path": str(manifest_path),
        "metrics": ensemble_manifest["metrics"]["mean"],
        "metrics_std": ensemble_manifest["metrics"]["std"],
        "status_summary": summarize_metrics_for_status(ensemble_manifest["metrics"]["mean"]),
        "ensemble_size": len(runs),
        "best_member_id": best_idx,
    }
    latest_eval_path.write_text(json.dumps(latest_eval_payload, indent=2), encoding="utf-8")

    run_eval_path = evaluation_dir / f"{ensemble_manifest['run_id']}_metrics.json"
    run_eval_path.write_text(json.dumps(ensemble_manifest["metrics"], indent=2), encoding="utf-8")

    print(f"Saved ensemble manifest to {manifest_path}")
    print(f"Saved best checkpoint to {checkpoint_path}")
    print(f"Saved training run metadata to {run_path}")
    print(f"Saved latest evaluation summary to {latest_eval_path}")
    print(f"Saved run evaluation metrics to {run_eval_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TimeMCL-inspired IPL score model")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--soft-wta-temperature", type=float, default=0.0)
    parser.add_argument("--encoder-type", choices=["mlp", "patch"], default=None)
    parser.add_argument("--patch-length", type=int, default=None)
    parser.add_argument("--patch-stride", type=int, default=None)
    parser.add_argument("--patch-model-dim", type=int, default=None)
    parser.add_argument("--patch-layers", type=int, default=None)
    parser.add_argument("--patch-attention-heads", type=int, default=None)
    parser.add_argument("--conformal-alpha", type=float, default=None)
    parser.add_argument("--ensemble-size", type=int, default=None)
    parser.add_argument("--no-bootstrap", action="store_true")
    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        raise SystemExit(
            "Torch is not installed. Running in fallback mode (no trained model available). "
            "Install optional torch via scripts/setup_ml.sh (or `pip install -r ml/requirements-torch.txt`)."
        )

    config = MLConfig()
    ensure_directories(config)
    set_seeds(config.random_seed)

    encoder_type = args.encoder_type or config.encoder_type
    patch_length = int(args.patch_length or config.patch_length)
    patch_stride = int(args.patch_stride or config.patch_stride)
    patch_model_dim = int(args.patch_model_dim or config.patch_model_dim)
    patch_layers = int(args.patch_layers or config.patch_layers)
    patch_attention_heads = int(args.patch_attention_heads or config.patch_attention_heads)
    conformal_alpha = float(args.conformal_alpha if args.conformal_alpha is not None else config.conformal_alpha)
    ensemble_size = int(args.ensemble_size if args.ensemble_size is not None else config.ensemble_size)
    bootstrap_train = bool(config.ensemble_bootstrap and not args.no_bootstrap)

    frame = load_training_frame(config)
    if len(frame) < 40:
        raise RuntimeError("Need at least 40 training rows. Run ingestion/features pipeline first.")

    if ensemble_size <= 1:
        model, metadata = train_model(
            frame=frame,
            config=config,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            soft_wta_temperature=args.soft_wta_temperature,
            encoder_type=encoder_type,
            patch_length=patch_length,
            patch_stride=patch_stride,
            patch_model_dim=patch_model_dim,
            patch_layers=patch_layers,
            patch_attention_heads=patch_attention_heads,
            conformal_alpha=conformal_alpha,
            member_seed=config.random_seed,
            bootstrap_train=bootstrap_train,
        )
        save_artifacts(model, metadata, config)
        return

    runs, ensemble_summary = train_ensemble(
        frame=frame,
        config=config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        soft_wta_temperature=args.soft_wta_temperature,
        encoder_type=encoder_type,
        patch_length=patch_length,
        patch_stride=patch_stride,
        patch_model_dim=patch_model_dim,
        patch_layers=patch_layers,
        patch_attention_heads=patch_attention_heads,
        conformal_alpha=conformal_alpha,
        ensemble_size=ensemble_size,
        bootstrap_train=bootstrap_train,
    )
    save_ensemble_artifacts(runs=runs, ensemble_summary=ensemble_summary, config=config)


if __name__ == "__main__":
    main()
