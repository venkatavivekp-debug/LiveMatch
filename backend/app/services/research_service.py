from __future__ import annotations

import json
import logging
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.repositories.research_repository import ResearchRepository
from app.db.models import (
    DatasetRecord,
    EvaluationResultRecord,
    ExperimentRecord,
    ModelArtifactRecord,
    TrainingJobRecord,
)
from app.db.session import SessionLocal

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.utcnow()


def _json_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    return {"value": payload}


class ResearchService:
    @staticmethod
    def _flatten_numeric_metrics(payload: dict[str, Any], prefix: str = "") -> list[tuple[str, float]]:
        rows: list[tuple[str, float]] = []
        for key, value in payload.items():
            metric_key = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, (int, float)):
                rows.append((metric_key, float(value)))
            elif isinstance(value, dict):
                rows.extend(ResearchService._flatten_numeric_metrics(value, prefix=metric_key))
        return rows

    @staticmethod
    def register_dataset(db: Session, payload: dict[str, Any]) -> DatasetRecord:
        key = str(payload["dataset_key"]).strip().lower().replace(" ", "_")
        existing = ResearchRepository.get_dataset_by_key(db=db, dataset_key=key)

        if existing is None:
            existing = DatasetRecord(
                dataset_key=key,
                sport=str(payload["sport"]).lower().strip(),
                tournament=(payload.get("tournament") or None),
                source_type=str(payload.get("source_type") or "local"),
                source_uri=(payload.get("source_uri") or None),
                manifest_path=(payload.get("manifest_path") or None),
                row_count=int(payload.get("row_count") or 0),
                schema_version=str(payload.get("schema_version") or "1.0"),
                details_json=_json_dict(payload.get("details") or {}),
            )
            db.add(existing)
        else:
            existing.sport = str(payload["sport"]).lower().strip()
            existing.tournament = payload.get("tournament") or None
            existing.source_type = str(payload.get("source_type") or existing.source_type)
            existing.source_uri = payload.get("source_uri") or None
            existing.manifest_path = payload.get("manifest_path") or None
            existing.row_count = int(payload.get("row_count") or existing.row_count)
            existing.schema_version = str(payload.get("schema_version") or existing.schema_version)
            existing.details_json = _json_dict(payload.get("details") or existing.details_json)

        db.commit()
        db.refresh(existing)
        return existing

    @staticmethod
    def list_datasets(db: Session, sport: Optional[str] = None) -> list[DatasetRecord]:
        return ResearchRepository.list_datasets(db=db, sport=sport)

    @staticmethod
    def create_experiment(db: Session, payload: dict[str, Any]) -> ExperimentRecord:
        experiment_id = f"exp_{_now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
        record = ExperimentRecord(
            experiment_id=experiment_id,
            name=str(payload["name"]).strip(),
            sport=str(payload["sport"]).lower().strip(),
            task=str(payload.get("task") or "match_forecast").strip(),
            status="created",
            dataset_key=(payload.get("dataset_key") or None),
            config_json=_json_dict(payload.get("config") or {}),
            notes=(payload.get("notes") or None),
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record

    @staticmethod
    def list_experiments(db: Session, sport: Optional[str] = None, status: Optional[str] = None) -> list[ExperimentRecord]:
        return ResearchRepository.list_experiments(db=db, sport=sport, status=status)

    @staticmethod
    def get_experiment(db: Session, experiment_id: str) -> Optional[ExperimentRecord]:
        return ResearchRepository.get_experiment(db=db, experiment_id=experiment_id)

    @classmethod
    def create_training_job(cls, db: Session, payload: dict[str, Any]) -> TrainingJobRecord:
        settings = get_settings()
        job = TrainingJobRecord(
            job_id=f"job_{_now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}",
            experiment_id=(payload.get("experiment_id") or None),
            status="queued",
            mode=settings.training_job_mode,
            message="Training job accepted.",
            request_json=_json_dict(payload),
            output_json={},
        )
        db.add(job)

        if payload.get("experiment_id"):
            exp = cls.get_experiment(db, str(payload["experiment_id"]))
            if exp is not None:
                exp.status = "queued"
                exp.updated_at = _now()

        db.commit()
        db.refresh(job)
        return job

    @classmethod
    def run_training_job(cls, job_id: str) -> None:
        settings = get_settings()
        db = SessionLocal()
        try:
            job = ResearchRepository.get_training_job(db=db, job_id=job_id)
            if job is None:
                logger.error("Training job %s not found", job_id)
                return

            job.status = "running"
            job.started_at = _now()
            job.updated_at = _now()
            job.message = "Training job started."
            db.commit()

            request_payload = _json_dict(job.request_json)
            mode = settings.training_job_mode.strip().lower()

            if mode not in {"subprocess", "dry-run"}:
                mode = "dry-run"

            if mode == "subprocess":
                result_payload = cls._run_subprocess_training(request_payload)
            else:
                result_payload = cls._run_dry_training(request_payload)

            job.output_json = result_payload
            job.status = "completed"
            job.finished_at = _now()
            job.updated_at = _now()
            job.message = result_payload.get("message", "Training finished.")

            experiment_id = job.experiment_id
            if experiment_id:
                exp = cls.get_experiment(db, experiment_id)
                if exp is not None:
                    exp.status = "completed"
                    exp.updated_at = _now()

                artifact_id = cls._sync_model_registry(db=db, experiment_id=experiment_id, output=result_payload)
                cls._sync_metric_rows(
                    db=db,
                    experiment_id=experiment_id,
                    output=result_payload,
                    artifact_id=artifact_id,
                )

            db.commit()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Training job %s failed: %s", job_id, exc)
            if db is not None:
                job = ResearchRepository.get_training_job(db=db, job_id=job_id)
                if job is not None:
                    job.status = "failed"
                    job.finished_at = _now()
                    job.updated_at = _now()
                    job.message = f"Training failed: {exc}"
                    job.output_json = {"error": str(exc)}

                    if job.experiment_id:
                        exp = cls.get_experiment(db, job.experiment_id)
                        if exp is not None:
                            exp.status = "failed"
                            exp.updated_at = _now()
                    db.commit()
        finally:
            db.close()

    @staticmethod
    def _run_dry_training(request_payload: dict[str, Any]) -> dict[str, Any]:
        settings = get_settings()
        checkpoint = settings.checkpoint_path
        metrics_path = settings.ml_artifacts_dir / "training_metrics.json"
        evaluation_path = settings.ml_artifacts_dir / "evaluation_metrics.json"
        latest_eval_path = settings.latest_evaluation_summary_path
        training_run_path = settings.training_run_path

        data_ready = (settings.data_processed_dir / "model_features.csv").exists()
        run_id = None
        model_version = None
        if training_run_path.exists():
            try:
                run_payload = json.loads(training_run_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                run_payload = {}
            run_id = run_payload.get("run_id")
            model_version = run_payload.get("run_id")

        return {
            "mode": "dry-run",
            "message": (
                "Dry-run completed. Use ml environment to run `python -m ml.train` for a full training pass."
            ),
            "data_ready": data_ready,
            "checkpoint_exists": checkpoint.exists(),
            "checkpoint_path": str(checkpoint),
            "metrics_path": str(metrics_path) if metrics_path.exists() else None,
            "evaluation_path": str(latest_eval_path) if latest_eval_path.exists() else (str(evaluation_path) if evaluation_path.exists() else None),
            "training_run_path": str(training_run_path) if training_run_path.exists() else None,
            "run_id": run_id,
            "model_version": model_version,
            "request": request_payload,
        }

    @staticmethod
    def _run_subprocess_training(request_payload: dict[str, Any]) -> dict[str, Any]:
        settings = get_settings()
        repo_root = settings.repo_root
        command = [
            sys.executable,
            "-m",
            "ml.train",
            "--epochs",
            str(request_payload.get("epochs", 60)),
            "--batch-size",
            str(request_payload.get("batch_size", 32)),
            "--lr",
            str(request_payload.get("learning_rate", 1e-3)),
            "--weight-decay",
            str(request_payload.get("weight_decay", 1e-4)),
            "--soft-wta-temperature",
            str(request_payload.get("soft_wta_temperature", 0.0)),
        ]
        optional_arg_map = {
            "encoder_type": "--encoder-type",
            "patch_length": "--patch-length",
            "patch_stride": "--patch-stride",
            "patch_model_dim": "--patch-model-dim",
            "patch_layers": "--patch-layers",
            "patch_attention_heads": "--patch-attention-heads",
            "conformal_alpha": "--conformal-alpha",
        }
        for request_key, arg_flag in optional_arg_map.items():
            value = request_payload.get(request_key)
            if value is None:
                continue
            command.extend([arg_flag, str(value)])

        result = subprocess.run(
            command,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=settings.training_subprocess_timeout_seconds,
        )

        output: dict[str, Any] = {
            "mode": "subprocess",
            "command": command,
            "return_code": int(result.returncode),
            "stdout_tail": result.stdout[-5000:],
            "stderr_tail": result.stderr[-5000:],
        }

        checkpoint_path = settings.checkpoint_path
        output["checkpoint_path"] = str(checkpoint_path)
        output["checkpoint_exists"] = checkpoint_path.exists()

        metrics_path = settings.ml_artifacts_dir / "training_metrics.json"
        output["metrics_path"] = str(metrics_path) if metrics_path.exists() else None

        eval_path = settings.ml_artifacts_dir / "evaluation_metrics.json"
        latest_eval_path = settings.latest_evaluation_summary_path
        training_run_path = settings.training_run_path
        if result.returncode == 0:
            eval_command = [sys.executable, "-m", "ml.evaluate"]
            eval_result = subprocess.run(
                eval_command,
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=settings.training_subprocess_timeout_seconds,
            )
            output["evaluation_command"] = eval_command
            output["evaluation_return_code"] = int(eval_result.returncode)
            output["evaluation_stdout_tail"] = eval_result.stdout[-5000:]
            output["evaluation_stderr_tail"] = eval_result.stderr[-5000:]
            output["evaluation_path"] = str(eval_path) if eval_path.exists() else None
            if latest_eval_path.exists():
                output["evaluation_path"] = str(latest_eval_path)
            if training_run_path.exists():
                output["training_run_path"] = str(training_run_path)
                try:
                    run_payload = json.loads(training_run_path.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    run_payload = {}
                output["run_id"] = run_payload.get("run_id")
                if run_payload.get("run_id"):
                    output["model_version"] = run_payload.get("run_id")

        if result.returncode != 0:
            raise RuntimeError(
                "Training subprocess failed. "
                f"stderr={result.stderr[-300:]}"
            )

        output["message"] = "Training subprocess completed."
        return output

    @staticmethod
    def _sync_model_registry(db: Session, experiment_id: str, output: dict[str, Any]) -> str | None:
        checkpoint_path = output.get("checkpoint_path")
        if not checkpoint_path:
            return None

        checkpoint = Path(str(checkpoint_path))
        if not checkpoint.exists():
            return None

        artifact_id = f"artifact_{_now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
        latest_summary_path = output.get("evaluation_path")
        training_run_path = output.get("training_run_path")
        artifact = ModelArtifactRecord(
            artifact_id=artifact_id,
            experiment_id=experiment_id,
            model_name="time_mcl",
            sport="cricket",
            version=str(output.get("model_version") or f"v{_now():%Y%m%d%H%M}"),
            checkpoint_path=str(checkpoint),
            metrics_path=str(latest_summary_path or output.get("metrics_path")) if (latest_summary_path or output.get("metrics_path")) else None,
            is_active=True,
            details_json={
                "mode": output.get("mode", "dry-run"),
                "run_id": output.get("run_id"),
                "training_run_path": training_run_path,
                "checkpoint_exists": bool(output.get("checkpoint_exists", checkpoint.exists())),
            },
        )

        active = ResearchRepository.list_active_model_artifacts(db=db, model_name="time_mcl", sport="cricket")
        for row in active:
            row.is_active = False

        db.add(artifact)
        return artifact_id

    @staticmethod
    def _sync_metric_rows(
        db: Session,
        experiment_id: str,
        output: dict[str, Any],
        artifact_id: str | None = None,
    ) -> None:
        metrics_path = output.get("evaluation_path")
        if not metrics_path:
            return

        path = Path(str(metrics_path))
        if not path.exists():
            return

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            logger.warning("Could not parse metrics json at %s", path)
            return

        metric_rows = ResearchService._flatten_numeric_metrics(payload)
        for metric_name, metric_value in metric_rows:
            row = EvaluationResultRecord(
                experiment_id=experiment_id,
                artifact_id=artifact_id,
                metric_name=metric_name,
                metric_value=metric_value,
                metric_json={"source": str(path), "run_id": output.get("run_id")},
            )
            db.add(row)

    @staticmethod
    def list_training_jobs(db: Session, experiment_id: Optional[str] = None) -> list[TrainingJobRecord]:
        return ResearchRepository.list_training_jobs(db=db, experiment_id=experiment_id)

    @staticmethod
    def get_training_job(db: Session, job_id: str) -> Optional[TrainingJobRecord]:
        return ResearchRepository.get_training_job(db=db, job_id=job_id)

    @staticmethod
    def list_metrics(db: Session, experiment_id: str) -> list[EvaluationResultRecord]:
        return ResearchRepository.list_metrics(db=db, experiment_id=experiment_id)
