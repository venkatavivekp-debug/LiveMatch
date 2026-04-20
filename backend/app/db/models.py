from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class DatasetRecord(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    dataset_key: Mapped[str] = mapped_column(String(96), unique=True, index=True, nullable=False)
    sport: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    tournament: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    source_type: Mapped[str] = mapped_column(String(48), nullable=False, default="local")
    source_uri: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    manifest_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    row_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    schema_version: Mapped[str] = mapped_column(String(32), default="1.0", nullable=False)
    details_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class ExperimentRecord(Base):
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    experiment_id: Mapped[str] = mapped_column(String(96), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    sport: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    task: Mapped[str] = mapped_column(String(64), default="match_forecast", nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="created", index=True, nullable=False)
    dataset_key: Mapped[Optional[str]] = mapped_column(String(96), nullable=True)
    config_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )


class TrainingJobRecord(Base):
    __tablename__ = "training_jobs"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    job_id: Mapped[str] = mapped_column(String(96), unique=True, index=True, nullable=False)
    experiment_id: Mapped[Optional[str]] = mapped_column(String(96), index=True, nullable=True)
    status: Mapped[str] = mapped_column(String(32), index=True, nullable=False, default="queued")
    mode: Mapped[str] = mapped_column(String(32), default="dry-run", nullable=False)
    message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    request_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    output_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )


class ModelArtifactRecord(Base):
    __tablename__ = "model_artifacts"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    artifact_id: Mapped[str] = mapped_column(String(96), unique=True, index=True, nullable=False)
    experiment_id: Mapped[Optional[str]] = mapped_column(String(96), index=True, nullable=True)
    model_name: Mapped[str] = mapped_column(String(96), nullable=False)
    sport: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    version: Mapped[str] = mapped_column(String(32), default="v1", nullable=False)
    checkpoint_path: Mapped[str] = mapped_column(String(512), nullable=False)
    metrics_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    details_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class EvaluationResultRecord(Base):
    __tablename__ = "evaluation_results"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    experiment_id: Mapped[str] = mapped_column(String(96), index=True, nullable=False)
    artifact_id: Mapped[Optional[str]] = mapped_column(String(96), index=True, nullable=True)
    metric_name: Mapped[str] = mapped_column(String(64), nullable=False)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    metric_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class PredictionResidualRecord(Base):
    __tablename__ = "prediction_residuals"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    record_id: Mapped[str] = mapped_column(String(96), unique=True, index=True, nullable=False)
    match_id: Mapped[Optional[str]] = mapped_column(String(120), index=True, nullable=True)
    sport: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    tournament: Mapped[Optional[str]] = mapped_column(String(64), index=True, nullable=True)
    team: Mapped[Optional[str]] = mapped_column(String(120), index=True, nullable=True)
    opponent: Mapped[Optional[str]] = mapped_column(String(120), index=True, nullable=True)
    venue: Mapped[Optional[str]] = mapped_column(String(160), index=True, nullable=True)
    match_state: Mapped[Optional[str]] = mapped_column(String(32), index=True, nullable=True)
    scenario: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    predicted_mean: Mapped[float] = mapped_column(Float, nullable=False)
    interval_low: Mapped[float] = mapped_column(Float, nullable=False)
    interval_high: Mapped[float] = mapped_column(Float, nullable=False)
    actual_value: Mapped[float] = mapped_column(Float, nullable=False)
    best_head_value: Mapped[float] = mapped_column(Float, nullable=False)
    error_value: Mapped[float] = mapped_column(Float, nullable=False)
    residual_value: Mapped[float] = mapped_column(Float, nullable=False)
    data_mode: Mapped[Optional[str]] = mapped_column(String(24), nullable=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
