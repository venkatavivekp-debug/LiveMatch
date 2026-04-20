from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class DatasetRegisterRequest(BaseModel):
    dataset_key: str = Field(min_length=3, max_length=96)
    sport: str = Field(min_length=3, max_length=32)
    tournament: Optional[str] = Field(default=None, max_length=64)
    source_type: str = Field(default="local", min_length=2, max_length=48)
    source_uri: Optional[str] = Field(default=None, max_length=512)
    manifest_path: Optional[str] = Field(default=None, max_length=512)
    row_count: int = Field(default=0, ge=0)
    schema_version: str = Field(default="1.0", min_length=1, max_length=32)
    details: dict[str, Any] = Field(default_factory=dict)


class DatasetResponse(BaseModel):
    dataset_key: str
    sport: str
    tournament: Optional[str] = None
    source_type: str
    source_uri: Optional[str] = None
    manifest_path: Optional[str] = None
    row_count: int
    schema_version: str
    details: dict[str, Any]
    created_at: datetime


class ExperimentCreateRequest(BaseModel):
    name: str = Field(min_length=3, max_length=160)
    sport: str = Field(min_length=3, max_length=32)
    task: str = Field(default="match_forecast", min_length=3, max_length=64)
    dataset_key: Optional[str] = Field(default=None, max_length=96)
    config: dict[str, Any] = Field(default_factory=dict)
    notes: Optional[str] = Field(default=None, max_length=2500)


class ExperimentResponse(BaseModel):
    experiment_id: str
    name: str
    sport: str
    task: str
    status: str
    dataset_key: Optional[str] = None
    config: dict[str, Any]
    notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class TrainingStartRequest(BaseModel):
    experiment_id: Optional[str] = Field(default=None, max_length=96)
    epochs: int = Field(default=60, ge=1, le=500)
    batch_size: int = Field(default=32, ge=4, le=512)
    learning_rate: float = Field(default=1e-3, gt=1e-6, le=1.0)
    weight_decay: float = Field(default=1e-4, ge=0.0, le=1.0)
    soft_wta_temperature: float = Field(default=0.0, ge=0.0, le=10.0)
    encoder_type: Optional[str] = Field(default=None, max_length=16)
    patch_length: Optional[int] = Field(default=None, ge=2, le=64)
    patch_stride: Optional[int] = Field(default=None, ge=1, le=32)
    patch_model_dim: Optional[int] = Field(default=None, ge=8, le=512)
    patch_layers: Optional[int] = Field(default=None, ge=1, le=12)
    patch_attention_heads: Optional[int] = Field(default=None, ge=1, le=16)
    conformal_alpha: Optional[float] = Field(default=None, gt=0.0, lt=0.5)
    sport: str = Field(default="cricket", min_length=3, max_length=32)
    notes: Optional[str] = Field(default=None, max_length=1000)


class TrainingJobResponse(BaseModel):
    job_id: str
    experiment_id: Optional[str] = None
    status: str
    mode: str
    message: Optional[str] = None
    request: dict[str, Any]
    output: dict[str, Any]
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


class MetricRow(BaseModel):
    metric_name: str
    metric_value: float
    details: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class ExperimentMetricsResponse(BaseModel):
    experiment_id: str
    metrics: list[MetricRow]
