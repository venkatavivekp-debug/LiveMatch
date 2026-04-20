from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.research import (
    DatasetRegisterRequest,
    DatasetResponse,
    ExperimentCreateRequest,
    ExperimentMetricsResponse,
    ExperimentResponse,
    MetricRow,
    TrainingJobResponse,
    TrainingStartRequest,
)
from app.services.research_service import ResearchService

router = APIRouter(tags=["research"])


def _dataset_to_response(record: object) -> DatasetResponse:
    return DatasetResponse(
        dataset_key=getattr(record, "dataset_key"),
        sport=getattr(record, "sport"),
        tournament=getattr(record, "tournament"),
        source_type=getattr(record, "source_type"),
        source_uri=getattr(record, "source_uri"),
        manifest_path=getattr(record, "manifest_path"),
        row_count=int(getattr(record, "row_count")),
        schema_version=getattr(record, "schema_version"),
        details=dict(getattr(record, "details_json") or {}),
        created_at=getattr(record, "created_at"),
    )


def _experiment_to_response(record: object) -> ExperimentResponse:
    return ExperimentResponse(
        experiment_id=getattr(record, "experiment_id"),
        name=getattr(record, "name"),
        sport=getattr(record, "sport"),
        task=getattr(record, "task"),
        status=getattr(record, "status"),
        dataset_key=getattr(record, "dataset_key"),
        config=dict(getattr(record, "config_json") or {}),
        notes=getattr(record, "notes"),
        created_at=getattr(record, "created_at"),
        updated_at=getattr(record, "updated_at"),
    )


def _job_to_response(record: object) -> TrainingJobResponse:
    return TrainingJobResponse(
        job_id=getattr(record, "job_id"),
        experiment_id=getattr(record, "experiment_id"),
        status=getattr(record, "status"),
        mode=getattr(record, "mode"),
        message=getattr(record, "message"),
        request=dict(getattr(record, "request_json") or {}),
        output=dict(getattr(record, "output_json") or {}),
        started_at=getattr(record, "started_at"),
        finished_at=getattr(record, "finished_at"),
        created_at=getattr(record, "created_at"),
        updated_at=getattr(record, "updated_at"),
    )


@router.post("/datasets/register", response_model=DatasetResponse)
def register_dataset(
    payload: DatasetRegisterRequest,
    db: Session = Depends(get_db),
) -> DatasetResponse:
    record = ResearchService.register_dataset(db=db, payload=payload.model_dump())
    return _dataset_to_response(record)


@router.get("/datasets/list", response_model=list[DatasetResponse])
def list_datasets(
    sport: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
) -> list[DatasetResponse]:
    rows = ResearchService.list_datasets(db=db, sport=sport)
    return [_dataset_to_response(row) for row in rows]


@router.post("/experiments/create", response_model=ExperimentResponse)
def create_experiment(
    payload: ExperimentCreateRequest,
    db: Session = Depends(get_db),
) -> ExperimentResponse:
    if payload.dataset_key:
        known_datasets = ResearchService.list_datasets(db=db)
        known_keys = {row.dataset_key for row in known_datasets}
        if payload.dataset_key.lower().strip() not in known_keys:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset '{payload.dataset_key}' is not registered",
            )

    record = ResearchService.create_experiment(db=db, payload=payload.model_dump())
    return _experiment_to_response(record)


@router.get("/experiments/list", response_model=list[ExperimentResponse])
def list_experiments(
    sport: Optional[str] = Query(default=None),
    status_filter: Optional[str] = Query(default=None, alias="status"),
    db: Session = Depends(get_db),
) -> list[ExperimentResponse]:
    rows = ResearchService.list_experiments(db=db, sport=sport, status=status_filter)
    return [_experiment_to_response(row) for row in rows]


@router.post("/training/start", response_model=TrainingJobResponse)
def start_training(
    payload: TrainingStartRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> TrainingJobResponse:
    if payload.experiment_id:
        experiment = ResearchService.get_experiment(db=db, experiment_id=payload.experiment_id)
        if experiment is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment '{payload.experiment_id}' not found",
            )

    job = ResearchService.create_training_job(db=db, payload=payload.model_dump())
    background_tasks.add_task(ResearchService.run_training_job, job.job_id)
    return _job_to_response(job)


@router.get("/training/status/{job_id}", response_model=TrainingJobResponse)
def training_status(
    job_id: str,
    db: Session = Depends(get_db),
) -> TrainingJobResponse:
    row = ResearchService.get_training_job(db=db, job_id=job_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Training job '{job_id}' not found")
    return _job_to_response(row)


@router.get("/metrics/{experiment_id}", response_model=ExperimentMetricsResponse)
def experiment_metrics(
    experiment_id: str,
    db: Session = Depends(get_db),
) -> ExperimentMetricsResponse:
    experiment = ResearchService.get_experiment(db=db, experiment_id=experiment_id)
    if experiment is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found")

    rows = ResearchService.list_metrics(db=db, experiment_id=experiment_id)
    return ExperimentMetricsResponse(
        experiment_id=experiment_id,
        metrics=[
            MetricRow(
                metric_name=row.metric_name,
                metric_value=row.metric_value,
                details=dict(row.metric_json or {}),
                created_at=row.created_at,
            )
            for row in rows
        ],
    )
