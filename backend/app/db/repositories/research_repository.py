from __future__ import annotations

from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import (
    DatasetRecord,
    EvaluationResultRecord,
    ExperimentRecord,
    ModelArtifactRecord,
    TrainingJobRecord,
)


class ResearchRepository:
    @staticmethod
    def get_dataset_by_key(db: Session, dataset_key: str) -> Optional[DatasetRecord]:
        return db.scalar(select(DatasetRecord).where(DatasetRecord.dataset_key == dataset_key))

    @staticmethod
    def list_datasets(db: Session, sport: Optional[str] = None) -> list[DatasetRecord]:
        statement = select(DatasetRecord).order_by(DatasetRecord.created_at.desc())
        if sport:
            statement = statement.where(DatasetRecord.sport == sport.lower().strip())
        return list(db.scalars(statement).all())

    @staticmethod
    def add_dataset(db: Session, row: DatasetRecord) -> DatasetRecord:
        db.add(row)
        db.commit()
        db.refresh(row)
        return row

    @staticmethod
    def save(db: Session, row: object) -> None:
        db.add(row)
        db.commit()

    @staticmethod
    def get_experiment(db: Session, experiment_id: str) -> Optional[ExperimentRecord]:
        return db.scalar(select(ExperimentRecord).where(ExperimentRecord.experiment_id == experiment_id))

    @staticmethod
    def list_experiments(
        db: Session,
        sport: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list[ExperimentRecord]:
        statement = select(ExperimentRecord).order_by(ExperimentRecord.updated_at.desc())
        if sport:
            statement = statement.where(ExperimentRecord.sport == sport.lower().strip())
        if status:
            statement = statement.where(ExperimentRecord.status == status.lower().strip())
        return list(db.scalars(statement).all())

    @staticmethod
    def get_training_job(db: Session, job_id: str) -> Optional[TrainingJobRecord]:
        return db.scalar(select(TrainingJobRecord).where(TrainingJobRecord.job_id == job_id))

    @staticmethod
    def list_training_jobs(db: Session, experiment_id: Optional[str] = None) -> list[TrainingJobRecord]:
        statement = select(TrainingJobRecord).order_by(TrainingJobRecord.created_at.desc())
        if experiment_id:
            statement = statement.where(TrainingJobRecord.experiment_id == experiment_id)
        return list(db.scalars(statement).all())

    @staticmethod
    def list_metrics(db: Session, experiment_id: str) -> list[EvaluationResultRecord]:
        statement = (
            select(EvaluationResultRecord)
            .where(EvaluationResultRecord.experiment_id == experiment_id)
            .order_by(EvaluationResultRecord.created_at.desc())
        )
        return list(db.scalars(statement).all())

    @staticmethod
    def list_active_model_artifacts(db: Session, model_name: str, sport: str) -> list[ModelArtifactRecord]:
        statement = select(ModelArtifactRecord).where(
            ModelArtifactRecord.model_name == model_name,
            ModelArtifactRecord.sport == sport,
            ModelArtifactRecord.is_active.is_(True),
        )
        return list(db.scalars(statement).all())

    @staticmethod
    def latest_active_model_artifact(db: Session) -> Optional[ModelArtifactRecord]:
        statement = (
            select(ModelArtifactRecord)
            .where(ModelArtifactRecord.is_active.is_(True))
            .order_by(ModelArtifactRecord.created_at.desc())
        )
        return db.scalar(statement)
