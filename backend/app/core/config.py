from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    project_name: str = "LiveMatch API"
    api_v1_prefix: str = ""
    secret_key: str = "change-me-in-production"
    token_algorithm: str = "HS256"
    access_token_expire_minutes: int = 120
    database_url: str = "sqlite:///./livematch.db"
    predictor_num_heads: int = 3
    max_batch_predictions: int = 24
    data_provider: str = "local-demo"
    realtime_provider: str = "cricapi"
    live_cricket_provider: str = "cricapi"
    live_cricket_api_base_url: str = "https://api.cricapi.com/v1"
    live_cricket_api_key: str = ""
    live_provider_timeout_seconds: float = 8.0
    live_provider_max_retries: int = 2
    live_provider_backoff_seconds: float = 0.8
    live_cache_ttl_seconds: int = 600
    live_cache_stale_ttl_seconds: int = 5400
    live_cache_file: str = "data/external/live_provider_cache.json"
    live_sync_status_file: str = "data/external/live_sync_status.json"
    live_feature_recency_weight: float = 0.65
    ml_checkpoint_path: str = "ml/artifacts/time_mcl.pt"
    ml_manifest_path: str = "data/processed/feature_manifest.json"
    training_job_mode: str = "dry-run"
    training_subprocess_timeout_seconds: int = 5400

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)

    @property
    def repo_root(self) -> Path:
        return Path(__file__).resolve().parents[3]

    @property
    def data_processed_dir(self) -> Path:
        return self.repo_root / "data" / "processed"

    @property
    def ml_artifacts_dir(self) -> Path:
        return self.repo_root / "ml" / "artifacts"

    @property
    def raw_data_dir(self) -> Path:
        return self.repo_root / "data" / "raw"

    @property
    def data_external_dir(self) -> Path:
        return self.repo_root / "data" / "external"

    @property
    def latest_evaluation_summary_path(self) -> Path:
        return self.ml_artifacts_dir / "latest_evaluation_summary.json"

    @property
    def training_run_path(self) -> Path:
        return self.ml_artifacts_dir / "training_run.json"

    def _resolve_repo_path(self, raw_path: str) -> Path:
        path = Path(raw_path.strip())
        if path.is_absolute():
            return path

        normalized = raw_path.strip()
        while normalized.startswith("../"):
            normalized = normalized[3:]
        if normalized.startswith("./"):
            normalized = normalized[2:]

        return (self.repo_root / normalized).resolve()

    @property
    def checkpoint_path(self) -> Path:
        return self._resolve_repo_path(self.ml_checkpoint_path)

    @property
    def manifest_path(self) -> Path:
        return self._resolve_repo_path(self.ml_manifest_path)

    @property
    def live_cache_path(self) -> Path:
        return self._resolve_repo_path(self.live_cache_file)

    @property
    def live_sync_status_path(self) -> Path:
        return self._resolve_repo_path(self.live_sync_status_file)


@lru_cache
def get_settings() -> Settings:
    return Settings()
