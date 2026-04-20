from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MLConfig:
    repo_root: Path = Path(__file__).resolve().parents[1]
    raw_data_dir: Path = repo_root / "data" / "raw"
    processed_data_dir: Path = repo_root / "data" / "processed"
    artifacts_dir: Path = repo_root / "ml" / "artifacts"
    ensemble_dir: Path = artifacts_dir / "ensemble"
    cricsheet_ipl_json_url: str = "https://cricsheet.org/downloads/ipl_json.zip"
    random_seed: int = 42
    ensemble_size: int = 5
    ensemble_bootstrap: bool = True
    num_heads: int = 4
    soft_wta_temperature: float = 0.0
    encoder_type: str = "mlp"
    patch_length: int = 4
    patch_stride: int = 2
    patch_model_dim: int = 64
    patch_layers: int = 2
    patch_attention_heads: int = 4
    conformal_alpha: float = 0.1
    history_window: int = 5
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    hidden_dims: tuple[int, int] = (128, 64)
    dropout: float = 0.1
    diversity_margin: float = 12.0
    diversity_weight: float = 0.05
    winner_loss_weight: float = 0.35
    experiment_tag: str = "timemcl_research"


FEATURE_COLUMNS: list[str] = [
    "team_a_avg_runs_last_5",
    "team_a_avg_runs_last_10",
    "team_a_avg_wickets_last_5",
    "team_a_run_rate_trend",
    "team_b_avg_runs_last_5",
    "team_b_avg_runs_last_10",
    "team_b_avg_wickets_last_5",
    "team_b_run_rate_trend",
    "team_a_win_rate_vs_b",
    "avg_score_team_a_vs_b",
    "avg_score_team_b_vs_a",
    "venue_avg_score",
    "venue_chase_success_rate",
    "venue_defend_bias",
    "team_a_runs_vs_opponent_avg",
    "team_b_runs_vs_opponent_avg",
    "batting_first",
    "team_a_bats_first",
    "team_b_bats_first",
    "team_a_chase_success_rate",
    "team_b_chase_success_rate",
    "team_a_defend_success_rate",
    "team_b_defend_success_rate",
    "chase_defend_edge_team_a_first",
    "chase_defend_edge_team_b_first",
    "venue_batting_first_advantage",
    "recent_form_diff",
    "recent_run_rate_diff",
    "head_to_head_win_diff",
    "wickets_taken_diff",
]

def ensure_directories(config: MLConfig) -> None:
    config.raw_data_dir.mkdir(parents=True, exist_ok=True)
    config.processed_data_dir.mkdir(parents=True, exist_ok=True)
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    config.ensemble_dir.mkdir(parents=True, exist_ok=True)
