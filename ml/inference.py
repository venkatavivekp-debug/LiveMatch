from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.calibration import apply_conformal_interval
from ml.config import FEATURE_COLUMNS, MLConfig
from ml.features import run_feature_pipeline
from ml.live_features import blend_cricket_features

try:
    import torch
    from ml.model import TimeMCLModel

    TORCH_AVAILABLE = True
except Exception:  # noqa: BLE001
    torch = None  # type: ignore[assignment]
    TimeMCLModel = Any  # type: ignore[assignment]
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)
PLACEHOLDER_PLAYER_PATTERN = re.compile(r"\bplayer\s*\d+\b", re.IGNORECASE)

CRICKET_FALLBACK_PLAYERS: dict[str, list[str]] = {
    "mumbai indians": ["Rohit Sharma", "Ishan Kishan", "Suryakumar Yadav", "Jasprit Bumrah"],
    "chennai super kings": ["Ruturaj Gaikwad", "Shivam Dube", "Ravindra Jadeja", "Deepak Chahar"],
    "royal challengers bengaluru": ["Virat Kohli", "Faf du Plessis", "Glenn Maxwell", "Mohammed Siraj"],
    "kolkata knight riders": ["Shreyas Iyer", "Andre Russell", "Sunil Narine", "Mitchell Starc"],
    "rajasthan royals": ["Sanju Samson", "Yashasvi Jaiswal", "Jos Buttler", "Yuzvendra Chahal"],
    "sunrisers hyderabad": ["Travis Head", "Abhishek Sharma", "Pat Cummins", "Bhuvneshwar Kumar"],
    "delhi capitals": ["Rishabh Pant", "Jake Fraser-McGurk", "Axar Patel", "Kuldeep Yadav"],
    "punjab kings": ["Shikhar Dhawan", "Liam Livingstone", "Sam Curran", "Arshdeep Singh"],
    "lucknow super giants": ["KL Rahul", "Nicholas Pooran", "Marcus Stoinis", "Ravi Bishnoi"],
    "gujarat titans": ["Shubman Gill", "Sai Sudharsan", "Rashid Khan", "Mohit Sharma"],
}

CRICKET_BOWLER_NAMES = {
    "jasprit bumrah",
    "piyush chawla",
    "gerald coetzee",
    "nuwan thushara",
    "akash madhwal",
    "mohammed siraj",
    "yash dayal",
    "karn sharma",
    "reece topley",
    "mayank dagar",
    "varun chakaravarthy",
    "mitchell starc",
    "harshit rana",
    "anukul roy",
    "ravichandran ashwin",
    "trent boult",
    "yuzvendra chahal",
    "sandeep sharma",
    "avesh khan",
    "pat cummins",
    "bhuvneshwar kumar",
    "t natarajan",
    "mayank markande",
    "axar patel",
    "kuldeep yadav",
    "khaleel ahmed",
    "anrich nortje",
    "mukesh kumar",
    "arshdeep singh",
    "kagiso rabada",
    "rahul chahar",
    "harpreet brar",
    "sam curran",
    "ravi bishnoi",
    "naveen-ul-haq",
    "mohsin khan",
    "yash thakur",
    "rashid khan",
    "noor ahmad",
    "mohit sharma",
    "umesh yadav",
    "azmatullah omarzai",
    "ravindra jadeja",
    "hardik pandya",
    "sunil narine",
    "andre russell",
    "krunal pandya",
}

CRICKET_BATTER_NAMES = {
    "rohit sharma",
    "ishan kishan",
    "suryakumar yadav",
    "tilak varma",
    "tim david",
    "ruturaj gaikwad",
    "devon conway",
    "ajinkya rahane",
    "shivam dube",
    "ms dhoni",
    "virat kohli",
    "faf du plessis",
    "glenn maxwell",
    "rajat patidar",
    "dinesh karthik",
    "shreyas iyer",
    "rinku singh",
    "venkatesh iyer",
    "phil salt",
    "nitish rana",
    "sanju samson",
    "yashasvi jaiswal",
    "jos buttler",
    "riyan parag",
    "shimron hetmyer",
    "dhruv jurel",
    "travis head",
    "abhishek sharma",
    "aiden markram",
    "heinrich klaasen",
    "rahul tripathi",
    "rishabh pant",
    "david warner",
    "prithvi shaw",
    "jake fraser-mcgurk",
    "tristan stubbs",
    "mitchell marsh",
    "shikhar dhawan",
    "jonny bairstow",
    "liam livingstone",
    "jitesh sharma",
    "shashank singh",
    "prabhsimran singh",
    "kl rahul",
    "quinton de kock",
    "nicholas pooran",
    "marcus stoinis",
    "deepak hooda",
    "ayush badoni",
    "shubman gill",
    "wriddhiman saha",
    "sai sudharsan",
    "david miller",
    "rahul tewatia",
    "kane williamson",
}

FOOTBALL_FALLBACK_PLAYERS: dict[str, list[str]] = {
    "arsenal": ["Bukayo Saka", "Martin Odegaard", "Kai Havertz", "Declan Rice"],
    "manchester city": ["Erling Haaland", "Phil Foden", "Kevin De Bruyne", "Rodri"],
    "liverpool": ["Mohamed Salah", "Darwin Nunez", "Virgil van Dijk", "Alexis Mac Allister"],
    "tottenham hotspur": ["Son Heung-min", "James Maddison", "Cristian Romero", "Dejan Kulusevski"],
    "chelsea": ["Cole Palmer", "Enzo Fernandez", "Reece James", "Nicolas Jackson"],
    "manchester united": ["Bruno Fernandes", "Marcus Rashford", "Rasmus Hojlund", "Lisandro Martinez"],
    "real madrid": ["Vinicius Junior", "Jude Bellingham", "Rodrygo", "Federico Valverde"],
    "barcelona": ["Robert Lewandowski", "Lamine Yamal", "Pedri", "Frenkie de Jong"],
    "atletico madrid": ["Antoine Griezmann", "Alvaro Morata", "Rodrigo De Paul", "Jan Oblak"],
    "bayern munich": ["Harry Kane", "Jamal Musiala", "Leroy Sane", "Joshua Kimmich"],
}


@dataclass
class EnsembleMember:
    member_id: int
    model: TimeMCLModel
    feature_columns: list[str]
    scaler_mean: np.ndarray
    scaler_std: np.ndarray
    num_heads: int


class LiveMatchPredictor:
    @staticmethod
    def _default_feature_value(feature: str) -> float:
        defaults = {
            "team_a_win_rate_vs_b": 0.5,
            "venue_chase_success_rate": 0.5,
            "venue_defend_bias": 0.0,
            "batting_first": 1.0,
            "team_a_bats_first": 1.0,
            "team_b_bats_first": 0.0,
            "team_a_chase_success_rate": 0.5,
            "team_b_chase_success_rate": 0.5,
            "team_a_defend_success_rate": 0.5,
            "team_b_defend_success_rate": 0.5,
            "chase_defend_edge_team_a_first": 0.0,
            "chase_defend_edge_team_b_first": 0.0,
            "venue_batting_first_advantage": 0.0,
            "recent_form_diff": 0.0,
            "recent_run_rate_diff": 0.0,
            "head_to_head_win_diff": 0.0,
            "wickets_taken_diff": 0.0,
        }
        return float(defaults.get(feature, 0.0))

    def __init__(
        self,
        checkpoint_path: Path | None = None,
        manifest_path: Path | None = None,
        num_heads: int | None = None,
    ) -> None:
        self.config = MLConfig()
        self.checkpoint_path = checkpoint_path or (self.config.artifacts_dir / "time_mcl.pt")
        self.manifest_path = manifest_path or (self.config.processed_data_dir / "feature_manifest.json")
        self.num_heads = num_heads or self.config.num_heads

        self.feature_columns = FEATURE_COLUMNS.copy()
        self.feature_medians: dict[str, float] = {
            feature: self._default_feature_value(feature) for feature in self.feature_columns
        }
        self.feature_means: dict[str, float] = {
            feature: self._default_feature_value(feature) for feature in self.feature_columns
        }
        self.baseline_score = 170.0

        self.model: TimeMCLModel | None = None
        self.scaler_mean: np.ndarray | None = None
        self.scaler_std: np.ndarray | None = None
        self.ensemble_members: list[EnsembleMember] = []
        self.ensemble_manifest: dict[str, Any] | None = None
        self.encoder_type = str(self.config.encoder_type)
        self.patch_encoder_config: dict[str, int] = {
            "patch_length": int(self.config.patch_length),
            "patch_stride": int(self.config.patch_stride),
            "patch_model_dim": int(self.config.patch_model_dim),
            "patch_layers": int(self.config.patch_layers),
            "patch_attention_heads": int(self.config.patch_attention_heads),
        }
        self.conformal_calibration: dict[str, Any] | None = None
        self.model_loaded = False
        self.runtime_mode = "FALLBACK"

        self.lookup_df = pd.DataFrame()
        self.match_results = pd.DataFrame()
        self.team_profiles = pd.DataFrame()
        self.venue_profiles = pd.DataFrame()
        self.player_form = pd.DataFrame()

        self.football_team_profiles = pd.DataFrame()
        self.football_player_form = pd.DataFrame()
        self.football_matches = pd.DataFrame()

        self._load_assets()

    def _load_assets(self) -> None:
        processed = self.config.processed_data_dir
        needed = [
            processed / "match_feature_lookup.csv",
            processed / "team_profiles.csv",
            processed / "venue_profiles.csv",
            processed / "player_form_latest.csv",
            self.manifest_path,
        ]
        if not all(path.exists() for path in needed):
            try:
                run_feature_pipeline(self.config, bootstrap_if_missing=False)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Feature pipeline refresh unavailable; using existing artifacts only: %s", exc)

        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            self.feature_columns = manifest.get("feature_columns", FEATURE_COLUMNS)
            manifest_medians = manifest.get("feature_medians", {})
            manifest_means = manifest.get("feature_means", {})
            self.feature_medians = {
                feature: float(manifest_medians.get(feature, self._default_feature_value(feature)))
                for feature in self.feature_columns
            }
            self.feature_means = {
                feature: float(manifest_means.get(feature, self._default_feature_value(feature)))
                for feature in self.feature_columns
            }
            self.baseline_score = float(manifest.get("baseline_first_innings_mean", 170.0))

        lookup_path = processed / "match_feature_lookup.csv"
        if lookup_path.exists():
            self.lookup_df = pd.read_csv(lookup_path)

        matches_path = processed / "matches.csv"
        if matches_path.exists():
            self.match_results = pd.read_csv(matches_path)

        team_path = processed / "team_profiles.csv"
        if team_path.exists():
            self.team_profiles = pd.read_csv(team_path)

        venue_path = processed / "venue_profiles.csv"
        if venue_path.exists():
            self.venue_profiles = pd.read_csv(venue_path)

        player_path = processed / "player_form_latest.csv"
        if player_path.exists():
            self.player_form = pd.read_csv(player_path)

        football_team_path = processed / "football_team_profiles.csv"
        if football_team_path.exists():
            self.football_team_profiles = pd.read_csv(football_team_path)

        football_player_path = processed / "football_player_form_latest.csv"
        if football_player_path.exists():
            self.football_player_form = pd.read_csv(football_player_path)

        football_matches_path = processed / "football_matches.csv"
        if football_matches_path.exists():
            self.football_matches = pd.read_csv(football_matches_path)

        if not self._load_ensemble_checkpoint():
            self._load_model_checkpoint()

    def _load_ensemble_checkpoint(self) -> bool:
        if not TORCH_AVAILABLE:
            return False

        ensemble_dir = self.config.ensemble_dir
        manifest_path = ensemble_dir / "ensemble_manifest.json"
        if not manifest_path.exists():
            return False

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                return False
            members = manifest.get("members")
            if not isinstance(members, list) or not members:
                return False
        except Exception as exc:  # noqa: BLE001
            logger.warning("Invalid ensemble manifest, skipping ensemble load: %s", exc)
            return False

        loaded_members: list[EnsembleMember] = []
        conformal_candidate: dict[str, Any] | None = None
        for row in members:
            if not isinstance(row, dict):
                continue
            raw_path = row.get("checkpoint_path")
            if not isinstance(raw_path, str) or not raw_path.strip():
                continue
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                candidate = (self.config.repo_root / raw_path).resolve()
            if not candidate.exists():
                continue
            try:
                checkpoint = torch.load(candidate, map_location="cpu")
                feature_columns = checkpoint.get("feature_columns", self.feature_columns)
                hidden_dims = tuple(checkpoint.get("hidden_dims", self.config.hidden_dims))
                dropout = float(checkpoint.get("dropout", self.config.dropout))
                num_heads = int(checkpoint.get("num_heads", self.num_heads))
                encoder_type = str(checkpoint.get("encoder_type", self.config.encoder_type))
                patch_config = checkpoint.get("patch_encoder") or {}
                model = TimeMCLModel(
                    input_dim=len(feature_columns),
                    num_heads=num_heads,
                    hidden_dims=hidden_dims,
                    dropout=dropout,
                    encoder_type=encoder_type,
                    patch_length=int(patch_config.get("patch_length", self.config.patch_length)),
                    patch_stride=int(patch_config.get("patch_stride", self.config.patch_stride)),
                    patch_model_dim=int(patch_config.get("patch_model_dim", self.config.patch_model_dim)),
                    patch_layers=int(patch_config.get("patch_layers", self.config.patch_layers)),
                    patch_attention_heads=int(
                        patch_config.get("patch_attention_heads", self.config.patch_attention_heads)
                    ),
                )
                model.load_state_dict(checkpoint["model_state"])
                model.eval()
                scaler = checkpoint.get("scaler", {})
                scaler_mean = np.array(
                    scaler.get("mean", [0.0] * len(feature_columns)),
                    dtype=np.float32,
                )
                scaler_std = np.array(
                    scaler.get("std", [1.0] * len(feature_columns)),
                    dtype=np.float32,
                )
                scaler_std = np.where(scaler_std < 1e-6, 1.0, scaler_std)
                member = EnsembleMember(
                    member_id=int(row.get("member_id", len(loaded_members))),
                    model=model,
                    feature_columns=list(feature_columns),
                    scaler_mean=scaler_mean,
                    scaler_std=scaler_std,
                    num_heads=num_heads,
                )
                loaded_members.append(member)
                if conformal_candidate is None:
                    conformal = checkpoint.get("conformal_calibration")
                    if isinstance(conformal, dict):
                        conformal_candidate = conformal
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping broken ensemble member %s: %s", candidate, exc)

        if not loaded_members:
            logger.warning("No valid ensemble members loaded. Falling back to single checkpoint.")
            return False

        loaded_members.sort(key=lambda row: row.member_id)
        first_member = loaded_members[0]
        self.ensemble_members = loaded_members
        self.ensemble_manifest = manifest
        self.feature_columns = first_member.feature_columns
        self.scaler_mean = first_member.scaler_mean
        self.scaler_std = first_member.scaler_std
        self.model = first_member.model
        self.num_heads = int(sum(member.num_heads for member in loaded_members))
        self.conformal_calibration = conformal_candidate
        self.model_loaded = True
        self.runtime_mode = "TRAINED MODEL"
        self.encoder_type = str(manifest.get("encoder_type") or self.encoder_type)
        logger.info("Running in TRAINED MODEL mode (ensemble members=%s)", len(loaded_members))
        return True

    def _load_model_checkpoint(self) -> None:
        self.ensemble_members = []
        self.ensemble_manifest = None
        if not TORCH_AVAILABLE:
            self.model_loaded = False
            self.runtime_mode = "FALLBACK"
            logger.info("Running in FALLBACK mode (no trained model available). Torch is not installed.")
            return
        if not self.checkpoint_path.exists():
            self.model_loaded = False
            self.runtime_mode = "FALLBACK"
            logger.info(
                "Running in FALLBACK mode (no trained model available). Missing checkpoint: %s",
                self.checkpoint_path,
            )
            return
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            checkpoint_feature_columns = checkpoint.get("feature_columns", self.feature_columns)
            self.feature_columns = checkpoint_feature_columns

            hidden_dims = tuple(checkpoint.get("hidden_dims", self.config.hidden_dims))
            dropout = float(checkpoint.get("dropout", self.config.dropout))
            checkpoint_heads = int(checkpoint.get("num_heads", self.num_heads))
            encoder_type = str(checkpoint.get("encoder_type", self.config.encoder_type))
            patch_config = checkpoint.get("patch_encoder") or {}
            patch_length = int(patch_config.get("patch_length", self.config.patch_length))
            patch_stride = int(patch_config.get("patch_stride", self.config.patch_stride))
            patch_model_dim = int(patch_config.get("patch_model_dim", self.config.patch_model_dim))
            patch_layers = int(patch_config.get("patch_layers", self.config.patch_layers))
            patch_attention_heads = int(
                patch_config.get("patch_attention_heads", self.config.patch_attention_heads)
            )

            model = TimeMCLModel(
                input_dim=len(self.feature_columns),
                num_heads=checkpoint_heads,
                hidden_dims=hidden_dims,
                dropout=dropout,
                encoder_type=encoder_type,
                patch_length=patch_length,
                patch_stride=patch_stride,
                patch_model_dim=patch_model_dim,
                patch_layers=patch_layers,
                patch_attention_heads=patch_attention_heads,
            )
            model.load_state_dict(checkpoint["model_state"])
            model.eval()

            scaler = checkpoint.get("scaler", {})
            self.scaler_mean = np.array(scaler.get("mean", [0.0] * len(self.feature_columns)), dtype=np.float32)
            self.scaler_std = np.array(scaler.get("std", [1.0] * len(self.feature_columns)), dtype=np.float32)
            self.scaler_std = np.where(self.scaler_std < 1e-6, 1.0, self.scaler_std)

            self.num_heads = checkpoint_heads
            self.encoder_type = encoder_type
            self.patch_encoder_config = {
                "patch_length": patch_length,
                "patch_stride": patch_stride,
                "patch_model_dim": patch_model_dim,
                "patch_layers": patch_layers,
                "patch_attention_heads": patch_attention_heads,
            }
            conformal = checkpoint.get("conformal_calibration")
            self.conformal_calibration = conformal if isinstance(conformal, dict) else None
            self.model = model
            self.model_loaded = True
            self.runtime_mode = "TRAINED MODEL"
            logger.info("Running in TRAINED MODEL mode (encoder=%s)", self.encoder_type)
            if self.conformal_calibration and self.conformal_calibration.get("enabled"):
                logger.info(
                    "Conformal calibration enabled (alpha=%s, q_hat=%s)",
                    self.conformal_calibration.get("alpha"),
                    self.conformal_calibration.get("q_hat"),
                )
        except Exception as exc:  # noqa: BLE001
            self.model = None
            self.model_loaded = False
            self.runtime_mode = "FALLBACK"
            self.conformal_calibration = None
            logger.warning(
                "Running in FALLBACK mode (checkpoint load failed: %s).",
                exc,
            )

    @staticmethod
    def _softmax_scores(values: np.ndarray, temperature: float = 8.0) -> list[float]:
        arr = values.astype(float)
        centered = arr - np.mean(arr)
        logits = np.exp(-np.abs(centered) / max(0.5, temperature))
        probs = logits / np.sum(logits)
        return [float(x) for x in probs]

    @staticmethod
    def _enforce_prediction_diversity(predictions: np.ndarray, min_gap: float = 4.0) -> np.ndarray:
        preds = np.sort(predictions.astype(float))
        if len(preds) <= 1:
            return preds.astype(np.float32)

        original_center = float(np.mean(preds))
        template = original_center + (
            np.arange(len(preds), dtype=float) - ((len(preds) - 1) / 2.0)
        ) * float(min_gap)
        blended = np.sort((0.68 * preds) + (0.32 * template))
        for idx in range(1, len(blended)):
            if blended[idx] - blended[idx - 1] < min_gap:
                blended[idx] = blended[idx - 1] + min_gap
        recentered = blended + (original_center - float(np.mean(blended)))
        return recentered.astype(np.float32)

    @staticmethod
    def _normalize_reason_factors(reasons: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for raw in reasons:
            feature = str(raw.get("feature", "unknown_feature"))
            value = raw.get("value")
            baseline = raw.get("baseline")
            delta = raw.get("delta")

            if delta is None and isinstance(value, (int, float)) and isinstance(baseline, (int, float)):
                delta = float(value) - float(baseline)
            if isinstance(delta, (int, float)):
                delta = round(float(delta), 2)

            factor = {
                "feature": feature,
                "value": value,
                "baseline": baseline,
                "delta": delta,
                "unit": raw.get("unit"),
                "impact": str(raw.get("impact", "neutral")),
                "explanation": str(
                    raw.get(
                        "explanation",
                        "feature signal.",
                    )
                ),
            }
            normalized.append(factor)
        return normalized

    def _normalize_player_block(self, payload: dict[str, Any] | None) -> dict[str, Any] | None:
        if payload is None:
            return None
        normalized = dict(payload)
        reasons = payload.get("reason", [])
        normalized["reason"] = self._normalize_reason_factors(reasons if isinstance(reasons, list) else [])
        confidence = normalized.get("confidence")
        if isinstance(confidence, (int, float)):
            normalized["confidence"] = round(float(confidence), 2)
        return normalized

    def _calibrated_interval_for_heads(self, predictions: np.ndarray) -> tuple[float, float, dict[str, Any]]:
        heads = np.asarray(predictions, dtype=np.float32).reshape(1, -1)
        calibration = self.conformal_calibration
        if isinstance(calibration, dict) and "team_a" in calibration and "team_b" in calibration:
            q_a = float((calibration.get("team_a") or {}).get("q_hat", 0.0))
            q_b = float((calibration.get("team_b") or {}).get("q_hat", 0.0))
            shared = {
                "enabled": bool(calibration.get("enabled", False)),
                "method": "split_conformal_interval",
                "alpha": float(calibration.get("alpha", 0.0)),
                "q_hat": float((q_a + q_b) / 2.0),
            }
            lower, upper = apply_conformal_interval(heads, shared)
            calibration_meta = {
                "enabled": bool(shared["enabled"]),
                "method": shared["method"],
                "alpha": float(shared["alpha"]),
                "q_hat": float(shared["q_hat"]),
            }
            return float(lower[0]), float(upper[0]), calibration_meta

        lower, upper = apply_conformal_interval(heads, calibration)
        calibration_meta = {
            "enabled": bool(calibration and calibration.get("enabled", False)),
            "method": (calibration or {}).get("method", "none"),
            "alpha": float((calibration or {}).get("alpha", 0.0)),
            "q_hat": float((calibration or {}).get("q_hat", 0.0)),
        }
        return float(lower[0]), float(upper[0]), calibration_meta

    def _scenario_labels(self, k: int) -> list[str]:
        labels = ["Low", "Baseline", "High", "Aggressive"]
        if k <= len(labels):
            return labels[:k]
        return labels + (["Aggressive"] * (k - len(labels)))

    @staticmethod
    def _scenario_anchor_quantiles(k: int) -> np.ndarray:
        if k <= 1:
            return np.asarray([0.5], dtype=np.float32)
        if k == 2:
            return np.asarray([0.28, 0.78], dtype=np.float32)
        if k == 3:
            return np.asarray([0.16, 0.5, 0.86], dtype=np.float32)
        if k == 4:
            return np.asarray([0.1, 0.42, 0.74, 0.93], dtype=np.float32)
        return np.linspace(0.08, 0.93, num=k, dtype=np.float32)

    def _anchor_indices_from_distribution(self, values: np.ndarray, k: int) -> np.ndarray:
        if values.size == 0:
            return np.asarray([], dtype=np.int64)
        quantiles = self._scenario_anchor_quantiles(k)
        targets = np.quantile(values, quantiles).astype(np.float32)
        order = np.argsort(values)
        selected: list[int] = []
        used: set[int] = set()
        for target in targets:
            ranked = order[np.argsort(np.abs(values[order] - target))]
            pick = None
            for candidate in ranked:
                if int(candidate) not in used:
                    pick = int(candidate)
                    break
            if pick is None:
                pick = int(ranked[0]) if ranked.size else 0
            used.add(pick)
            selected.append(pick)
        return np.asarray(selected, dtype=np.int64)

    def _get_team_profile(self, team: str) -> dict[str, float]:
        if self.team_profiles.empty:
            return {
                "avg_runs_scored": self.baseline_score,
                "avg_wickets_lost": 7.0,
                "avg_runs_conceded": self.baseline_score,
                "avg_wickets_taken": 7.0,
                "win_rate": 0.5,
                "batting_strength_index": 32.0,
                "bowling_strength_index": 25.0,
            }

        subset = self.team_profiles[self.team_profiles["team"].str.lower() == team.lower()]
        if subset.empty:
            means = self.team_profiles.mean(numeric_only=True)
            return {
                "avg_runs_scored": float(means.get("avg_runs_scored", self.baseline_score)),
                "avg_wickets_lost": float(means.get("avg_wickets_lost", 7.0)),
                "avg_runs_conceded": float(means.get("avg_runs_conceded", self.baseline_score)),
                "avg_wickets_taken": float(means.get("avg_wickets_taken", 7.0)),
                "win_rate": float(means.get("win_rate", 0.5)),
                "batting_strength_index": float(means.get("batting_strength_index", 32.0)),
                "bowling_strength_index": float(means.get("bowling_strength_index", 25.0)),
            }

        row = subset.iloc[0]
        return {
            "avg_runs_scored": float(row["avg_runs_scored"]),
            "avg_wickets_lost": float(row["avg_wickets_lost"]),
            "avg_runs_conceded": float(row["avg_runs_conceded"]),
            "avg_wickets_taken": float(row["avg_wickets_taken"]),
            "win_rate": float(row["win_rate"]),
            "batting_strength_index": float(row["batting_strength_index"]),
            "bowling_strength_index": float(row["bowling_strength_index"]),
        }

    def _team_recent_form(self, team: str) -> dict[str, float]:
        profile = self._get_team_profile(team)
        defaults = {
            "avg_runs_last_5": float(profile["avg_runs_scored"]),
            "avg_runs_last_10": float(profile["avg_runs_scored"]),
            "avg_wickets_last_5": float(profile["avg_wickets_lost"]),
            "avg_runs_conceded_last_5": float(profile["avg_runs_conceded"]),
        }
        if self.match_results.empty:
            return defaults

        frame = self.match_results.copy()
        if "match_date" in frame.columns:
            frame["match_date"] = pd.to_datetime(frame["match_date"], errors="coerce")
            frame = frame.sort_values("match_date")
        subset = frame[
            (frame["team_a"].astype(str).str.lower() == team.lower())
            | (frame["team_b"].astype(str).str.lower() == team.lower())
        ]
        if subset.empty:
            return defaults

        scored: list[float] = []
        conceded: list[float] = []
        wickets_lost: list[float] = []
        recent = subset.tail(10)
        for _, row in recent.iterrows():
            first_team = str(row.get("first_innings_team", ""))
            if first_team.lower() == team.lower():
                score_val = pd.to_numeric(row.get("first_innings_total"), errors="coerce")
                conceded_val = pd.to_numeric(row.get("second_innings_total"), errors="coerce")
                wickets_val = pd.to_numeric(row.get("first_innings_wickets"), errors="coerce")
            else:
                score_val = pd.to_numeric(row.get("second_innings_total"), errors="coerce")
                conceded_val = pd.to_numeric(row.get("first_innings_total"), errors="coerce")
                wickets_val = pd.to_numeric(row.get("second_innings_wickets"), errors="coerce")
            if pd.notna(score_val):
                scored.append(float(score_val))
            if pd.notna(conceded_val):
                conceded.append(float(conceded_val))
            if pd.notna(wickets_val):
                wickets_lost.append(float(wickets_val))

        if not scored:
            return defaults
        scored_last5 = scored[-5:] if len(scored) >= 5 else scored
        conceded_last5 = conceded[-5:] if len(conceded) >= 5 else conceded
        wickets_last5 = wickets_lost[-5:] if len(wickets_lost) >= 5 else wickets_lost
        return {
            "avg_runs_last_5": float(np.mean(scored_last5)),
            "avg_runs_last_10": float(np.mean(scored)),
            "avg_wickets_last_5": float(np.mean(wickets_last5)) if wickets_last5 else defaults["avg_wickets_last_5"],
            "avg_runs_conceded_last_5": float(np.mean(conceded_last5))
            if conceded_last5
            else defaults["avg_runs_conceded_last_5"],
        }

    def _team_chase_defend_rates(self, team: str) -> tuple[float, float]:
        if self.match_results.empty:
            return 0.5, 0.5
        frame = self.match_results.copy()
        subset = frame[
            (frame["team_a"].astype(str).str.lower() == team.lower())
            | (frame["team_b"].astype(str).str.lower() == team.lower())
        ]
        if subset.empty:
            return 0.5, 0.5
        chase_attempts = 0
        chase_wins = 0
        defend_attempts = 0
        defend_wins = 0
        for _, row in subset.iterrows():
            first_team = str(row.get("first_innings_team", "")).strip().lower()
            second_team = str(row.get("second_innings_team", "")).strip().lower()
            winner = str(row.get("winner", "")).strip().lower()
            if second_team == team.lower():
                chase_attempts += 1
                if winner == team.lower():
                    chase_wins += 1
            elif first_team == team.lower():
                defend_attempts += 1
                if winner == team.lower():
                    defend_wins += 1
        chase_rate = float(chase_wins / chase_attempts) if chase_attempts else 0.5
        defend_rate = float(defend_wins / defend_attempts) if defend_attempts else 0.5
        return float(np.clip(chase_rate, 0.0, 1.0)), float(np.clip(defend_rate, 0.0, 1.0))

    def _head_to_head_stats(self, team_a: str, team_b: str) -> tuple[float, float, float]:
        if self.match_results.empty:
            return 0.5, self.baseline_score, self.baseline_score
        frame = self.match_results.copy()
        if "match_date" in frame.columns:
            frame["match_date"] = pd.to_datetime(frame["match_date"], errors="coerce")
            frame = frame.sort_values("match_date")
        subset = frame[
            (
                (frame["team_a"].astype(str).str.lower() == team_a.lower())
                & (frame["team_b"].astype(str).str.lower() == team_b.lower())
            )
            | (
                (frame["team_a"].astype(str).str.lower() == team_b.lower())
                & (frame["team_b"].astype(str).str.lower() == team_a.lower())
            )
        ]
        if subset.empty:
            return 0.5, self.baseline_score, self.baseline_score

        wins_a = 0
        scores_a: list[float] = []
        scores_b: list[float] = []
        for _, row in subset.iterrows():
            winner = str(row.get("winner", ""))
            if winner.lower() == team_a.lower():
                wins_a += 1
            first_team = str(row.get("first_innings_team", ""))
            if first_team.lower() == team_a.lower():
                team_a_score = pd.to_numeric(row.get("first_innings_total"), errors="coerce")
                team_b_score = pd.to_numeric(row.get("second_innings_total"), errors="coerce")
            else:
                team_a_score = pd.to_numeric(row.get("second_innings_total"), errors="coerce")
                team_b_score = pd.to_numeric(row.get("first_innings_total"), errors="coerce")
            if pd.notna(team_a_score):
                scores_a.append(float(team_a_score))
            if pd.notna(team_b_score):
                scores_b.append(float(team_b_score))
        return (
            float(wins_a / len(subset)),
            float(np.mean(scores_a)) if scores_a else self.baseline_score,
            float(np.mean(scores_b)) if scores_b else self.baseline_score,
        )

    def _venue_context(self, venue: str) -> tuple[float, float, float]:
        if self.match_results.empty:
            return self.baseline_score, 0.5, 0.0
        frame = self.match_results.copy()
        subset = frame[frame["venue"].astype(str).str.lower() == venue.lower()]
        if subset.empty:
            return self.baseline_score, 0.5, 0.0
        first = pd.to_numeric(subset["first_innings_total"], errors="coerce")
        second = pd.to_numeric(subset["second_innings_total"], errors="coerce")
        innings = pd.concat([first, second], axis=0).dropna()
        venue_avg_score = float(innings.mean()) if not innings.empty else self.baseline_score
        chase_rate = float(np.mean(second > first))
        batting_first_advantage = float(np.mean(first > second) - 0.5) * 2.0
        return venue_avg_score, float(np.clip(chase_rate, 0.0, 1.0)), float(
            np.clip(batting_first_advantage, -1.0, 1.0)
        )

    @staticmethod
    def _normalize_live_feature_aliases(live_features: dict[str, Any] | None) -> dict[str, float] | None:
        if not isinstance(live_features, dict):
            return None
        alias_map = {
            "batting_team_avg_runs_last5": "team_a_avg_runs_last_5",
            "batting_team_avg_wickets_last5": "team_a_avg_wickets_last_5",
            "bowling_team_avg_runs_conceded_last5": "team_b_avg_runs_last_5",
            "bowling_team_avg_wickets_taken_last5": "team_b_avg_wickets_last_5",
            "head_to_head_avg_first_innings": "avg_score_team_a_vs_b",
            "venue_avg_first_innings": "venue_avg_score",
            "toss_bat_first": "batting_first",
            "venue_defend_bias": "venue_defend_bias",
            "team_a_chase_success_rate": "team_a_chase_success_rate",
            "team_b_chase_success_rate": "team_b_chase_success_rate",
            "team_a_defend_success_rate": "team_a_defend_success_rate",
            "team_b_defend_success_rate": "team_b_defend_success_rate",
        }
        normalized: dict[str, float] = {}
        for key, value in live_features.items():
            if not isinstance(value, (int, float)):
                continue
            normalized_key = alias_map.get(str(key), str(key))
            normalized[normalized_key] = float(value)
            if normalized_key == "avg_score_team_a_vs_b":
                normalized["avg_score_team_b_vs_a"] = float(value)
        return normalized

    def _feature_dict_from_match(self, match_payload: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
        match_id = match_payload.get("match_id")
        base_feature_dict: dict[str, float] | None = None
        if match_id and not self.lookup_df.empty:
            subset = self.lookup_df[self.lookup_df["match_id"] == match_id]
            if not subset.empty:
                row = subset.iloc[0]
                feature_map: dict[str, float] = {}
                for feature in self.feature_columns:
                    raw_value = row.get(feature, self.feature_medians.get(feature, 0.0))
                    try:
                        value = float(raw_value)
                    except (TypeError, ValueError):
                        value = float(self.feature_medians.get(feature, 0.0))
                    if not np.isfinite(value):
                        value = float(self.feature_medians.get(feature, 0.0))
                    feature_map[feature] = value
                base_feature_dict = feature_map

        if base_feature_dict is None:
            team_a = str(match_payload["team_a"])
            team_b = str(match_payload["team_b"])
            venue = str(match_payload["venue"])
            team_a_form = self._team_recent_form(team_a)
            team_b_form = self._team_recent_form(team_b)
            team_a_win_rate_vs_b, avg_score_a_vs_b, avg_score_b_vs_a = self._head_to_head_stats(team_a, team_b)
            team_a_chase_rate, team_a_defend_rate = self._team_chase_defend_rates(team_a)
            team_b_chase_rate, team_b_defend_rate = self._team_chase_defend_rates(team_b)
            venue_avg_score, venue_chase_success_rate, venue_batting_first_advantage = self._venue_context(venue)
            run_rate_a = team_a_form["avg_runs_last_5"] - team_a_form["avg_runs_last_10"]
            run_rate_b = team_b_form["avg_runs_last_5"] - team_b_form["avg_runs_last_10"]

            base_feature_dict = {
                "team_a_avg_runs_last_5": team_a_form["avg_runs_last_5"],
                "team_a_avg_runs_last_10": team_a_form["avg_runs_last_10"],
                "team_a_avg_wickets_last_5": team_a_form["avg_wickets_last_5"],
                "team_a_run_rate_trend": run_rate_a,
                "team_b_avg_runs_last_5": team_b_form["avg_runs_last_5"],
                "team_b_avg_runs_last_10": team_b_form["avg_runs_last_10"],
                "team_b_avg_wickets_last_5": team_b_form["avg_wickets_last_5"],
                "team_b_run_rate_trend": run_rate_b,
                "team_a_win_rate_vs_b": team_a_win_rate_vs_b,
                "avg_score_team_a_vs_b": avg_score_a_vs_b,
                "avg_score_team_b_vs_a": avg_score_b_vs_a,
                "venue_avg_score": venue_avg_score,
                "venue_chase_success_rate": venue_chase_success_rate,
                "venue_defend_bias": -venue_batting_first_advantage,
                "team_a_runs_vs_opponent_avg": team_a_form["avg_runs_last_5"] - team_b_form["avg_runs_conceded_last_5"],
                "team_b_runs_vs_opponent_avg": team_b_form["avg_runs_last_5"] - team_a_form["avg_runs_conceded_last_5"],
                "batting_first": 1.0,
                "team_a_bats_first": 1.0,
                "team_b_bats_first": 0.0,
                "team_a_chase_success_rate": team_a_chase_rate,
                "team_b_chase_success_rate": team_b_chase_rate,
                "team_a_defend_success_rate": team_a_defend_rate,
                "team_b_defend_success_rate": team_b_defend_rate,
                "chase_defend_edge_team_a_first": team_b_chase_rate - team_a_defend_rate,
                "chase_defend_edge_team_b_first": team_a_chase_rate - team_b_defend_rate,
                "venue_batting_first_advantage": venue_batting_first_advantage,
                "recent_form_diff": team_a_form["avg_runs_last_5"] - team_b_form["avg_runs_last_5"],
                "recent_run_rate_diff": run_rate_a - run_rate_b,
                "head_to_head_win_diff": (2.0 * team_a_win_rate_vs_b) - 1.0,
                "wickets_taken_diff": team_a_form["avg_wickets_last_5"] - team_b_form["avg_wickets_last_5"],
            }
            for name in self.feature_columns:
                base_feature_dict.setdefault(name, float(self.feature_medians.get(name, self._default_feature_value(name))))

        live_features = self._normalize_live_feature_aliases(match_payload.get("live_context"))
        recency_weight = float(match_payload.get("live_recency_weight", 0.65))
        blended, blend_meta = blend_cricket_features(
            historical_features=base_feature_dict,
            live_features=live_features,
            recency_weight=recency_weight,
        )
        residual_context = match_payload.get("residual_context")
        if isinstance(residual_context, dict) and residual_context:
            bias = float(residual_context.get("combined_bias", 0.0))
            trend_delta = float(
                residual_context.get("recent_underprediction_rate", 0.0)
                - residual_context.get("recent_overprediction_rate", 0.0)
            )
            if abs(bias) > 0.01:
                for feature_name in [
                    "team_a_avg_runs_last_5",
                    "venue_avg_score",
                    "avg_score_team_a_vs_b",
                ]:
                    blended[feature_name] = float(blended.get(feature_name, 0.0) + bias)
            if abs(trend_delta) > 1e-6:
                blended["team_a_run_rate_trend"] = float(
                    blended.get("team_a_run_rate_trend", 0.0) + trend_delta
                )
                blended["team_b_run_rate_trend"] = float(
                    blended.get("team_b_run_rate_trend", 0.0) - trend_delta
                )
            blend_meta["residual_context"] = {
                "combined_bias": round(float(bias), 3),
                "trend_delta": round(float(trend_delta), 3),
                "samples": int(residual_context.get("samples", 0)),
                "residual_shift_score": round(float(residual_context.get("residual_shift_score", 0.0)), 3),
            }
        for feature in self.feature_columns:
            if feature not in blended:
                blended[feature] = float(self.feature_medians.get(feature, self._default_feature_value(feature)))
                continue
            try:
                value = float(blended[feature])
            except (TypeError, ValueError):
                value = float(self.feature_medians.get(feature, self._default_feature_value(feature)))
            if not np.isfinite(value):
                value = float(self.feature_medians.get(feature, self._default_feature_value(feature)))
            blended[feature] = value
        return blended, blend_meta

    def _compute_anomaly_signal(
        self,
        feature_dict: dict[str, float],
        residual_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        med = self.feature_medians
        checks = [
            ("team_a_avg_runs_last_5", 18.0),
            ("venue_avg_score", 16.0),
            ("team_a_run_rate_trend", 8.0),
            ("team_a_win_rate_vs_b", 0.35),
        ]
        z_like: list[float] = []
        for name, scale in checks:
            value = float(feature_dict.get(name, 0.0))
            base = float(med.get(name, value))
            z_like.append(abs(value - base) / max(1.0, scale))
        base_score = float(np.clip(np.mean(z_like), 0.0, 2.0))

        residual_shift = float((residual_context or {}).get("residual_shift_score", 0.0))
        residual_component = float(np.clip(residual_shift / 12.0, 0.0, 1.5))
        anomaly_score = float(np.clip((0.7 * base_score) + (0.3 * residual_component), 0.0, 1.0))
        odd_variant_flag = anomaly_score >= 0.62

        return {
            "anomaly_score": round(anomaly_score, 3),
            "odd_variant_flag": odd_variant_flag,
            "residual_shift_score": round(residual_shift, 3),
            "signal_breakdown": {
                "feature_deviation_score": round(base_score, 3),
                "residual_shift_component": round(residual_component, 3),
            },
        }

    def _fallback_outcome_samples(
        self,
        feature_vector: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.lookup_df.empty:
            empty = np.asarray([], dtype=np.float32)
            return empty, empty, empty
        required_targets = ["target_team_a_score", "target_team_b_score", "target_winner_team_a"]
        if not set(required_targets).issubset(self.lookup_df.columns):
            empty = np.asarray([], dtype=np.float32)
            return empty, empty, empty

        usable_features = [name for name in self.feature_columns if name in self.lookup_df.columns]
        if not usable_features:
            empty = np.asarray([], dtype=np.float32)
            return empty, empty, empty

        frame = self.lookup_df[usable_features + required_targets].dropna(subset=required_targets).copy()
        if frame.empty:
            empty = np.asarray([], dtype=np.float32)
            return empty, empty, empty

        matrix = frame[usable_features].to_numpy(dtype=np.float32)
        query = np.asarray([feature_vector[self.feature_columns.index(name)] for name in usable_features], dtype=np.float32)
        scales = matrix.std(axis=0)
        scales = np.where(scales < 1e-6, 1.0, scales)
        distances = np.linalg.norm((matrix - query) / scales, axis=1)
        nearest_n = min(len(frame), max(24, self.num_heads * 8))
        nearest_idx = np.argsort(distances)[:nearest_n]
        nearest = frame.iloc[nearest_idx]
        if len(nearest) < max(8, self.num_heads * 2):
            nearest = frame
        return (
            nearest["target_team_a_score"].to_numpy(dtype=np.float32),
            nearest["target_team_b_score"].to_numpy(dtype=np.float32),
            nearest["target_winner_team_a"].to_numpy(dtype=np.float32),
        )

    def _member_feature_vector(self, feature_vector: np.ndarray, member: EnsembleMember) -> np.ndarray:
        if len(member.feature_columns) == len(self.feature_columns) and member.feature_columns == self.feature_columns:
            return feature_vector
        base_map = {
            name: float(feature_vector[idx])
            for idx, name in enumerate(self.feature_columns)
            if idx < len(feature_vector)
        }
        return np.asarray(
            [
                float(base_map.get(name, self.feature_medians.get(name, 0.0)))
                for name in member.feature_columns
            ],
            dtype=np.float32,
        )

    def _branch_feature_vector(
        self,
        feature_dict: dict[str, float],
        *,
        team_a_bats_first: bool,
    ) -> np.ndarray:
        row = dict(feature_dict)
        batting_first = 1.0 if team_a_bats_first else 0.0
        row["batting_first"] = batting_first
        row["team_a_bats_first"] = batting_first
        row["team_b_bats_first"] = 1.0 - batting_first

        team_a_chase = float(row.get("team_a_chase_success_rate", 0.5))
        team_b_chase = float(row.get("team_b_chase_success_rate", 0.5))
        team_a_defend = float(row.get("team_a_defend_success_rate", 0.5))
        team_b_defend = float(row.get("team_b_defend_success_rate", 0.5))
        row["chase_defend_edge_team_a_first"] = float(team_b_chase - team_a_defend)
        row["chase_defend_edge_team_b_first"] = float(team_a_chase - team_b_defend)

        return np.asarray(
            [
                float(row.get(name, self.feature_medians.get(name, self._default_feature_value(name))))
                for name in self.feature_columns
            ],
            dtype=np.float32,
        )

    def _predict_conditional_raw_outcomes(
        self,
        feature_dict: dict[str, float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
        team_a_first_vector = self._branch_feature_vector(feature_dict, team_a_bats_first=True)
        team_b_first_vector = self._branch_feature_vector(feature_dict, team_a_bats_first=False)

        a_scores, a_probs, source_a = self._predict_raw_outcomes(team_a_first_vector)
        b_scores, b_probs, source_b = self._predict_raw_outcomes(team_b_first_vector)

        size = min(a_scores.shape[0], b_scores.shape[0], a_probs.shape[0], b_probs.shape[0])
        if size <= 0:
            fallback_scores = np.asarray([[self.baseline_score, self.baseline_score]], dtype=np.float32)
            fallback_probs = np.asarray([0.5], dtype=np.float32)
            return fallback_scores, fallback_probs, fallback_scores, fallback_probs, "fallback"

        if a_scores.shape[0] != size:
            a_scores = a_scores[:size]
        if b_scores.shape[0] != size:
            b_scores = b_scores[:size]
        if a_probs.shape[0] != size:
            a_probs = a_probs[:size]
        if b_probs.shape[0] != size:
            b_probs = b_probs[:size]

        source = source_a if source_a == source_b else f"{source_a}+{source_b}"
        return a_scores, a_probs, b_scores, b_probs, source

    def _predict_raw_outcomes(
        self,
        feature_vector: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        if self.ensemble_members:
            score_rows: list[np.ndarray] = []
            prob_rows: list[np.ndarray] = []
            for member in self.ensemble_members:
                try:
                    member_vector = self._member_feature_vector(feature_vector, member)
                    normalized = (member_vector - member.scaler_mean) / member.scaler_std
                    with torch.no_grad():
                        score_heads, winner_logits = member.model(
                            torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)
                        )
                    score_rows.append(score_heads.squeeze(0).cpu().numpy().astype(np.float32))
                    prob_rows.append(
                        torch.sigmoid(winner_logits.squeeze(0)).cpu().numpy().astype(np.float32)
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Skipping ensemble member %s during inference: %s", member.member_id, exc)
            if score_rows and prob_rows:
                return (
                    np.concatenate(score_rows, axis=0),
                    np.concatenate(prob_rows, axis=0),
                    "ensemble",
                )

        if not self.model_loaded or self.model is None or self.scaler_mean is None or self.scaler_std is None:
            team_a_samples, team_b_samples, winner_samples = self._fallback_outcome_samples(feature_vector)
            if team_a_samples.size > 0 and team_b_samples.size > 0:
                return (
                    np.column_stack(
                        [
                            team_a_samples,
                            team_b_samples,
                        ]
                    ).astype(np.float32),
                    np.clip(winner_samples.astype(np.float32), 0.02, 0.98),
                    "fallback",
                )
            base_feature_a = float(feature_vector[0]) if feature_vector.shape[0] > 0 else self.baseline_score
            base_feature_b = float(feature_vector[4]) if feature_vector.shape[0] > 4 else self.baseline_score
            baseline_a = float(self.baseline_score + (0.15 * (base_feature_a - self.baseline_score)))
            baseline_b = float(self.baseline_score + (0.15 * (base_feature_b - self.baseline_score)))
            default_scores = np.array([[baseline_a, baseline_b]], dtype=np.float32)
            default_probs = np.array([0.5], dtype=np.float32)
            return default_scores, default_probs, "fallback"

        normalized = (feature_vector - self.scaler_mean) / self.scaler_std
        with torch.no_grad():
            score_heads, winner_logits = self.model(
                torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)
            )
        score_np = score_heads.squeeze(0).cpu().numpy().astype(np.float32)
        winner_prob_np = torch.sigmoid(winner_logits.squeeze(0)).cpu().numpy().astype(np.float32)
        return score_np, winner_prob_np, "single"

    def _distribution_scenarios(
        self,
        score_samples: np.ndarray,
        winner_probs: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if score_samples.ndim != 2 or score_samples.shape[1] != 2:
            raise ValueError("score_samples must have shape [n, 2]")
        if winner_probs.ndim != 1 or winner_probs.shape[0] != score_samples.shape[0]:
            raise ValueError("winner_probs must have shape [n]")

        scores = np.asarray(score_samples, dtype=np.float32)
        winners = np.asarray(winner_probs, dtype=np.float32)
        finite_mask = np.isfinite(scores).all(axis=1) & np.isfinite(winners)
        scores = scores[finite_mask]
        winners = winners[finite_mask]

        if scores.shape[0] == 0:
            scores = np.asarray([[self.baseline_score, self.baseline_score]], dtype=np.float32)
            winners = np.asarray([0.5], dtype=np.float32)

        totals = scores.mean(axis=1)
        if scores.shape[0] < k:
            quantiles = np.linspace(0.1, 0.9, num=max(2, k), dtype=np.float32)
            padded_scores = np.column_stack(
                [
                    np.quantile(scores[:, 0], quantiles),
                    np.quantile(scores[:, 1], quantiles),
                ]
            ).astype(np.float32)
            padded_winners = np.quantile(winners, quantiles).astype(np.float32)
            scores = padded_scores
            winners = np.clip(padded_winners, 0.02, 0.98)
            totals = scores.mean(axis=1)

        centroids = np.quantile(totals, np.linspace(0.1, 0.9, num=k)).astype(np.float32)
        assignments = np.zeros(shape=(totals.shape[0],), dtype=np.int64)
        for _ in range(8):
            distance_matrix = np.abs(totals[:, None] - centroids[None, :])
            assignments = np.argmin(distance_matrix, axis=1)
            updated = centroids.copy()
            for idx in range(k):
                bucket = totals[assignments == idx]
                if bucket.size:
                    updated[idx] = float(bucket.mean())
            if np.allclose(updated, centroids, atol=1e-3):
                centroids = updated
                break
            centroids = updated

        scenario_scores: list[np.ndarray] = []
        scenario_winners: list[float] = []
        scenario_probs: list[float] = []
        scenario_totals: list[float] = []

        for idx in range(k):
            mask = assignments == idx
            if not np.any(mask):
                nearest_idx = int(np.argmin(np.abs(totals - float(centroids[idx]))))
                mask = np.zeros_like(totals, dtype=bool)
                mask[nearest_idx] = True
            block_scores = scores[mask]
            block_winners = winners[mask]
            scenario_score = block_scores.mean(axis=0)
            scenario_scores.append(scenario_score)
            scenario_winners.append(float(np.clip(block_winners.mean(), 0.02, 0.98)))
            scenario_probs.append(float(np.mean(mask)))
            scenario_totals.append(float(np.mean(block_scores.mean(axis=1))))

        scenario_scores_arr = np.asarray(scenario_scores, dtype=np.float32)
        scenario_winners_arr = np.asarray(scenario_winners, dtype=np.float32)
        scenario_probs_arr = np.asarray(scenario_probs, dtype=np.float32)
        scenario_totals_arr = np.asarray(scenario_totals, dtype=np.float32)

        order = np.argsort(scenario_totals_arr)
        scenario_scores_arr = scenario_scores_arr[order]
        scenario_winners_arr = scenario_winners_arr[order]
        scenario_probs_arr = scenario_probs_arr[order]
        scenario_totals_arr = scenario_totals_arr[order]
        separation_ready = bool(np.std(totals) >= 4.0 or np.std(winners) >= 0.04)
        if separation_ready and scenario_totals_arr.size > 1:
            min_gap = max(2.0, float(np.std(totals) * 0.18))
            gaps = np.diff(scenario_totals_arr)
            linear_pattern = bool(
                gaps.size >= 2
                and float(np.std(gaps)) <= max(0.9, float(np.std(totals)) * 0.08)
            )
            if np.any(gaps < min_gap) or linear_pattern:
                anchor_indices = self._anchor_indices_from_distribution(totals, k)
                if anchor_indices.size != k:
                    sorted_idx = np.argsort(totals)
                    anchor_positions = np.linspace(0, len(sorted_idx) - 1, num=k)
                    anchor_indices = sorted_idx[np.round(anchor_positions).astype(int)]
                scenario_scores_arr = scores[anchor_indices].astype(np.float32)
                scenario_winners_arr = np.clip(winners[anchor_indices], 0.02, 0.98).astype(np.float32)
                scenario_totals_arr = scenario_scores_arr.mean(axis=1).astype(np.float32)
                order = np.argsort(scenario_totals_arr)
                scenario_scores_arr = scenario_scores_arr[order]
                scenario_winners_arr = scenario_winners_arr[order]
                scenario_totals_arr = scenario_totals_arr[order]
                distances = np.abs(totals[:, None] - scenario_totals_arr[None, :])
                nearest = np.argmin(distances, axis=1)
                counts = np.bincount(nearest, minlength=k).astype(np.float32)
                total_counts = float(np.sum(counts))
                if total_counts > 0:
                    scenario_probs_arr = counts / total_counts
                else:
                    scenario_probs_arr = np.full(shape=(k,), fill_value=1.0 / float(k), dtype=np.float32)
        prob_sum = float(np.sum(scenario_probs_arr))
        if prob_sum <= 0:
            scenario_probs_arr = np.full(shape=(k,), fill_value=1.0 / float(k), dtype=np.float32)
        else:
            scenario_probs_arr = scenario_probs_arr / prob_sum
        confidence_base = 0.6 if self.model_loaded else 0.48
        winner_certainty = np.abs(scenario_winners_arr - 0.5) * 2.0
        spread_penalty = float(np.clip(np.std(totals) / 22.0, 0.0, 0.22))
        confidence_arr = np.clip(
            confidence_base + (0.24 * scenario_probs_arr) + (0.16 * winner_certainty) - spread_penalty,
            0.28,
            0.92,
        )
        return (
            scenario_scores_arr,
            scenario_winners_arr,
            scenario_probs_arr,
            confidence_arr.astype(np.float32),
            scenario_totals_arr,
        )

    def _distribution_conditional_scenarios(
        self,
        team_a_first_scores: np.ndarray,
        team_a_first_winner_probs: np.ndarray,
        team_b_first_scores: np.ndarray,
        team_b_first_winner_probs: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if team_a_first_scores.ndim != 2 or team_a_first_scores.shape[1] != 2:
            raise ValueError("team_a_first_scores must have shape [n, 2]")
        if team_b_first_scores.ndim != 2 or team_b_first_scores.shape[1] != 2:
            raise ValueError("team_b_first_scores must have shape [n, 2]")
        if team_a_first_winner_probs.ndim != 1 or team_b_first_winner_probs.ndim != 1:
            raise ValueError("winner probs must have shape [n]")

        size = min(
            team_a_first_scores.shape[0],
            team_b_first_scores.shape[0],
            team_a_first_winner_probs.shape[0],
            team_b_first_winner_probs.shape[0],
        )
        if size <= 0:
            a_scores = np.asarray([[self.baseline_score, self.baseline_score]], dtype=np.float32)
            b_scores = np.asarray([[self.baseline_score, self.baseline_score]], dtype=np.float32)
            a_winners = np.asarray([0.5], dtype=np.float32)
            b_winners = np.asarray([0.5], dtype=np.float32)
        else:
            a_scores = np.asarray(team_a_first_scores[:size], dtype=np.float32)
            b_scores = np.asarray(team_b_first_scores[:size], dtype=np.float32)
            a_winners = np.asarray(team_a_first_winner_probs[:size], dtype=np.float32)
            b_winners = np.asarray(team_b_first_winner_probs[:size], dtype=np.float32)

        finite_mask = (
            np.isfinite(a_scores).all(axis=1)
            & np.isfinite(b_scores).all(axis=1)
            & np.isfinite(a_winners)
            & np.isfinite(b_winners)
        )
        a_scores = a_scores[finite_mask]
        b_scores = b_scores[finite_mask]
        a_winners = a_winners[finite_mask]
        b_winners = b_winners[finite_mask]
        if a_scores.shape[0] == 0:
            a_scores = np.asarray([[self.baseline_score, self.baseline_score]], dtype=np.float32)
            b_scores = np.asarray([[self.baseline_score, self.baseline_score]], dtype=np.float32)
            a_winners = np.asarray([0.5], dtype=np.float32)
            b_winners = np.asarray([0.5], dtype=np.float32)

        combined_totals = (a_scores.mean(axis=1) + b_scores.mean(axis=1)) / 2.0
        if a_scores.shape[0] < k:
            quantiles = np.linspace(0.1, 0.9, num=max(2, k), dtype=np.float32)
            a_scores = np.column_stack(
                [np.quantile(a_scores[:, 0], quantiles), np.quantile(a_scores[:, 1], quantiles)]
            ).astype(np.float32)
            b_scores = np.column_stack(
                [np.quantile(b_scores[:, 0], quantiles), np.quantile(b_scores[:, 1], quantiles)]
            ).astype(np.float32)
            a_winners = np.clip(np.quantile(a_winners, quantiles), 0.02, 0.98).astype(np.float32)
            b_winners = np.clip(np.quantile(b_winners, quantiles), 0.02, 0.98).astype(np.float32)
            combined_totals = (a_scores.mean(axis=1) + b_scores.mean(axis=1)) / 2.0

        centroids = np.quantile(combined_totals, np.linspace(0.1, 0.9, num=k)).astype(np.float32)
        assignments = np.zeros(shape=(combined_totals.shape[0],), dtype=np.int64)
        for _ in range(8):
            distances = np.abs(combined_totals[:, None] - centroids[None, :])
            assignments = np.argmin(distances, axis=1)
            updated = centroids.copy()
            for idx in range(k):
                bucket = combined_totals[assignments == idx]
                if bucket.size:
                    updated[idx] = float(bucket.mean())
            if np.allclose(updated, centroids, atol=1e-3):
                centroids = updated
                break
            centroids = updated

        scenario_a_scores: list[np.ndarray] = []
        scenario_b_scores: list[np.ndarray] = []
        scenario_a_winners: list[float] = []
        scenario_b_winners: list[float] = []
        scenario_probs: list[float] = []
        scenario_totals: list[float] = []
        for idx in range(k):
            mask = assignments == idx
            if not np.any(mask):
                nearest_idx = int(np.argmin(np.abs(combined_totals - float(centroids[idx]))))
                mask = np.zeros_like(combined_totals, dtype=bool)
                mask[nearest_idx] = True
            block_a = a_scores[mask]
            block_b = b_scores[mask]
            scenario_a_scores.append(block_a.mean(axis=0))
            scenario_b_scores.append(block_b.mean(axis=0))
            scenario_a_winners.append(float(np.clip(a_winners[mask].mean(), 0.02, 0.98)))
            scenario_b_winners.append(float(np.clip(b_winners[mask].mean(), 0.02, 0.98)))
            scenario_probs.append(float(np.mean(mask)))
            scenario_totals.append(float((block_a.mean(axis=1).mean() + block_b.mean(axis=1).mean()) / 2.0))

        scenario_a_arr = np.asarray(scenario_a_scores, dtype=np.float32)
        scenario_b_arr = np.asarray(scenario_b_scores, dtype=np.float32)
        scenario_a_winners_arr = np.asarray(scenario_a_winners, dtype=np.float32)
        scenario_b_winners_arr = np.asarray(scenario_b_winners, dtype=np.float32)
        scenario_probs_arr = np.asarray(scenario_probs, dtype=np.float32)
        scenario_totals_arr = np.asarray(scenario_totals, dtype=np.float32)

        order = np.argsort(scenario_totals_arr)
        scenario_a_arr = scenario_a_arr[order]
        scenario_b_arr = scenario_b_arr[order]
        scenario_a_winners_arr = scenario_a_winners_arr[order]
        scenario_b_winners_arr = scenario_b_winners_arr[order]
        scenario_probs_arr = scenario_probs_arr[order]
        scenario_totals_arr = scenario_totals_arr[order]

        spread_ready = bool(
            np.std(combined_totals) >= 4.0
            or np.std(a_winners) >= 0.04
            or np.std(b_winners) >= 0.04
        )
        if spread_ready and scenario_totals_arr.size > 1:
            min_gap = max(2.0, float(np.std(combined_totals) * 0.18))
            gaps = np.diff(scenario_totals_arr)
            linear_pattern = bool(
                gaps.size >= 2
                and float(np.std(gaps)) <= max(0.9, float(np.std(combined_totals)) * 0.08)
            )
            if np.any(gaps < min_gap) or linear_pattern:
                anchor_indices = self._anchor_indices_from_distribution(combined_totals, k)
                if anchor_indices.size != k:
                    sorted_idx = np.argsort(combined_totals)
                    anchor_positions = np.linspace(0, len(sorted_idx) - 1, num=k)
                    anchor_indices = sorted_idx[np.round(anchor_positions).astype(int)]
                scenario_a_arr = a_scores[anchor_indices].astype(np.float32)
                scenario_b_arr = b_scores[anchor_indices].astype(np.float32)
                scenario_a_winners_arr = np.clip(a_winners[anchor_indices], 0.02, 0.98).astype(np.float32)
                scenario_b_winners_arr = np.clip(b_winners[anchor_indices], 0.02, 0.98).astype(np.float32)
                scenario_totals_arr = ((scenario_a_arr.mean(axis=1) + scenario_b_arr.mean(axis=1)) / 2.0).astype(
                    np.float32
                )
                order = np.argsort(scenario_totals_arr)
                scenario_a_arr = scenario_a_arr[order]
                scenario_b_arr = scenario_b_arr[order]
                scenario_a_winners_arr = scenario_a_winners_arr[order]
                scenario_b_winners_arr = scenario_b_winners_arr[order]
                scenario_totals_arr = scenario_totals_arr[order]
                distances = np.abs(combined_totals[:, None] - scenario_totals_arr[None, :])
                nearest = np.argmin(distances, axis=1)
                counts = np.bincount(nearest, minlength=k).astype(np.float32)
                total_counts = float(np.sum(counts))
                if total_counts > 0:
                    scenario_probs_arr = counts / total_counts
                else:
                    scenario_probs_arr = np.full(shape=(k,), fill_value=1.0 / float(k), dtype=np.float32)

        prob_sum = float(np.sum(scenario_probs_arr))
        if prob_sum <= 0:
            scenario_probs_arr = np.full(shape=(k,), fill_value=1.0 / float(k), dtype=np.float32)
        else:
            scenario_probs_arr = scenario_probs_arr / prob_sum
        confidence_base = 0.61 if self.model_loaded else 0.48
        winner_center = np.clip((scenario_a_winners_arr + scenario_b_winners_arr) / 2.0, 0.02, 0.98)
        winner_certainty = np.abs(winner_center - 0.5) * 2.0
        global_spread = float(np.std(combined_totals))
        spread_penalty = float(np.clip(global_spread / 24.0, 0.0, 0.24))
        confidence_arr = np.clip(
            confidence_base + (0.22 * scenario_probs_arr) + (0.18 * winner_certainty) - spread_penalty,
            0.26,
            0.92,
        )

        return (
            scenario_a_arr,
            scenario_a_winners_arr,
            scenario_b_arr,
            scenario_b_winners_arr,
            scenario_probs_arr,
            confidence_arr.astype(np.float32),
            scenario_totals_arr,
        )

    @staticmethod
    def _ensemble_uncertainty(
        score_samples: np.ndarray,
        winner_probs: np.ndarray,
    ) -> dict[str, float | bool]:
        if score_samples.size == 0:
            return {
                "variance_team_a": 0.0,
                "variance_team_b": 0.0,
                "variance_winner_prob": 0.0,
                "std_team_a": 0.0,
                "std_team_b": 0.0,
                "std_winner_prob": 0.0,
                "ensemble_disagreement_score": 0.0,
                "low_uncertainty_case": True,
            }
        var_a = float(np.var(score_samples[:, 0]))
        var_b = float(np.var(score_samples[:, 1]))
        var_w = float(np.var(winner_probs))
        std_a = float(np.sqrt(var_a))
        std_b = float(np.sqrt(var_b))
        std_w = float(np.sqrt(var_w))
        disagreement = float(np.clip(((std_a + std_b) / 40.0 + (std_w / 0.25)) / 2.0, 0.0, 1.0))
        low_uncertainty = bool((std_a < 3.0) and (std_b < 3.0) and (std_w < 0.03))
        return {
            "variance_team_a": round(var_a, 4),
            "variance_team_b": round(var_b, 4),
            "variance_winner_prob": round(var_w, 6),
            "std_team_a": round(std_a, 4),
            "std_team_b": round(std_b, 4),
            "std_winner_prob": round(std_w, 6),
            "ensemble_disagreement_score": round(disagreement, 4),
            "low_uncertainty_case": low_uncertainty,
        }

    @staticmethod
    def _cricket_scenario_story(
        *,
        label: str,
        team_a_score: float,
        team_b_score: float,
    ) -> str:
        label_key = label.strip().lower()
        total = float(team_a_score + team_b_score)
        gap = abs(float(team_a_score) - float(team_b_score))
        leader = "Team A" if team_a_score >= team_b_score else "Team B"
        if label_key == "low":
            if gap >= 12:
                return f"Controlled match with {leader} better in low-scoring conditions"
            return "Controlled match where both batting units face pressure"
        if label_key == "baseline":
            if gap <= 6:
                return "Balanced contest close to expected scoring conditions"
            return f"Expected conditions with a slight edge to {leader}"
        if label_key == "high":
            return "Batting-friendly conditions pushing both totals upward"
        if total >= 360:
            return "Volatile high-scoring script with late-innings swing potential"
        return "Volatile match script with multiple finishing paths"

    @staticmethod
    def _football_scenario_story(
        *,
        label: str,
        home_goals: float,
        away_goals: float,
    ) -> str:
        label_key = str(label).strip().lower()
        total = float(home_goals + away_goals)
        gap = abs(float(home_goals) - float(away_goals))
        if label_key == "low":
            if gap >= 2:
                return "Low-event match script with one side controlling tempo"
            return "Cagey game state with few clear-cut chances"
        if label_key == "baseline":
            if gap <= 1:
                return "Even xG-style script with either side able to edge it late"
            return "Typical league scoring band with finishing variance deciding margins"
        if label_key == "high":
            return "Open game model with transition chances lifting the goal expectation"
        if total >= 5:
            return "High-tempo script where defensive structure breaks down repeatedly"
        return "Wide outcome band from set-piece and finishing volatility"

    @staticmethod
    def _reason_factor(
        *,
        feature: str,
        value: float,
        baseline: float,
        unit: str,
        impact: str,
        explanation: str,
    ) -> dict[str, Any]:
        return {
            "feature": feature,
            "value": round(float(value), 3),
            "baseline": round(float(baseline), 3),
            "delta": round(float(value) - float(baseline), 3),
            "unit": unit,
            "impact": impact,
            "explanation": explanation.strip(),
        }

    def _cricket_scenario_reasons(
        self,
        *,
        label: str,
        score: float,
        team_a: str,
        team_b: str,
        feature_dict: dict[str, float],
        anomaly: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        med = self.feature_medians

        def _value(name: str, default: float = 0.0) -> float:
            raw = feature_dict.get(name, med.get(name, default))
            try:
                return float(raw)
            except (TypeError, ValueError):
                return float(med.get(name, default))

        team_a_runs = _value("team_a_avg_runs_last_5", self.baseline_score)
        team_b_runs = _value("team_b_avg_runs_last_5", self.baseline_score)
        team_a_wickets = _value("team_a_avg_wickets_last_5", 7.0)
        team_b_wickets = _value("team_b_avg_wickets_last_5", 7.0)
        team_a_run_rate = _value("team_a_run_rate_trend", 0.0)
        team_b_run_rate = _value("team_b_run_rate_trend", 0.0)
        venue_avg = _value("venue_avg_score", self.baseline_score)
        venue_chase = _value("venue_chase_success_rate", 0.5)
        venue_defend = _value("venue_defend_bias", 0.0)
        run_diff = team_a_runs - team_b_runs
        run_rate_diff = team_a_run_rate - team_b_run_rate
        form_diff = _value("recent_form_diff", 0.0)
        h2h_diff = _value("head_to_head_win_diff", 0.0)
        team_a_chase = _value("team_a_chase_success_rate", 0.5)
        team_b_chase = _value("team_b_chase_success_rate", 0.5)
        team_a_defend = _value("team_a_defend_success_rate", 0.5)
        team_b_defend = _value("team_b_defend_success_rate", 0.5)
        label_key = str(label).strip().lower()
        reasons: list[dict[str, Any]] = []
        form_gap = abs(form_diff)
        run_gap = abs(run_diff)
        pace_gap = abs(run_rate_diff)
        chase_gap = abs(team_a_chase - team_b_chase)
        defend_gap = abs(team_a_defend - team_b_defend)
        team_with_scoring_edge = team_a if run_diff >= 0 else team_b
        team_with_chase_edge = team_a if team_a_chase >= team_b_chase else team_b
        team_with_defend_edge = team_a if team_a_defend >= team_b_defend else team_b

        if label_key == "low":
            pressure_team = team_a if team_a_wickets >= team_b_wickets else team_b
            if max(team_a_wickets, team_b_wickets) >= float(med.get("team_a_avg_wickets_last_5", 7.0)) + 0.4:
                reason_one = (
                    f"{pressure_team} have shown fragile starts, which pulls this outcome toward a controlled innings."
                )
            else:
                reason_one = "Early batting rhythm looks vulnerable on both sides, keeping this script contained."
            if venue_avg <= self.baseline_score - 4.0 or venue_chase < 0.49:
                reason_two = "Venue patterns favor disciplined bowling spells over long acceleration phases."
            else:
                reason_two = "Late chase pressure on this ground has recently prevented sustained run surges."
            if form_gap < 4.5:
                reason_three = "Neither lineup carries enough recent separation to lift the scoring floor."
            else:
                trailing_team = team_b if team_with_scoring_edge == team_a else team_a
                reason_three = f"{trailing_team} arrive with a softer recent batting base, narrowing low-end outcomes."
            reasons.extend(
                [
                    self._reason_factor(
                        feature="wickets_pressure",
                        value=max(team_a_wickets, team_b_wickets),
                        baseline=float(med.get("team_a_avg_wickets_last_5", 7.0)),
                        unit="wickets",
                        impact="negative",
                        explanation=reason_one,
                    ),
                    self._reason_factor(
                        feature="venue_control",
                        value=venue_avg,
                        baseline=self.baseline_score,
                        unit="runs",
                        impact="negative" if venue_avg < self.baseline_score else "neutral",
                        explanation=reason_two,
                    ),
                    self._reason_factor(
                        feature="batting_floor_gap",
                        value=min(team_a_runs, team_b_runs),
                        baseline=self.baseline_score,
                        unit="runs",
                        impact="negative",
                        explanation=reason_three,
                    ),
                ]
            )
        elif label_key == "baseline":
            if form_gap <= 5.0:
                reason_one = "Both teams enter with similar recent scoring, so the central outcome stays balanced."
            else:
                reason_one = f"{team_with_scoring_edge} hold a mild form edge, but not enough to break match balance."
            if abs(h2h_diff) <= 0.12:
                reason_two = "Recent head-to-head meetings have been tight, keeping this path close to neutral."
            else:
                edge_team = team_a if h2h_diff >= 0 else team_b
                reason_two = f"{edge_team} carry a slight historical edge, though not a decisive one."
            if max(chase_gap, defend_gap) <= 0.1:
                reason_three = "Chase and defend profiles are close enough that execution should decide late phases."
            else:
                profile_team = team_with_chase_edge if chase_gap >= defend_gap else team_with_defend_edge
                reason_three = f"{profile_team} have a situational edge, but this script still projects as balanced."
            reasons.extend(
                [
                    self._reason_factor(
                        feature="recent_form_diff",
                        value=form_diff,
                        baseline=float(med.get("recent_form_diff", 0.0)),
                        unit="index",
                        impact="neutral",
                        explanation=reason_one,
                    ),
                    self._reason_factor(
                        feature="head_to_head_balance",
                        value=h2h_diff,
                        baseline=float(med.get("head_to_head_win_diff", 0.0)),
                        unit="ratio",
                        impact="neutral",
                        explanation=reason_two,
                    ),
                    self._reason_factor(
                        feature="branch_balance",
                        value=(team_a_chase + team_b_chase + team_a_defend + team_b_defend) / 4.0,
                        baseline=0.5,
                        unit="ratio",
                        impact="neutral",
                        explanation=reason_three,
                    ),
                ]
            )
        elif label_key == "high":
            if run_gap >= 6.0:
                reason_one = f"{team_with_scoring_edge} carry stronger recent batting continuity, raising this ceiling."
            else:
                reason_one = "Both batting units have enough recent momentum to sustain a higher-scoring path."
            if venue_avg >= self.baseline_score + 4.0:
                reason_two = "Venue history has rewarded proactive batting, lifting expected totals in this scenario."
            else:
                reason_two = "Current conditions still leave room for extended scoring phases from both sides."
            if chase_gap >= 0.08:
                reason_three = f"{team_with_chase_edge} have the chase profile to keep pressure on even with big targets."
            else:
                reason_three = "Chasing efficiency from both teams keeps high totals realistic deep into the game."
            reasons.extend(
                [
                    self._reason_factor(
                        feature="recent_runs_edge",
                        value=max(team_a_runs, team_b_runs),
                        baseline=self.baseline_score,
                        unit="runs",
                        impact="positive",
                        explanation=reason_one,
                    ),
                    self._reason_factor(
                        feature="venue_batting_signal",
                        value=venue_avg,
                        baseline=self.baseline_score,
                        unit="runs",
                        impact="positive" if venue_avg >= self.baseline_score else "neutral",
                        explanation=reason_two,
                    ),
                    self._reason_factor(
                        feature="chase_profile_edge",
                        value=max(team_a_chase, team_b_chase),
                        baseline=0.5,
                        unit="ratio",
                        impact="positive",
                        explanation=reason_three,
                    ),
                ]
            )
        else:
            volatility = pace_gap + (form_gap / 18.0) + abs(venue_defend)
            if pace_gap >= 0.14:
                reason_one = "Recent scoring tempo has been uneven, making this script prone to sharp momentum swings."
            else:
                reason_one = "Shot-making rhythm has fluctuated enough to keep this outcome volatile."
            if max(chase_gap, defend_gap) >= 0.11:
                swing_team = team_with_chase_edge if chase_gap >= defend_gap else team_with_defend_edge
                reason_two = f"{swing_team} show a stronger chase-defend split, widening late-game outcome paths."
            else:
                reason_two = "Neither side has held a stable branch advantage, so late phases can turn quickly."
            if anomaly and bool(anomaly.get("odd_variant_flag")):
                reason_three = "Recent match conditions have been irregular, increasing late-innings unpredictability."
            else:
                reason_three = "End-game pressure remains sensitive to small execution swings in this scenario."
            reasons.extend(
                [
                    self._reason_factor(
                        feature="tempo_volatility",
                        value=pace_gap,
                        baseline=0.0,
                        unit="run_rate",
                        impact="positive" if pace_gap >= 0.1 else "neutral",
                        explanation=reason_one,
                    ),
                    self._reason_factor(
                        feature="branch_sensitivity",
                        value=max(chase_gap, defend_gap),
                        baseline=0.0,
                        unit="ratio",
                        impact="positive",
                        explanation=reason_two,
                    ),
                    self._reason_factor(
                        feature="volatility_signal",
                        value=volatility,
                        baseline=0.45,
                        unit="index",
                        impact="positive" if volatility >= 0.45 else "neutral",
                        explanation=reason_three,
                    ),
                ]
            )

        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in reasons:
            explanation = str(row.get("explanation", "")).strip().lower()
            if not explanation or explanation in seen:
                continue
            seen.add(explanation)
            unique.append(row)
            if len(unique) >= 3:
                break
        if unique:
            return unique
        return [
            self._reason_factor(
                feature="match_balance",
                value=0.0,
                baseline=0.0,
                unit="index",
                impact="neutral",
                explanation="Balanced matchup with no strong differentiator",
            )
        ]

    def _cricket_match_insight(
        self,
        *,
        team_a: str,
        team_b: str,
        feature_dict: dict[str, float],
        scenarios: list[dict[str, Any]],
        anomaly: dict[str, Any] | None,
    ) -> str:
        def _value(name: str, default: float) -> float:
            raw = feature_dict.get(name, self.feature_medians.get(name, default))
            try:
                return float(raw)
            except (TypeError, ValueError):
                return float(default)

        run_diff = _value("team_a_avg_runs_last_5", self.baseline_score) - _value(
            "team_b_avg_runs_last_5",
            self.baseline_score,
        )
        chase_diff = _value("team_a_chase_success_rate", 0.5) - _value("team_b_chase_success_rate", 0.5)
        venue_avg = _value("venue_avg_score", self.baseline_score)
        venue_delta = venue_avg - self.baseline_score

        order_sensitive = any(
            isinstance(row.get("team_a_first"), dict)
            and isinstance(row.get("team_b_first"), dict)
            and str(row["team_a_first"].get("winner") or "") != str(row["team_b_first"].get("winner") or "")
            for row in scenarios
            if isinstance(row, dict)
        )
        signal_strength = {
            "form": abs(run_diff),
            "chase": abs(chase_diff) * 60.0,
            "venue": abs(venue_delta),
        }
        strongest_signal = max(signal_strength, key=signal_strength.get)
        edge_team = team_a if (run_diff + (chase_diff * 25.0)) >= 0 else team_b

        if signal_strength[strongest_signal] < 5.0:
            insight = "Balanced matchup with no clear separation in recent signals."
        elif strongest_signal == "form":
            insight = f"{edge_team} are slightly favored by stronger recent batting form."
        elif strongest_signal == "chase":
            insight = f"{edge_team} carry a small edge from chase-versus-defend profile."
        elif venue_delta >= 0:
            insight = "Venue trend supports a higher-scoring game with both teams in contention."
        else:
            insight = "Venue trend points to a tighter scoring game where execution matters most."

        if order_sensitive:
            insight = insight.rstrip(".") + "; batting order can still flip the result."
        if anomaly and bool(anomaly.get("odd_variant_flag")):
            insight = insight.rstrip(".") + "; volatility is elevated."
        return insight

    def _confidence_scores(self, predictions: np.ndarray) -> list[float]:
        center = float(np.mean(predictions))
        spread = float(max(predictions) - min(predictions) + 1.0)
        base = 0.72 if self.model_loaded else 0.54
        confidences = []
        for pred in predictions:
            distance = abs(float(pred) - center)
            conf = base - (0.16 * (distance / spread))
            confidences.append(float(np.clip(conf, 0.35, 0.92)))
        return confidences

    @staticmethod
    def _is_placeholder_player(name: Any) -> bool:
        return bool(PLACEHOLDER_PLAYER_PATTERN.search(str(name or "")))

    def _fallback_cricket_pool(self, team: str) -> list[str]:
        pool = CRICKET_FALLBACK_PLAYERS.get(team.strip().lower())
        if pool:
            return pool
        all_players = [name for values in CRICKET_FALLBACK_PLAYERS.values() for name in values]
        return all_players[:4]

    def _fallback_football_pool(self, team: str) -> list[str]:
        pool = FOOTBALL_FALLBACK_PLAYERS.get(team.strip().lower())
        if pool:
            return pool
        all_players = [name for values in FOOTBALL_FALLBACK_PLAYERS.values() for name in values]
        return all_players[:4]

    @staticmethod
    def _branch_outcome(
        *,
        team_a: str,
        team_b: str,
        batting_team: str,
        bowling_team: str,
        batting_score: float,
        chase_score: float,
        winner_prob_team_a: float,
    ) -> dict[str, Any]:
        batting = int(np.clip(round(batting_score), 90, 280))
        chase = int(np.clip(round(chase_score), 90, 280))
        if chase > batting:
            winner = bowling_team
        elif chase < batting:
            winner = batting_team
        else:
            winner = team_a if float(winner_prob_team_a) >= 0.5 else team_b
        return {
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "batting_score": batting,
            "chase_score": chase,
            "winner": winner,
            "winner_probability_team_a": round(float(np.clip(winner_prob_team_a, 0.02, 0.98)), 2),
        }

    @staticmethod
    def _scenario_winner(
        *,
        team_a: str,
        team_b: str,
        team_a_score: float,
        team_b_score: float,
        winner_prob_center: float,
        team_a_first: dict[str, Any] | None,
        team_b_first: dict[str, Any] | None,
    ) -> str:
        branch_winners = [
            str((team_a_first or {}).get("winner") or "").strip(),
            str((team_b_first or {}).get("winner") or "").strip(),
        ]
        non_empty_branch_winners = [winner for winner in branch_winners if winner]
        if len(non_empty_branch_winners) == 2 and non_empty_branch_winners[0] == non_empty_branch_winners[1]:
            return non_empty_branch_winners[0]
        if len(non_empty_branch_winners) == 2 and non_empty_branch_winners[0] != non_empty_branch_winners[1]:
            return team_a if winner_prob_center >= 0.5 else team_b

        winner = team_a if team_a_score > team_b_score else team_b
        if int(round(team_a_score)) == int(round(team_b_score)):
            winner = team_a if winner_prob_center >= 0.5 else team_b
        return winner

    def _predict_cricket_players(self, team_a: str, team_b: str) -> dict[str, Any]:
        def _fallback_candidates(
            *,
            role: str,
            team: str,
            fallback_team: str,
            count: int = 3,
        ) -> list[dict[str, Any]]:
            pool = self._fallback_cricket_pool(team) + self._fallback_cricket_pool(fallback_team)
            if role == "bowler":
                pool = sorted(pool, key=lambda name: 0 if str(name).strip().lower() in CRICKET_BOWLER_NAMES else 1)
            if role == "batsman":
                pool = sorted(pool, key=lambda name: 0 if str(name).strip().lower() in CRICKET_BATTER_NAMES else 1)
            unique_names: list[str] = []
            for name in pool:
                if name not in unique_names:
                    unique_names.append(name)
                if len(unique_names) >= count:
                    break
            rows: list[dict[str, Any]] = []
            for idx, name in enumerate(unique_names):
                primary_team = team if idx < 2 else fallback_team
                if role == "batsman":
                    reason_options = [
                        (
                            "Consistent top-order scoring makes him central to building the innings foundation",
                            "Composure against early pressure helps protect the batting core",
                        ),
                        (
                            "Fast scoring intent gives this lineup a strong launch in attacking phases",
                            "Powerplay momentum can shape the direction of this matchup",
                        ),
                        (
                            "Middle-order control helps recover quickly after early setbacks",
                            "Strike rotation value rises when chase pressure tightens",
                        ),
                    ]
                elif role == "bowler":
                    reason_options = [
                        (
                            "Control through middle overs can slow scoring before the finish",
                            "Pressure overs often decide whether batting momentum survives",
                        ),
                        (
                            "Death-over discipline is vital when totals stay within reach",
                            "Late breakthroughs can swing the result in either direction",
                        ),
                        (
                            "New-ball pressure helps keep the scoring floor under control",
                            "Containment spells matter most when both sides are close",
                        ),
                    ]
                else:
                    reason_options = [
                        (
                            "Involvement across both innings increases impact in tight matches",
                            "Balanced batting and bowling value fits this matchup",
                        ),
                        (
                            "Two-phase contribution keeps this player relevant across changing scripts",
                            "Can influence both defend and chase situations",
                        ),
                        (
                            "Reliable all-round involvement raises impact in volatile finishes",
                            "Field pressure plus utility overs can stabilize momentum",
                        ),
                    ]
                reason_one, reason_two = reason_options[idx % len(reason_options)]
                rows.append(
                    {
                        "name": name,
                        "role": "all-round impact" if role == "match_impact" else role,
                        "team": primary_team,
                        "reason": [
                            self._reason_factor(
                                feature=f"{role}_fallback_primary",
                                value=1.0,
                                baseline=0.0,
                                unit="index",
                                impact="positive",
                                explanation=reason_one,
                            ),
                            self._reason_factor(
                                feature=f"{role}_fallback_secondary",
                                value=1.0,
                                baseline=0.0,
                                unit="index",
                                impact="neutral",
                                explanation=reason_two,
                            ),
                        ],
                        "confidence": round(0.52 - (idx * 0.03), 2),
                    }
                )
            return rows

        if self.player_form.empty:
            batters = _fallback_candidates(role="batsman", team=team_a, fallback_team=team_b, count=3)
            bowlers = _fallback_candidates(role="bowler", team=team_b, fallback_team=team_a, count=3)
            impact = _fallback_candidates(role="match_impact", team=team_a, fallback_team=team_b, count=3)
            return {
                "best_player": batters[0],
                "best_bowler": bowlers[0],
                "man_of_the_match": impact[0],
                "players": {
                    "top_batsmen": batters[:3],
                    "top_bowlers": bowlers[:3],
                    "top_match_impact": impact[:3],
                    "top_goal_scorers": [],
                    "top_standout": [],
                },
            }

        subset = self.player_form[self.player_form["team"].str.lower().isin([team_a.lower(), team_b.lower()])].copy()
        if subset.empty:
            subset = self.player_form.copy()
        subset = subset[~subset["player"].apply(self._is_placeholder_player)].copy()
        if subset.empty:
            batters = _fallback_candidates(role="batsman", team=team_a, fallback_team=team_b, count=3)
            bowlers = _fallback_candidates(role="bowler", team=team_b, fallback_team=team_a, count=3)
            impact = _fallback_candidates(role="match_impact", team=team_a, fallback_team=team_b, count=3)
            return {
                "best_player": batters[0],
                "best_bowler": bowlers[0],
                "man_of_the_match": impact[0],
                "players": {
                    "top_batsmen": batters[:3],
                    "top_bowlers": bowlers[:3],
                    "top_match_impact": impact[:3],
                    "top_goal_scorers": [],
                    "top_standout": [],
                },
            }

        for col in [
            "recent_runs",
            "batting_form",
            "strike_rate",
            "win_rate",
            "recent_wickets",
            "avg_wickets",
            "bowling_form",
            "economy",
            "impact_score",
            "total_balls_bowled",
            "total_balls",
        ]:
            source = subset[col] if col in subset.columns else pd.Series(0.0, index=subset.index)
            subset[col] = pd.to_numeric(source, errors="coerce").fillna(0.0)
        player_name_norm = subset["player"].astype(str).str.lower().str.strip()
        subset["known_bowler"] = player_name_norm.isin(CRICKET_BOWLER_NAMES).astype(float)
        subset["known_batter"] = player_name_norm.isin(CRICKET_BATTER_NAMES).astype(float)

        subset["batter_rank_score"] = (
            subset["recent_runs"] * 0.45
            + subset["batting_form"] * 0.35
            + subset["strike_rate"] * 0.12
            + subset["win_rate"] * 14.0
            + subset["known_batter"] * 14.0
            - subset["known_bowler"] * 7.0
        )
        subset["bowler_rank_score"] = (
            subset["recent_wickets"] * 11.5
            + subset["avg_wickets"] * 8.5
            + subset["bowling_form"] * 0.22
            + np.maximum(0.0, 8.8 - subset["economy"]) * 7.0
            + subset["win_rate"] * 8.0
            + subset["known_bowler"] * 20.0
            - subset["known_batter"] * 10.0
        )
        subset["bowling_specialist_score"] = (
            subset["recent_wickets"] * 10.0
            + subset["avg_wickets"] * 8.0
            + np.maximum(0.0, subset["bowling_form"] - subset["batting_form"]) * 0.35
            + np.maximum(0.0, 8.4 - subset["economy"]) * 6.5
            - (subset["recent_runs"] * 0.08)
        )
        subset["bowler_rank_score"] = subset["bowler_rank_score"] + subset["bowling_specialist_score"] * 0.7
        subset["impact_rank_score"] = (
            subset["impact_score"] * 0.55
            + subset["batting_form"] * 0.2
            + subset["bowling_form"] * 0.15
            + subset["win_rate"] * 16.0
        )

        runs_median = float(subset["recent_runs"].median()) if not subset.empty else 30.0
        strike_median = float(subset["strike_rate"].median()) if not subset.empty else 130.0
        wicket_median = float(subset["recent_wickets"].median()) if not subset.empty else 1.0
        economy_median = float(subset["economy"].replace(0.0, np.nan).median()) if not subset.empty else 8.4
        if np.isnan(economy_median):
            economy_median = 8.4
        impact_median = float(subset["impact_score"].median()) if not subset.empty else 55.0

        def _rank_conf(rank: float, max_rank: float) -> float:
            if max_rank <= 0.0:
                return 0.45
            return float(np.clip(0.45 + (0.45 * (rank / max_rank)), 0.45, 0.92))

        def _as_batter(row: pd.Series, max_rank: float) -> dict[str, Any]:
            recent_runs = float(row["recent_runs"])
            strike_rate = float(row["strike_rate"])
            win_rate = float(row["win_rate"])
            reason_primary = "Reliable top-order scoring helps keep the innings structure intact"
            if recent_runs >= runs_median + 8.0:
                reason_primary = "Recent top-order output suggests this batter can control long scoring phases"
            elif strike_rate >= strike_median + 8.0:
                reason_primary = "Shot tempo and strike rotation make this batter a catalyst in faster scripts"
            if recent_runs <= max(5.0, runs_median - 5.0):
                reason_primary = "Even with mixed recent returns, this role remains key to rebuilding after early setbacks"
            reason_secondary = "Composure against phase changes helps sustain middle-over momentum"
            if win_rate >= 0.55:
                reason_secondary = "Recent influence in winning passages strengthens expected match impact"
            return {
                "name": str(row["player"]),
                "role": "batsman",
                "team": str(row["team"]),
                "reason": [
                    self._reason_factor(
                        feature="batsman_recent_runs",
                        value=recent_runs,
                        baseline=runs_median,
                        unit="runs",
                        impact="positive" if recent_runs >= runs_median else "neutral",
                        explanation=reason_primary,
                    ),
                    self._reason_factor(
                        feature="batsman_scoring_tempo",
                        value=strike_rate,
                        baseline=strike_median,
                        unit="strike_rate",
                        impact="positive" if strike_rate >= strike_median else "neutral",
                        explanation=reason_secondary,
                    ),
                ],
                "confidence": round(_rank_conf(float(row["batter_rank_score"]), max_rank), 2),
            }

        def _as_bowler(row: pd.Series, max_rank: float) -> dict[str, Any]:
            recent_wickets = float(row["recent_wickets"])
            economy = float(row["economy"])
            reason_primary = "Regular wicket pressure gives this bowler leverage in middle phases"
            if recent_wickets < max(0.2, wicket_median - 0.3):
                reason_primary = "Expected high-leverage overs still give this bowler breakthrough value"
            reason_secondary = "Control across transition overs helps contain scoring acceleration"
            if economy <= 0:
                reason_secondary = "Projected workload keeps this bowler relevant in tight passages"
            elif economy <= economy_median:
                reason_secondary = "Recent economy control supports both defend and chase pressure phases"
            elif recent_wickets >= wicket_median:
                reason_secondary = "Even with variable economy, wicket threat keeps this bowler high impact"
            return {
                "name": str(row["player"]),
                "role": "bowler",
                "team": str(row["team"]),
                "reason": [
                    self._reason_factor(
                        feature="bowler_recent_wickets",
                        value=recent_wickets,
                        baseline=wicket_median,
                        unit="wickets",
                        impact="positive" if recent_wickets >= wicket_median else "neutral",
                        explanation=reason_primary,
                    ),
                    self._reason_factor(
                        feature="bowler_economy",
                        value=economy,
                        baseline=economy_median,
                        unit="economy",
                        impact="positive" if economy > 0 and economy <= economy_median else "neutral",
                        explanation=reason_secondary,
                    ),
                ],
                "confidence": round(_rank_conf(float(row["bowler_rank_score"]), max_rank), 2),
            }

        def _as_impact(row: pd.Series, max_rank: float) -> dict[str, Any]:
            impact_score = float(row["impact_score"])
            win_rate = float(row["win_rate"])
            batting_form = float(row["batting_form"])
            bowling_form = float(row["bowling_form"])
            if impact_score >= impact_median + 8.0:
                reason_primary = "Influence across both innings increases impact in tight matches"
            elif batting_form >= bowling_form + 10.0:
                reason_primary = "Batting-led influence can decide momentum turns in close games"
            elif bowling_form >= batting_form + 10.0:
                reason_primary = "Bowling-led influence can swing pressure phases late"
            else:
                reason_primary = "Balanced batting and bowling contribution suits volatile match scripts"

            reason_secondary = "Two-phase involvement keeps this player relevant across scenario swings"
            if win_rate >= 0.55:
                reason_secondary = "Contribution pattern in wins points to dependable match influence"
            return {
                "name": str(row["player"]),
                "role": "all-round impact",
                "team": str(row["team"]),
                "reason": [
                    self._reason_factor(
                        feature="impact_score",
                        value=impact_score,
                        baseline=impact_median,
                        unit="index",
                        impact="positive" if impact_score >= impact_median else "neutral",
                        explanation=reason_primary,
                    ),
                    self._reason_factor(
                        feature="impact_win_rate",
                        value=win_rate,
                        baseline=0.5,
                        unit="ratio",
                        impact="positive" if win_rate >= 0.5 else "neutral",
                        explanation=reason_secondary,
                    ),
                ],
                "confidence": round(_rank_conf(float(row["impact_rank_score"]), max_rank), 2),
            }

        batter_pool = subset[
            (subset["batting_form"] >= subset["bowling_form"] * 0.82)
            | (subset["recent_runs"] >= runs_median)
            | (subset["strike_rate"] >= strike_median)
        ]
        if batter_pool.empty:
            batter_pool = subset
        top_batters = batter_pool.sort_values("batter_rank_score", ascending=False).head(3)
        bowling_workload_median = float(subset["total_balls_bowled"].median()) if not subset.empty else 0.0
        bowler_pool = subset[
            (subset["recent_wickets"] >= max(0.8, wicket_median - 0.1))
            | (subset["avg_wickets"] >= 0.7)
            | (subset["bowling_form"] >= subset["batting_form"] * 0.95)
            | (subset["total_balls_bowled"] >= max(30.0, bowling_workload_median))
            | (
                subset["bowling_specialist_score"]
                >= float(subset["bowling_specialist_score"].median())
            )
        ]
        if bowler_pool.empty:
            bowler_pool = subset
        top_bowlers = bowler_pool.sort_values("bowler_rank_score", ascending=False).head(3)
        top_impact = subset.sort_values("impact_rank_score", ascending=False).head(3)

        max_batter = float(top_batters["batter_rank_score"].max()) if not top_batters.empty else 0.0
        max_bowler = float(top_bowlers["bowler_rank_score"].max()) if not top_bowlers.empty else 0.0
        max_impact = float(top_impact["impact_rank_score"].max()) if not top_impact.empty else 0.0

        batsmen_candidates = [_as_batter(row, max_batter) for _, row in top_batters.iterrows()]
        bowler_candidates = [_as_bowler(row, max_bowler) for _, row in top_bowlers.iterrows()]
        impact_candidates = [_as_impact(row, max_impact) for _, row in top_impact.iterrows()]

        def _dedupe(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
            deduped: list[dict[str, Any]] = []
            seen: set[str] = set()
            for row in items:
                name = str(row.get("name", "")).strip().lower()
                if not name or name in seen:
                    continue
                seen.add(name)
                deduped.append(row)
            return deduped

        batsmen_candidates = _dedupe(batsmen_candidates)
        bowler_candidates = _dedupe(bowler_candidates)
        impact_candidates = _dedupe(impact_candidates)
        bowler_name_ok = [
            row
            for row in bowler_candidates
            if not (
                str(row.get("name", "")).strip().lower() in CRICKET_BATTER_NAMES
                and str(row.get("name", "")).strip().lower() not in CRICKET_BOWLER_NAMES
            )
        ]
        if bowler_name_ok:
            remaining = [row for row in bowler_candidates if row not in bowler_name_ok]
            bowler_candidates = bowler_name_ok + remaining
        else:
            bowler_candidates = []

        def _ensure_min(
            items: list[dict[str, Any]],
            *,
            min_count: int,
            role: str,
            team_primary: str,
            team_secondary: str,
        ) -> list[dict[str, Any]]:
            if len(items) >= min_count:
                return items[:3]
            fallback = _fallback_candidates(
                role="match_impact" if role == "all-round impact" else role,
                team=team_primary,
                fallback_team=team_secondary,
                count=3,
            )
            existing = {str(row.get("name", "")).strip().lower() for row in items}
            for candidate in fallback:
                if len(items) >= min_count:
                    break
                name = str(candidate.get("name", "")).strip().lower()
                if not name or name in existing:
                    continue
                candidate["role"] = role
                items.append(candidate)
                existing.add(name)
            return items[:3]

        batsmen_candidates = _ensure_min(
            batsmen_candidates,
            min_count=2,
            role="batsman",
            team_primary=team_a,
            team_secondary=team_b,
        )
        bowler_candidates = _ensure_min(
            bowler_candidates,
            min_count=2,
            role="bowler",
            team_primary=team_b,
            team_secondary=team_a,
        )
        impact_candidates = _ensure_min(
            impact_candidates,
            min_count=2,
            role="all-round impact",
            team_primary=team_a,
            team_secondary=team_b,
        )

        def _diversify_reason_heads(items: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
            alternatives = {
                "batsman": [
                    "Reliable top-order presence helps stabilize the innings shape",
                    "Tempo control makes this batter valuable when scoring rhythm shifts",
                    "Shot-making range adds pressure relief during chase passages",
                ],
                "bowler": [
                    "Middle-over control is key to limiting scoring acceleration",
                    "Breakthrough pressure in transition overs can shift momentum quickly",
                    "Death-over discipline matters most when totals stay close",
                ],
                "all-round impact": [
                    "Involvement across phases keeps this player influential in tight matches",
                    "Two-way contribution helps when match momentum turns suddenly",
                    "Balanced skills keep this player relevant across scenario swings",
                ],
            }
            phrase_set = alternatives.get(role, [])
            if not phrase_set:
                return items
            seen: set[str] = set()
            for idx, row in enumerate(items):
                reasons = row.get("reason")
                if not isinstance(reasons, list) or not reasons:
                    continue
                first = reasons[0]
                if not isinstance(first, dict):
                    continue
                current = str(first.get("explanation") or "").strip().lower()
                if not current:
                    continue
                if current in seen:
                    phrase = phrase_set[idx % len(phrase_set)]
                    first["explanation"] = phrase
                    current = phrase.lower()
                seen.add(current)
            return items

        batsmen_candidates = _diversify_reason_heads(batsmen_candidates, "batsman")
        bowler_candidates = _diversify_reason_heads(bowler_candidates, "bowler")
        impact_candidates = _diversify_reason_heads(impact_candidates, "all-round impact")

        if bowler_candidates and batsmen_candidates and bowler_candidates[0]["name"] == batsmen_candidates[0]["name"]:
            replacement = next((row for row in bowler_candidates if row["name"] != batsmen_candidates[0]["name"]), None)
            if replacement is not None:
                bowler_candidates[0] = replacement

        best_player = batsmen_candidates[0] if batsmen_candidates else impact_candidates[0]
        best_bowler = bowler_candidates[0] if bowler_candidates else impact_candidates[0]
        man_of_the_match = impact_candidates[0] if impact_candidates else best_player

        return {
            "best_player": best_player,
            "best_bowler": best_bowler,
            "man_of_the_match": man_of_the_match,
            "players": {
                "top_batsmen": batsmen_candidates[:3],
                "top_bowlers": bowler_candidates[:3],
                "top_match_impact": impact_candidates[:3],
                "top_goal_scorers": [],
                "top_standout": [],
            },
        }

    def _football_team_profile(self, team: str) -> dict[str, float]:
        defaults = {
            "attack_index": 1.42,
            "defense_index": 1.1,
            "xg_for": 1.6,
            "xg_against": 1.2,
            "form_points_last5": 8.0,
            "home_advantage": 0.12,
        }
        if self.football_team_profiles.empty:
            return defaults

        subset = self.football_team_profiles[self.football_team_profiles["team"].str.lower() == team.lower()]
        if subset.empty:
            return defaults

        row = subset.iloc[0]
        return {
            "attack_index": float(row.get("attack_index", defaults["attack_index"])),
            "defense_index": float(row.get("defense_index", defaults["defense_index"])),
            "xg_for": float(row.get("xg_for", defaults["xg_for"])),
            "xg_against": float(row.get("xg_against", defaults["xg_against"])),
            "form_points_last5": float(row.get("form_points_last5", defaults["form_points_last5"])),
            "home_advantage": float(row.get("home_advantage", defaults["home_advantage"])),
        }

    def _predict_football_players(self, team_a: str, team_b: str) -> dict[str, Any]:
        if self.football_player_form.empty:
            rows = []
            for team_name in [team_a, team_b]:
                for idx, player_name in enumerate(self._fallback_football_pool(team_name)[:3]):
                    rows.append(
                        {
                            "player": player_name,
                            "team": team_name,
                            "goals_last5": max(0.0, 3.0 - idx),
                            "xg_per90": max(0.2, 0.64 - (idx * 0.12)),
                            "impact_score": 72.0 - (idx * 3.5),
                            "form_points_last5": 10.0 - idx,
                        }
                    )
            frame = pd.DataFrame(rows)
        else:
            frame = self.football_player_form.copy()
            frame = frame[frame["team"].str.lower().isin([team_a.lower(), team_b.lower()])]
            if frame.empty:
                frame = self.football_player_form.copy()
            frame = frame[~frame["player"].apply(self._is_placeholder_player)].copy()
            if frame.empty:
                rows = []
                for team_name in [team_a, team_b]:
                    for idx, player_name in enumerate(self._fallback_football_pool(team_name)[:3]):
                        rows.append(
                            {
                                "player": player_name,
                                "team": team_name,
                                "goals_last5": max(0.0, 2.0 - idx),
                                "xg_per90": max(0.2, 0.58 - (idx * 0.1)),
                                "impact_score": 69.0 - (idx * 2.8),
                                "form_points_last5": 9.0 - (idx * 0.4),
                            }
                        )
                frame = pd.DataFrame(rows)

        for col in ["goals_last5", "xg_per90", "impact_score", "form_points_last5"]:
            source = frame[col] if col in frame.columns else pd.Series(0.0, index=frame.index)
            frame[col] = pd.to_numeric(source, errors="coerce").fillna(0.0)

        frame["scorer_rank"] = frame["goals_last5"] * 8.0 + frame["xg_per90"] * 30.0 + frame["impact_score"] * 0.45
        frame["standout_rank"] = frame["impact_score"] * 0.7 + frame["form_points_last5"] * 2.5 + frame["goals_last5"] * 2.0

        goals_median = float(frame["goals_last5"].median()) if not frame.empty else 1.0
        xg_median = float(frame["xg_per90"].median()) if not frame.empty else 0.45
        impact_median = float(frame["impact_score"].median()) if not frame.empty else 60.0

        top_scorers = frame.sort_values("scorer_rank", ascending=False).head(3)
        top_standouts = frame.sort_values("standout_rank", ascending=False).head(3)
        scorer_max = float(top_scorers["scorer_rank"].max()) if not top_scorers.empty else 1.0
        standout_max = float(top_standouts["standout_rank"].max()) if not top_standouts.empty else 1.0

        def _rank_conf(value: float, max_value: float) -> float:
            if max_value <= 0:
                return 0.45
            return float(np.clip(0.45 + (0.45 * (value / max_value)), 0.45, 0.9))

        def _as_scorer(row: pd.Series) -> dict[str, Any]:
            goals = float(row.get("goals_last5", 0.0))
            xg = float(row.get("xg_per90", 0.0))
            reason_primary = "Consistent goal threat in recent matches"
            if goals >= goals_median + 0.8:
                reason_primary = "Finishing form is trending up"
            reason_secondary = "Finds shots in high-value positions"
            if xg < max(0.15, xg_median - 0.12):
                reason_secondary = "Creates chances even in tighter games"
            return {
                "name": str(row.get("player", row.get("name", "Unavailable"))),
                "role": "goal_scorer",
                "team": str(row.get("team", team_a)),
                "reason": [
                    self._reason_factor(
                        feature="goals_last5",
                        value=goals,
                        baseline=goals_median,
                        unit="goals",
                        impact="positive" if goals >= goals_median else "neutral",
                        explanation=reason_primary,
                    ),
                    self._reason_factor(
                        feature="xg_per90",
                        value=xg,
                        baseline=xg_median,
                        unit="xg",
                        impact="positive" if xg >= xg_median else "neutral",
                        explanation=reason_secondary,
                    ),
                ],
                "confidence": round(_rank_conf(float(row["scorer_rank"]), scorer_max), 2),
            }

        def _as_standout(row: pd.Series) -> dict[str, Any]:
            impact = float(row.get("impact_score", 0.0))
            form = float(row.get("form_points_last5", 0.0))
            reason_primary = "Strong all-phase match impact"
            if impact >= impact_median + 6.0:
                reason_primary = "Driving match tempo across phases"
            reason_secondary = "Reliable involvement across recent matches"
            if form >= 9.5:
                reason_secondary = "Recent form points are consistently high"
            return {
                "name": str(row.get("player", row.get("name", "Unavailable"))),
                "role": "standout",
                "team": str(row.get("team", team_b)),
                "reason": [
                    self._reason_factor(
                        feature="impact_score",
                        value=impact,
                        baseline=impact_median,
                        unit="index",
                        impact="positive" if impact >= impact_median else "neutral",
                        explanation=reason_primary,
                    ),
                    self._reason_factor(
                        feature="form_points_last5",
                        value=form,
                        baseline=8.5,
                        unit="points",
                        impact="positive" if form >= 8.5 else "neutral",
                        explanation=reason_secondary,
                    ),
                ],
                "confidence": round(_rank_conf(float(row["standout_rank"]), standout_max), 2),
            }

        scorer_candidates = [_as_scorer(row) for _, row in top_scorers.iterrows()]
        standout_candidates = [_as_standout(row) for _, row in top_standouts.iterrows()]

        def _ensure_min(items: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
            if len(items) >= 2:
                return items[:3]
            existing = {str(row.get("name", "")).strip().lower() for row in items}
            fallback = self._fallback_football_pool(team_a) + self._fallback_football_pool(team_b)
            for idx, name in enumerate(fallback):
                if len(items) >= 2:
                    break
                lowered = str(name).strip().lower()
                if not lowered or lowered in existing:
                    continue
                items.append(
                    {
                        "name": name,
                        "role": role,
                        "team": team_a if idx % 2 == 0 else team_b,
                        "reason": [
                            self._reason_factor(
                                feature=f"{role}_fallback",
                                value=1.0,
                                baseline=0.0,
                                unit="index",
                                impact="neutral",
                                explanation="Stable recent form signal",
                            )
                        ],
                        "confidence": 0.47,
                    }
                )
                existing.add(lowered)
            return items[:3]

        scorer_candidates = _ensure_min(scorer_candidates, "goal_scorer")
        standout_candidates = _ensure_min(standout_candidates, "standout")

        return {
            "best_player": scorer_candidates[0],
            "best_bowler": None,
            "man_of_the_match": standout_candidates[0],
            "players": {
                "top_goal_scorers": scorer_candidates[:3],
                "top_standout": standout_candidates[:3],
                "top_match_impact": standout_candidates[:3],
                "top_batsmen": [],
                "top_bowlers": [],
            },
        }

    def _predict_cricket(self, match_payload: dict[str, Any], k: int) -> dict[str, Any]:
        feature_dict, blend_meta = self._feature_dict_from_match(match_payload)
        residual_context = (
            match_payload.get("residual_context")
            if isinstance(match_payload.get("residual_context"), dict)
            else {}
        )
        anomaly = self._compute_anomaly_signal(feature_dict=feature_dict, residual_context=residual_context)
        team_a = str(match_payload.get("team_a", "Team A"))
        team_b = str(match_payload.get("team_b", "Team B"))
        venue = str(match_payload.get("venue", "Unknown Venue"))
        state = str(match_payload.get("state", "upcoming"))

        (
            team_a_first_samples,
            team_a_first_winner_probs,
            team_b_first_samples,
            team_b_first_winner_probs,
            model_source,
        ) = self._predict_conditional_raw_outcomes(feature_dict)
        team_a_first_samples = np.clip(team_a_first_samples, 90, 280)
        team_b_first_samples = np.clip(team_b_first_samples, 90, 280)
        team_a_first_winner_probs = np.clip(team_a_first_winner_probs, 0.02, 0.98)
        team_b_first_winner_probs = np.clip(team_b_first_winner_probs, 0.02, 0.98)

        combined_score_samples = (team_a_first_samples + team_b_first_samples) / 2.0
        combined_winner_probs = (team_a_first_winner_probs + team_b_first_winner_probs) / 2.0
        ensemble_uncertainty = self._ensemble_uncertainty(
            score_samples=combined_score_samples,
            winner_probs=combined_winner_probs,
        )
        (
            team_a_first_heads,
            team_a_first_head_winners,
            team_b_first_heads,
            team_b_first_head_winners,
            probs,
            confidences,
            total_projection,
        ) = self._distribution_conditional_scenarios(
            team_a_first_scores=team_a_first_samples,
            team_a_first_winner_probs=team_a_first_winner_probs,
            team_b_first_scores=team_b_first_samples,
            team_b_first_winner_probs=team_b_first_winner_probs,
            k=k,
        )
        labels = self._scenario_labels(k)

        scenario_predictions = []
        for idx, (
            a_scores,
            b_scores,
            a_winner_prob,
            b_winner_prob,
            confidence,
            probability,
            _scenario_total,
        ) in enumerate(
            zip(
                team_a_first_heads,
                team_b_first_heads,
                team_a_first_head_winners,
                team_b_first_head_winners,
                confidences,
                probs,
                total_projection,
            )
        ):
            team_a_score = float((a_scores[0] + b_scores[0]) / 2.0)
            team_b_score = float((a_scores[1] + b_scores[1]) / 2.0)
            scenario_score = float((team_a_score + team_b_score) / 2.0)
            scenario_label = labels[min(idx, len(labels) - 1)]
            reasons = self._normalize_reason_factors(
                self._cricket_scenario_reasons(
                    label=scenario_label,
                    score=scenario_score,
                    team_a=team_a,
                    team_b=team_b,
                    feature_dict=feature_dict,
                    anomaly=anomaly,
                )
            )
            unique_reasons: list[dict[str, Any]] = []
            seen_reason_explanations: set[str] = set()
            for reason in reasons:
                explanation = str(reason.get("explanation") or "").strip().lower()
                if not explanation:
                    continue
                if explanation in seen_reason_explanations:
                    continue
                seen_reason_explanations.add(explanation)
                unique_reasons.append(reason)
                if len(unique_reasons) >= 3:
                    break
            if not unique_reasons:
                fallback_reason = self._reason_factor(
                    feature="scenario_total",
                    value=scenario_score,
                    baseline=self.baseline_score,
                    unit="runs",
                    impact="neutral",
                    explanation=f"{scenario_label} scenario follows a balanced scoring script",
                )
                unique_reasons = [fallback_reason]
            team_a_first = self._branch_outcome(
                team_a=team_a,
                team_b=team_b,
                batting_team=team_a,
                bowling_team=team_b,
                batting_score=float(a_scores[0]),
                chase_score=float(a_scores[1]),
                winner_prob_team_a=float(a_winner_prob),
            )
            team_b_first = self._branch_outcome(
                team_a=team_a,
                team_b=team_b,
                batting_team=team_b,
                bowling_team=team_a,
                batting_score=float(b_scores[1]),
                chase_score=float(b_scores[0]),
                winner_prob_team_a=float(b_winner_prob),
            )
            winner_prob_center = float((a_winner_prob + b_winner_prob) / 2.0)
            scenario_winner = self._scenario_winner(
                team_a=team_a,
                team_b=team_b,
                team_a_score=team_a_score,
                team_b_score=team_b_score,
                winner_prob_center=winner_prob_center,
                team_a_first=team_a_first,
                team_b_first=team_b_first,
            )
            scenario_story = self._cricket_scenario_story(
                label=scenario_label,
                team_a_score=team_a_score,
                team_b_score=team_b_score,
            )
            scenario_predictions.append(
                {
                    "score": int(round(scenario_score)),
                    "scenario": scenario_label,
                    "label": scenario_label,
                    "team_a_score": int(round(team_a_score)),
                    "team_b_score": int(round(team_b_score)),
                    "winner": scenario_winner,
                    "story": scenario_story,
                    "team_a_first": team_a_first,
                    "team_b_first": team_b_first,
                    "reason": unique_reasons,
                    "scenario_probability": round(float(probability), 2),
                    "confidence": round(float(confidence), 2),
                }
            )

        player_predictions = self._predict_cricket_players(team_a=team_a, team_b=team_b)
        best_player = self._normalize_player_block(player_predictions["best_player"])
        best_bowler = self._normalize_player_block(player_predictions["best_bowler"])
        man_of_the_match = self._normalize_player_block(player_predictions["man_of_the_match"])
        players_payload = player_predictions.get("players") if isinstance(player_predictions.get("players"), dict) else {}

        sample_totals = (team_a_first_samples.mean(axis=1) + team_b_first_samples.mean(axis=1)) / 2.0
        interval_low, interval_high, calibration_meta = self._calibrated_interval_for_heads(sample_totals)
        if anomaly["odd_variant_flag"]:
            interval_low -= 4.0
            interval_high += 4.0
        match_insight = self._cricket_match_insight(
            team_a=team_a,
            team_b=team_b,
            feature_dict=feature_dict,
            scenarios=scenario_predictions,
            anomaly=anomaly,
        )

        return {
            "match": {
                "sport": "cricket",
                "tournament": str(match_payload.get("tournament", "IPL")),
                "team_a": team_a,
                "team_b": team_b,
                "venue": venue,
                "match_date": match_payload.get("match_date"),
                "state": state,
            },
            "predictions": scenario_predictions,
            "best_player": best_player,
            "best_bowler": best_bowler,
            "man_of_the_match": man_of_the_match,
            "players": players_payload,
            "match_insight": match_insight,
            "uncertainty": {
                "spread": round(float(np.max(total_projection) - np.min(total_projection)), 3),
                "interval_low": round(interval_low, 3),
                "interval_high": round(interval_high, 3),
                "mean_prediction": round(float(np.mean(total_projection)), 3),
                "std_prediction": round(float(np.std(total_projection)), 3),
            },
            "metadata": {
                "model_mode": "trained" if self.model_loaded else "fallback",
                "num_heads": int(len(combined_score_samples)),
                "feature_baseline": round(float(self.baseline_score), 2),
                "timemcl_style": {
                    "shared_encoder": True,
                    "winner_takes_all": True,
                    "diversity_regularization": True,
                    "multi_hypothesis": True,
                },
                "conditional_batting_order": True,
                "encoder_type": self.encoder_type,
                "patch_encoder": self.patch_encoder_config,
                "calibration": calibration_meta,
                "feature_blend": blend_meta,
                "forecast_engine": model_source,
                "ensemble_members": int(len(self.ensemble_members)),
                "ensemble_disagreement_score": ensemble_uncertainty["ensemble_disagreement_score"],
                "low_uncertainty_case": ensemble_uncertainty["low_uncertainty_case"],
                "ensemble_uncertainty": ensemble_uncertainty,
                "anomaly_score": anomaly["anomaly_score"],
                "odd_variant_flag": anomaly["odd_variant_flag"],
                "residual_shift_score": anomaly["residual_shift_score"],
                "anomaly_breakdown": anomaly["signal_breakdown"],
                "residual_context": residual_context,
                "scenario_probabilities": [round(float(prob), 2) for prob in probs.tolist()],
                "winner_probabilities_team_a": [
                    round(float((a + b) / 2.0), 2)
                    for a, b in zip(team_a_first_head_winners.tolist(), team_b_first_head_winners.tolist())
                ],
                "scenario_ranking": sorted(
                    [
                        {
                            "scenario": item["scenario"],
                            "score": item["score"],
                            "winner_if_team_a_bats_first": item["team_a_first"]["winner"],
                            "winner_if_team_b_bats_first": item["team_b_first"]["winner"],
                            "probability": item["scenario_probability"],
                        }
                        for item in scenario_predictions
                    ],
                    key=lambda row: row["probability"],
                    reverse=True,
                ),
            },
        }

    def _predict_football(self, match_payload: dict[str, Any], k: int) -> dict[str, Any]:
        team_a = str(match_payload.get("team_a", "Home Team"))
        team_b = str(match_payload.get("team_b", "Away Team"))
        venue = str(match_payload.get("venue", "Unknown Venue"))
        tournament = str(match_payload.get("tournament", "EPL"))
        state = str(match_payload.get("state", "upcoming"))

        home = self._football_team_profile(team_a)
        away = self._football_team_profile(team_b)

        expected_home = max(0.2, home["xg_for"] * 0.68 + away["xg_against"] * 0.35 + home["home_advantage"])
        expected_away = max(0.15, away["xg_for"] * 0.63 + home["xg_against"] * 0.33)
        expected_total = expected_home + expected_away
        anomaly_score = float(np.clip(abs(expected_total - 2.6) / 2.2, 0.0, 1.0))
        odd_variant_flag = anomaly_score >= 0.62

        totals_samples: list[float] = []
        diff_samples: list[float] = []
        history = self.football_matches.copy() if not self.football_matches.empty else pd.DataFrame()
        if not history.empty:
            history = history[history["tournament"].astype(str).str.lower() == tournament.lower()].copy()
            if history.empty:
                history = self.football_matches.copy()
            for col in ["home_goals", "away_goals"]:
                history[col] = pd.to_numeric(history[col], errors="coerce")

            pair_mask = (
                (history["team_a"].astype(str).str.lower() == team_a.lower())
                & (history["team_b"].astype(str).str.lower() == team_b.lower())
            ) | (
                (history["team_a"].astype(str).str.lower() == team_b.lower())
                & (history["team_b"].astype(str).str.lower() == team_a.lower())
            )
            pair_rows = history[pair_mask].copy()
            if not pair_rows.empty:
                for _, row in pair_rows.iterrows():
                    if str(row.get("team_a", "")).strip().lower() == team_a.lower():
                        goals_a = float(row.get("home_goals", 0.0))
                        goals_b = float(row.get("away_goals", 0.0))
                    else:
                        goals_a = float(row.get("away_goals", 0.0))
                        goals_b = float(row.get("home_goals", 0.0))
                    totals_samples.append(goals_a + goals_b)
                    diff_samples.append(goals_a - goals_b)
            if len(totals_samples) < max(5, k):
                totals_samples = (history["home_goals"] + history["away_goals"]).dropna().astype(float).tolist()
                diff_samples = (history["home_goals"] - history["away_goals"]).dropna().astype(float).tolist()

        if not totals_samples:
            totals_samples = [max(0.2, expected_total - 1.0), expected_total, min(5.4, expected_total + 1.1)]
        if not diff_samples:
            diff_samples = [expected_home - expected_away]

        quantiles = np.linspace(0.15, 0.85, max(2, min(7, k)))
        totals = np.quantile(np.asarray(totals_samples, dtype=np.float32), quantiles).astype(np.float32)
        totals = self._enforce_prediction_diversity(totals, min_gap=0.55)
        totals = np.clip(totals, 0.2, 5.4)
        diffs = np.quantile(np.asarray(diff_samples, dtype=np.float32), quantiles).astype(np.float32)
        diffs = np.clip(diffs, -3.8, 3.8)
        confidences = self._confidence_scores(totals)
        probs = self._softmax_scores(totals, temperature=1.25)

        scenario_labels = ["Low", "Baseline", "High", "Aggressive"]

        scenario_predictions = []
        for idx, (total_goals, diff_goals, confidence, prob) in enumerate(zip(totals, diffs, confidences, probs)):
            home_goals = int(max(0, round((float(total_goals) + float(diff_goals)) / 2.0)))
            away_goals = int(max(0, round(float(total_goals) - home_goals)))
            if home_goals + away_goals == 0 and float(total_goals) > 0.4:
                home_goals = 1

            if home_goals > away_goals:
                likely_result = f"{team_a} win"
            elif away_goals > home_goals:
                likely_result = f"{team_b} win"
            else:
                likely_result = "Draw"
            scenario_label = scenario_labels[min(idx, len(scenario_labels) - 1)]
            scenario_story = self._football_scenario_story(
                label=scenario_label,
                home_goals=float(home_goals),
                away_goals=float(away_goals),
            )
            if scenario_label == "Low":
                total_reason = "Defensive trends keep total goals in a lower band"
            elif scenario_label == "High":
                total_reason = "Attacking form supports a higher total-goals script"
            elif scenario_label == "Aggressive":
                total_reason = "Conversion volatility keeps this as the widest outcome band"
            else:
                total_reason = "Expected goals remain near the central outcome band"
            edge_reason = (
                f"{team_a} carries the stronger attacking profile"
                if expected_home >= expected_away
                else f"{team_b} carries the stronger attacking profile"
            )

            scenario_predictions.append(
                {
                    "scoreline": f"{home_goals}-{away_goals}",
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "scenario": scenario_label,
                    "story": scenario_story,
                    "likely_result": likely_result,
                    "reason": self._normalize_reason_factors(
                        [
                            {
                                "feature": "attack_edge",
                                "value": round(expected_home - expected_away, 3),
                                "baseline": 0.0,
                                "delta": round(expected_home - expected_away, 3),
                                "impact": "positive" if expected_home >= expected_away else "negative",
                                "unit": "xg",
                                "explanation": edge_reason,
                            },
                            {
                                "feature": "total_goals_band",
                                "value": round(float(total_goals), 3),
                                "baseline": round(float(expected_total), 3),
                                "delta": round(float(total_goals - expected_total), 3),
                                "impact": "positive" if total_goals >= expected_total else "neutral",
                                "unit": "goals",
                                "explanation": total_reason,
                            },
                            {
                                "feature": "anomaly_score",
                                "value": round(float(anomaly_score), 3),
                                "baseline": 0.5,
                                "delta": round(float(anomaly_score - 0.5), 3),
                                "impact": "negative" if odd_variant_flag else "neutral",
                                "unit": "ratio",
                                "explanation": (
                                    "Unusual match conditions increase variance"
                                    if odd_variant_flag
                                    else "No major anomaly signal in current context"
                                ),
                            },
                        ]
                    ),
                    "scenario_probability": round(float(prob), 2),
                    "confidence": round(float(np.clip((confidence + prob) / 2.0, 0.35, 0.92)), 2),
                }
            )

        players = self._predict_football_players(team_a=team_a, team_b=team_b)
        best_player = self._normalize_player_block(players["best_player"])
        man_of_the_match = self._normalize_player_block(players["man_of_the_match"])
        players_payload = players.get("players") if isinstance(players.get("players"), dict) else {}

        interval_low = float(np.percentile(totals, 10))
        interval_high = float(np.percentile(totals, 90))

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
            "predictions": scenario_predictions,
            "best_player": best_player,
            "best_bowler": None,
            "man_of_the_match": man_of_the_match,
            "players": players_payload,
            "match_insight": (
                f"{team_a} vs {team_b} projects near {expected_total:.1f} goals with "
                f"{'elevated' if odd_variant_flag else 'moderate'} variance."
            ),
            "uncertainty": {
                "spread": round(float(np.max(totals) - np.min(totals)), 3),
                "interval_low": round(interval_low, 3),
                "interval_high": round(interval_high, 3),
                "mean_prediction": round(float(np.mean(totals)), 3),
                "std_prediction": round(float(np.std(totals)), 3),
            },
            "metadata": {
                "model_mode": "fallback",
                "num_heads": int(len(totals)),
                "feature_baseline": round(float(expected_total), 3),
                "timemcl_style": {
                    "shared_encoder": True,
                    "winner_takes_all": True,
                    "diversity_regularization": True,
                    "multi_hypothesis": True,
                },
                "encoder_type": "football-rule-engine",
                "calibration": {"enabled": False, "method": "not_applicable"},
                "anomaly_score": round(anomaly_score, 3),
                "odd_variant_flag": odd_variant_flag,
                "residual_shift_score": 0.0,
                "scenario_probabilities": [round(float(prob), 2) for prob in probs],
                "scenario_ranking": sorted(
                    [
                        {
                            "scenario": row["scenario"],
                            "scoreline": row["scoreline"],
                            "probability": row["scenario_probability"],
                        }
                        for row in scenario_predictions
                    ],
                    key=lambda row: row["probability"],
                    reverse=True,
                ),
            },
        }

    def predict(self, match_payload: dict[str, Any], k: int = 3) -> dict[str, Any]:
        sport = str(match_payload.get("sport", "cricket")).lower()
        if sport == "football":
            return self._predict_football(match_payload=match_payload, k=k)
        return self._predict_cricket(match_payload=match_payload, k=k)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LiveMatch inference for a single fixture")
    parser.add_argument("--sport", default="cricket")
    parser.add_argument("--team-a", required=True)
    parser.add_argument("--team-b", required=True)
    parser.add_argument("--venue", required=True)
    parser.add_argument("--tournament", default="IPL")
    parser.add_argument("--state", default="upcoming")
    parser.add_argument("--match-id", default=None)
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    predictor = LiveMatchPredictor()
    result = predictor.predict(
        {
            "sport": args.sport,
            "match_id": args.match_id,
            "team_a": args.team_a,
            "team_b": args.team_b,
            "venue": args.venue,
            "tournament": args.tournament,
            "state": args.state,
        },
        k=args.k,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
