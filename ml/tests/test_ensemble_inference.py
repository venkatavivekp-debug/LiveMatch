from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.inference import EnsembleMember, LiveMatchPredictor


def _bare_predictor() -> LiveMatchPredictor:
    predictor = LiveMatchPredictor.__new__(LiveMatchPredictor)
    predictor.baseline_score = 170.0
    predictor.model_loaded = True
    predictor.feature_columns = ["f1", "f2", "f3"]
    predictor.feature_medians = {"f1": 0.0, "f2": 0.0, "f3": 0.0}
    predictor.num_heads = 2
    predictor.model = None
    predictor.scaler_mean = np.zeros(3, dtype=np.float32)
    predictor.scaler_std = np.ones(3, dtype=np.float32)
    predictor.ensemble_members = []
    predictor.lookup_df = pd.DataFrame()
    return predictor


def test_distribution_scenarios_are_probability_backed() -> None:
    predictor = _bare_predictor()
    score_samples = np.array(
        [
            [148.0, 154.0],
            [160.0, 162.0],
            [170.0, 168.0],
            [182.0, 176.0],
            [196.0, 184.0],
            [205.0, 192.0],
        ],
        dtype=np.float32,
    )
    winner_probs = np.array([0.39, 0.44, 0.49, 0.57, 0.63, 0.71], dtype=np.float32)

    scenario_scores, scenario_winners, scenario_probs, confidences, scenario_totals = predictor._distribution_scenarios(
        score_samples=score_samples,
        winner_probs=winner_probs,
        k=4,
    )

    assert scenario_scores.shape == (4, 2)
    assert scenario_winners.shape == (4,)
    assert scenario_probs.shape == (4,)
    assert confidences.shape == (4,)
    assert np.all(np.diff(scenario_totals) >= -1e-6)
    assert float(np.sum(scenario_probs)) == pytest.approx(1.0, rel=1e-6, abs=1e-6)
    assert float(np.min(scenario_winners)) < 0.5
    assert float(np.max(scenario_winners)) > 0.5


def test_branch_feature_vector_flips_batting_order_flags() -> None:
    predictor = _bare_predictor()
    predictor.feature_columns = [
        "batting_first",
        "team_a_bats_first",
        "team_b_bats_first",
        "team_a_chase_success_rate",
        "team_b_chase_success_rate",
        "team_a_defend_success_rate",
        "team_b_defend_success_rate",
        "chase_defend_edge_team_a_first",
        "chase_defend_edge_team_b_first",
    ]
    predictor.feature_medians = {name: 0.0 for name in predictor.feature_columns}
    feature_dict = {
        "batting_first": 1.0,
        "team_a_chase_success_rate": 0.55,
        "team_b_chase_success_rate": 0.48,
        "team_a_defend_success_rate": 0.6,
        "team_b_defend_success_rate": 0.52,
    }
    a_vec = predictor._branch_feature_vector(feature_dict, team_a_bats_first=True)
    b_vec = predictor._branch_feature_vector(feature_dict, team_a_bats_first=False)
    assert float(a_vec[0]) == 1.0
    assert float(a_vec[1]) == 1.0
    assert float(a_vec[2]) == 0.0
    assert float(b_vec[0]) == 0.0
    assert float(b_vec[1]) == 0.0
    assert float(b_vec[2]) == 1.0


def test_ensemble_uncertainty_marks_low_uncertainty_case() -> None:
    predictor = _bare_predictor()
    score_samples = np.array(
        [
            [170.0, 169.0],
            [170.2, 169.1],
            [169.9, 169.0],
            [170.1, 169.2],
        ],
        dtype=np.float32,
    )
    winner_probs = np.array([0.51, 0.5, 0.52, 0.5], dtype=np.float32)
    summary = predictor._ensemble_uncertainty(score_samples, winner_probs)
    assert summary["low_uncertainty_case"] is True
    assert float(summary["ensemble_disagreement_score"]) < 0.2


def test_ensemble_uncertainty_detects_meaningful_variation() -> None:
    predictor = _bare_predictor()
    score_samples = np.array(
        [
            [148.0, 141.0],
            [164.0, 158.0],
            [176.0, 171.0],
            [194.0, 186.0],
        ],
        dtype=np.float32,
    )
    winner_probs = np.array([0.34, 0.47, 0.56, 0.69], dtype=np.float32)
    summary = predictor._ensemble_uncertainty(score_samples, winner_probs)
    assert summary["low_uncertainty_case"] is False
    assert float(summary["ensemble_disagreement_score"]) > 0.2


def test_distribution_scenario_weights_reflect_sample_density() -> None:
    predictor = _bare_predictor()
    score_samples = np.array(
        [
            [150.0, 148.0],
            [152.0, 150.0],
            [153.0, 151.0],
            [155.0, 152.0],
            [182.0, 176.0],
            [184.0, 177.0],
        ],
        dtype=np.float32,
    )
    winner_probs = np.array([0.42, 0.43, 0.45, 0.46, 0.63, 0.64], dtype=np.float32)

    _, _, scenario_probs, _, scenario_totals = predictor._distribution_scenarios(
        score_samples=score_samples,
        winner_probs=winner_probs,
        k=4,
    )

    assert float(np.sum(scenario_probs)) == pytest.approx(1.0, rel=1e-6, abs=1e-6)
    assert float(np.max(scenario_totals) - np.min(scenario_totals)) > 8.0


def test_branch_outcome_winner_uses_branch_scores() -> None:
    predictor = _bare_predictor()
    team_a_first = predictor._branch_outcome(
        team_a="Team A",
        team_b="Team B",
        batting_team="Team A",
        bowling_team="Team B",
        batting_score=182.0,
        chase_score=171.0,
        winner_prob_team_a=0.35,
    )
    team_b_first = predictor._branch_outcome(
        team_a="Team A",
        team_b="Team B",
        batting_team="Team B",
        bowling_team="Team A",
        batting_score=171.0,
        chase_score=182.0,
        winner_prob_team_a=0.35,
    )

    assert team_a_first["winner"] == "Team A"
    assert team_b_first["winner"] == "Team A"


def test_scenario_winner_uses_branch_context_when_branches_disagree() -> None:
    predictor = _bare_predictor()
    winner = predictor._scenario_winner(
        team_a="Team A",
        team_b="Team B",
        team_a_score=171.0,
        team_b_score=169.0,
        winner_prob_center=0.44,
        team_a_first={"winner": "Team A"},
        team_b_first={"winner": "Team B"},
    )
    assert winner == "Team B"


def test_scenario_winner_prefers_consistent_branch_winner() -> None:
    predictor = _bare_predictor()
    winner = predictor._scenario_winner(
        team_a="Team A",
        team_b="Team B",
        team_a_score=168.0,
        team_b_score=172.0,
        winner_prob_center=0.56,
        team_a_first={"winner": "Team A"},
        team_b_first={"winner": "Team A"},
    )
    assert winner == "Team A"


def test_conditional_scenarios_can_produce_different_branch_winners() -> None:
    predictor = _bare_predictor()
    a_first_scores = np.array(
        [
            [184.0, 170.0],
            [172.0, 169.0],
            [165.0, 171.0],
            [190.0, 176.0],
        ],
        dtype=np.float32,
    )
    b_first_scores = np.array(
        [
            [166.0, 179.0],
            [174.0, 170.0],
            [170.0, 162.0],
            [181.0, 193.0],
        ],
        dtype=np.float32,
    )
    a_probs = np.array([0.61, 0.55, 0.46, 0.68], dtype=np.float32)
    b_probs = np.array([0.39, 0.52, 0.58, 0.44], dtype=np.float32)

    (
        a_heads,
        a_winners,
        b_heads,
        b_winners,
        _probs,
        _conf,
        _totals,
    ) = predictor._distribution_conditional_scenarios(
        team_a_first_scores=a_first_scores,
        team_a_first_winner_probs=a_probs,
        team_b_first_scores=b_first_scores,
        team_b_first_winner_probs=b_probs,
        k=4,
    )

    has_branch_winner_difference = False
    for idx in range(a_heads.shape[0]):
        a_branch = predictor._branch_outcome(
            team_a="Team A",
            team_b="Team B",
            batting_team="Team A",
            bowling_team="Team B",
            batting_score=float(a_heads[idx][0]),
            chase_score=float(a_heads[idx][1]),
            winner_prob_team_a=float(a_winners[idx]),
        )
        b_branch = predictor._branch_outcome(
            team_a="Team A",
            team_b="Team B",
            batting_team="Team B",
            bowling_team="Team A",
            batting_score=float(b_heads[idx][1]),
            chase_score=float(b_heads[idx][0]),
            winner_prob_team_a=float(b_winners[idx]),
        )
        if a_branch["winner"] != b_branch["winner"]:
            has_branch_winner_difference = True
            break
    assert has_branch_winner_difference


def test_distribution_scenarios_have_separation_when_variance_exists() -> None:
    predictor = _bare_predictor()
    score_samples = np.array(
        [
            [145.0, 149.0],
            [153.0, 157.0],
            [162.0, 165.0],
            [175.0, 171.0],
            [188.0, 182.0],
            [203.0, 195.0],
        ],
        dtype=np.float32,
    )
    winner_probs = np.array([0.38, 0.41, 0.46, 0.54, 0.61, 0.69], dtype=np.float32)
    _, _, _, _, totals = predictor._distribution_scenarios(score_samples, winner_probs, k=4)
    assert float(np.max(totals) - np.min(totals)) >= 8.0


def test_distribution_scenarios_avoid_even_spacing_when_samples_support_shape() -> None:
    predictor = _bare_predictor()
    score_samples = np.array(
        [
            [150.0, 150.0],
            [160.0, 160.0],
            [170.0, 170.0],
            [180.0, 180.0],
            [190.0, 190.0],
            [200.0, 200.0],
            [210.0, 210.0],
        ],
        dtype=np.float32,
    )
    winner_probs = np.array([0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65], dtype=np.float32)
    _, _, _, _, totals = predictor._distribution_scenarios(score_samples, winner_probs, k=4)
    gaps = np.diff(totals)
    assert gaps.shape[0] == 3
    assert float(np.std(gaps)) > 0.0


def test_predict_raw_outcomes_uses_available_ensemble_members() -> None:
    torch = pytest.importorskip("torch")
    from ml.model import TimeMCLModel

    predictor = _bare_predictor()

    class BrokenModel:
        def __call__(self, *_args, **_kwargs):  # noqa: ANN001, ANN002
            raise RuntimeError("broken member")

    valid_model = TimeMCLModel(
        input_dim=3,
        num_heads=2,
        hidden_dims=(16, 8),
        dropout=0.0,
    )
    valid_model.eval()

    predictor.ensemble_members = [
        EnsembleMember(
            member_id=0,
            model=BrokenModel(),  # type: ignore[arg-type]
            feature_columns=["f1", "f2", "f3"],
            scaler_mean=np.zeros(3, dtype=np.float32),
            scaler_std=np.ones(3, dtype=np.float32),
            num_heads=2,
        ),
        EnsembleMember(
            member_id=1,
            model=valid_model,
            feature_columns=["f1", "f2", "f3"],
            scaler_mean=np.zeros(3, dtype=np.float32),
            scaler_std=np.ones(3, dtype=np.float32),
            num_heads=2,
        ),
    ]

    feature_vector = np.array([1.0, -0.3, 0.8], dtype=np.float32)
    score_samples, winner_probs, source = predictor._predict_raw_outcomes(feature_vector)

    assert source == "ensemble"
    assert score_samples.shape == (2, 2)
    assert winner_probs.shape == (2,)
    assert torch.isfinite(torch.tensor(score_samples)).all().item()


def test_predict_raw_outcomes_fallback_without_model() -> None:
    predictor = _bare_predictor()
    predictor.model_loaded = False
    predictor.scaler_mean = None
    predictor.scaler_std = None

    score_samples, winner_probs, source = predictor._predict_raw_outcomes(np.array([0.0, 0.0, 0.0], dtype=np.float32))

    assert source == "fallback"
    assert score_samples.shape[1] == 2
    assert winner_probs.ndim == 1
    assert score_samples.shape[0] == winner_probs.shape[0]
