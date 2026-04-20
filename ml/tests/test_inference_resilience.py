from __future__ import annotations

from types import SimpleNamespace

from ml import inference


def test_checkpoint_load_failure_drops_to_fallback(tmp_path, monkeypatch) -> None:
    checkpoint_path = tmp_path / "time_mcl.pt"
    checkpoint_path.write_bytes(b"not-a-valid-checkpoint")

    predictor = inference.LiveMatchPredictor.__new__(inference.LiveMatchPredictor)
    predictor.config = SimpleNamespace(
        hidden_dims=(128, 64),
        dropout=0.2,
        encoder_type="mlp",
        patch_length=4,
        patch_stride=2,
        patch_model_dim=64,
        patch_layers=2,
        patch_attention_heads=2,
    )
    predictor.checkpoint_path = checkpoint_path
    predictor.feature_columns = ["f1", "f2"]
    predictor.num_heads = 3
    predictor.model = "stale-model"
    predictor.model_loaded = True
    predictor.runtime_mode = "TRAINED MODEL"
    predictor.conformal_calibration = {"enabled": True}

    class _BrokenTorch:
        @staticmethod
        def load(*args, **kwargs):  # noqa: ANN002, ANN003
            raise RuntimeError("corrupted checkpoint")

    monkeypatch.setattr(inference, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(inference, "torch", _BrokenTorch())

    inference.LiveMatchPredictor._load_model_checkpoint(predictor)

    assert predictor.model is None
    assert predictor.model_loaded is False
    assert predictor.runtime_mode == "FALLBACK"
    assert predictor.conformal_calibration is None
