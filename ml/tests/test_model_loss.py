from __future__ import annotations

import pytest
torch = pytest.importorskip("torch")

from ml.model import wta_diverse_loss
from ml.model import TimeMCLModel


def test_wta_diverse_loss_hard_and_soft() -> None:
    preds = torch.tensor([[150.0, 165.0, 190.0], [140.0, 170.0, 200.0]], dtype=torch.float32)
    target = torch.tensor([160.0, 168.0], dtype=torch.float32)

    hard = wta_diverse_loss(predictions=preds, target=target, diversity_margin=8.0, diversity_weight=0.1)
    soft = wta_diverse_loss(
        predictions=preds,
        target=target,
        diversity_margin=8.0,
        diversity_weight=0.1,
        soft_wta_temperature=2.0,
    )

    assert hard.loss.item() >= 0.0
    assert soft.loss.item() >= 0.0
    assert hard.winner_indices.shape[0] == target.shape[0]
    assert soft.winner_soft_entropy.item() >= 0.0


@pytest.mark.parametrize("temperature", [0.0, 0.5, 2.5])
def test_soft_temperature_is_stable(temperature: float) -> None:
    preds = torch.randn(4, 3)
    target = torch.randn(4)
    output = wta_diverse_loss(predictions=preds, target=target, soft_wta_temperature=temperature)
    assert torch.isfinite(output.loss)


def test_patch_encoder_mode_forward_shape() -> None:
    model = TimeMCLModel(
        input_dim=13,
        num_heads=4,
        hidden_dims=(64, 32),
        encoder_type="patch",
        patch_length=4,
        patch_stride=2,
        patch_model_dim=32,
        patch_layers=1,
        patch_attention_heads=4,
    )
    x = torch.randn(5, 13)
    score_heads, winner_logits = model(x)
    assert score_heads.shape == (5, 4, 2)
    assert winner_logits.shape == (5, 4)
