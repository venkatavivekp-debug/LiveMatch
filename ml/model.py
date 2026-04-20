from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


EncoderType = Literal["mlp", "patch"]


class PatchSequenceEncoder(nn.Module):
    """
    PatchTST-style encoder over feature vectors.

    This is intentionally scoped to encoder representation only:
    TimeMCL-style K-head forecasting remains unchanged.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        patch_length: int = 4,
        patch_stride: int = 2,
        model_dim: int = 64,
        num_layers: int = 2,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = max(1, int(input_dim))
        self.patch_length = max(2, int(patch_length))
        self.patch_stride = max(1, int(patch_stride))
        self.model_dim = max(8, int(model_dim))

        max_sequence = max(self.input_dim, self.patch_length)
        self.max_patches = 1 + ((max_sequence - self.patch_length) // self.patch_stride)
        self.max_patches = max(1, self.max_patches)

        attention_heads = max(1, int(num_attention_heads))
        while self.model_dim % attention_heads != 0 and attention_heads > 1:
            attention_heads -= 1

        self.patch_projection = nn.Linear(self.patch_length, self.model_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.max_patches, self.model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=attention_heads,
            dim_feedforward=self.model_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=max(1, int(num_layers)))
        self.norm = nn.LayerNorm(self.model_dim)
        self.output_projection = nn.Sequential(
            nn.Linear(self.model_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def _to_patches(self, features: torch.Tensor) -> torch.Tensor:
        feature_length = features.shape[1]
        if feature_length < self.patch_length:
            pad_width = self.patch_length - feature_length
            features = nn.functional.pad(features, pad=(0, pad_width), mode="constant", value=0.0)

        patches = features.unfold(dimension=1, size=self.patch_length, step=self.patch_stride)
        if patches.size(1) > self.max_patches:
            patches = patches[:, : self.max_patches, :]
        return patches

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        patches = self._to_patches(features)
        tokens = self.patch_projection(patches)
        tokens = tokens + self.position_embedding[:, : tokens.size(1), :]
        encoded = self.transformer(tokens)
        pooled = self.norm(encoded.mean(dim=1))
        return self.output_projection(pooled)


class TimeMCLModel(nn.Module):
    """Shared encoder + K hypothesis heads for full match outcomes."""

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        hidden_dims: tuple[int, int] = (128, 64),
        dropout: float = 0.1,
        encoder_type: EncoderType = "mlp",
        patch_length: int = 4,
        patch_stride: int = 2,
        patch_model_dim: int = 64,
        patch_layers: int = 2,
        patch_attention_heads: int = 4,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.encoder_type = encoder_type
        head_input_dim = hidden_dims[1]

        if encoder_type == "patch":
            self.encoder = PatchSequenceEncoder(
                input_dim=input_dim,
                output_dim=head_input_dim,
                patch_length=patch_length,
                patch_stride=patch_stride,
                model_dim=patch_model_dim,
                num_layers=patch_layers,
                num_attention_heads=patch_attention_heads,
                dropout=dropout,
            )
        else:
            layers: list[nn.Module] = [
                nn.Linear(input_dim, hidden_dims[0]),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            self.encoder = nn.Sequential(*layers)

        self.score_heads = nn.ModuleList([nn.Linear(head_input_dim, 2) for _ in range(num_heads)])
        self.winner_heads = nn.ModuleList([nn.Linear(head_input_dim, 1) for _ in range(num_heads)])

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(features)
        score_outputs = torch.stack([head(encoded) for head in self.score_heads], dim=1)
        winner_logits = torch.cat([head(encoded) for head in self.winner_heads], dim=1)
        return score_outputs, winner_logits


@dataclass
class WTALossOutput:
    loss: torch.Tensor
    primary_loss: torch.Tensor
    winner_loss: torch.Tensor
    diversity_penalty: torch.Tensor
    winner_indices: torch.Tensor
    winner_soft_entropy: torch.Tensor


def wta_diverse_loss(
    predictions: torch.Tensor,
    target: torch.Tensor,
    winner_logits: torch.Tensor | None = None,
    winner_target: torch.Tensor | None = None,
    diversity_margin: float = 10.0,
    diversity_weight: float = 0.05,
    winner_loss_weight: float = 0.35,
    soft_wta_temperature: float = 0.0,
) -> WTALossOutput:
    """
    Winner-takes-all objective with light diversity pressure.

    - Hard WTA: only the closest head contributes to supervised loss.
    - Soft WTA (temperature > 0): weighted supervision over heads where lower
      error heads receive larger weights.
    - Pairwise head predictions are encouraged to stay separated by margin.
    """
    if predictions.ndim == 2:
        predictions = predictions.unsqueeze(2)
    if target.ndim == 1:
        target = target.unsqueeze(1)

    expanded_target = target.unsqueeze(1)
    absolute_errors = torch.mean(torch.abs(predictions - expanded_target), dim=2)
    winner_indices = torch.argmin(absolute_errors, dim=1)
    squared_errors = torch.mean((predictions - expanded_target) ** 2, dim=2)

    if soft_wta_temperature > 0:
        soft_weights = torch.softmax(-absolute_errors / soft_wta_temperature, dim=1)
        primary_loss = torch.mean(torch.sum(soft_weights * squared_errors, dim=1))
        winner_soft_entropy = -torch.mean(
            torch.sum(soft_weights * torch.log(soft_weights.clamp_min(1e-8)), dim=1)
        )
    else:
        batch_indices = torch.arange(predictions.size(0), device=predictions.device)
        winning_predictions = predictions[batch_indices, winner_indices, :]
        primary_loss = torch.mean((winning_predictions - target) ** 2)
        winner_soft_entropy = torch.zeros(1, device=predictions.device).squeeze(0)

    if winner_logits is not None and winner_target is not None:
        if winner_target.ndim > 1:
            winner_target = winner_target.squeeze(1)
        winner_target = winner_target.float()
        expanded_winner_target = winner_target.unsqueeze(1).expand_as(winner_logits)
        bce_per_head = nn.functional.binary_cross_entropy_with_logits(
            winner_logits,
            expanded_winner_target,
            reduction="none",
        )
        if soft_wta_temperature > 0:
            winner_loss = torch.mean(torch.sum(soft_weights * bce_per_head, dim=1))
        else:
            selected_logits = winner_logits.gather(1, winner_indices.unsqueeze(1)).squeeze(1)
            winner_loss = nn.functional.binary_cross_entropy_with_logits(selected_logits, winner_target)
    else:
        winner_loss = torch.zeros(1, device=predictions.device).squeeze(0)

    pairwise_penalties: list[torch.Tensor] = []
    num_heads = predictions.shape[1]
    for i in range(num_heads):
        for j in range(i + 1, num_heads):
            score_gap = torch.linalg.norm(predictions[:, i, :] - predictions[:, j, :], dim=1)
            if winner_logits is not None:
                prob_gap = torch.abs(
                    torch.sigmoid(winner_logits[:, i]) - torch.sigmoid(winner_logits[:, j])
                )
                head_gap = score_gap + (2.5 * prob_gap)
            else:
                head_gap = score_gap
            pairwise_penalties.append(torch.relu(diversity_margin - head_gap))

    if pairwise_penalties:
        diversity_penalty = torch.stack(pairwise_penalties, dim=1).mean()
    else:
        diversity_penalty = torch.zeros(1, device=predictions.device).squeeze(0)

    total_loss = (
        primary_loss
        + (winner_loss_weight * winner_loss)
        + (diversity_weight * diversity_penalty)
    )
    return WTALossOutput(
        loss=total_loss,
        primary_loss=primary_loss,
        winner_loss=winner_loss,
        diversity_penalty=diversity_penalty,
        winner_indices=winner_indices,
        winner_soft_entropy=winner_soft_entropy,
    )
