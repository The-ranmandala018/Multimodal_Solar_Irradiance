"""Fusion module for the multimodal solar irradiance model.

The visual CNN produces a 256-D feature vector and the temporal LSTM
produces a 128-D feature vector. This module concatenates them into a
384-D multimodal representation.
"""

from __future__ import annotations

import torch
from torch import nn


class ConcatenationFusion(nn.Module):
    """Concatenate visual and temporal feature vectors."""

    def __init__(self, visual_dim: int = 256, temporal_dim: int = 128):
        super().__init__()
        self.output_dim = visual_dim + temporal_dim

    def forward(
        self,
        visual_features: torch.Tensor,
        temporal_features: torch.Tensor,
    ) -> torch.Tensor:
        if visual_features.ndim != 2:
            raise ValueError(
                f"Expected visual features with shape (B, D), got {tuple(visual_features.shape)}"
            )

        if temporal_features.ndim != 2:
            raise ValueError(
                f"Expected temporal features with shape (B, D), got {tuple(temporal_features.shape)}"
            )

        if visual_features.shape[0] != temporal_features.shape[0]:
            raise ValueError("Visual and temporal batch sizes must match.")

        if visual_features.shape[1] != 256:
            raise ValueError(
                f"Expected 256-D visual features, got {visual_features.shape[1]}"
            )

        if temporal_features.shape[1] != 128:
            raise ValueError(
                f"Expected 128-D temporal features, got {temporal_features.shape[1]}"
            )

        return torch.cat((visual_features, temporal_features), dim=1)


__all__ = ["ConcatenationFusion"]
