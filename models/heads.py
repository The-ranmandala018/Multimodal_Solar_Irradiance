"""Regression head for multimodal solar irradiance forecasting.

Input:
    (B, 384) fused visual + temporal representation

Output:
    (B, output_dim) forecast
"""

from __future__ import annotations

import torch
from torch import nn


class BaselineRegressionHead(nn.Module):
    """Oshadha-style regression head adapted to the 384-D fused features."""

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 256,
        output_dim: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map fused multimodal features to the forecast."""
        if x.ndim != 2:
            raise ValueError(
                f"Expected a 2D tensor (B, D), got shape {tuple(x.shape)}"
            )

        if x.shape[1] != 384:
            raise ValueError(
                f"Expected 384-D fused features, got {x.shape[1]}"
            )

        return self.network(x)


__all__ = ["BaselineRegressionHead"]
