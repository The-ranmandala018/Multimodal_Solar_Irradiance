"""LSTM temporal encoder matching the Oshadha benchmark baseline.

Input:
    (B, 40, 7)

Output:
    (B, 128) temporal feature vector
"""

from __future__ import annotations

import torch
from torch import nn


class BaselineTemporalEncoder(nn.Module):
    """Single-layer LSTM encoder for the 40-step, 7-feature weather history."""

    def __init__(self, input_dim: int = 7, hidden_dim: int = 128):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a weather sequence.

        Args:
            x: Tensor with shape (B, 40, 7).

        Returns:
            Tensor with shape (B, 128).
        """
        if x.ndim != 3:
            raise ValueError(
                f"Expected a 3D tensor (B, T, F), got shape {tuple(x.shape)}"
            )

        if x.shape[-1] != 7:
            raise ValueError(
                f"Expected 7 weather features, got {x.shape[-1]}"
            )

        _, (hidden_state, _) = self.lstm(x)
        return hidden_state[-1]


__all__ = ["BaselineTemporalEncoder"]
