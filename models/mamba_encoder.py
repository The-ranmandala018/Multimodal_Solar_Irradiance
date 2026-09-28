"""Mamba temporal encoder for the Folsom weather-history branch.

Input:
    (B, 40, 7)

Output:
    (B, 128)

The 40-step / 7-feature input follows the weather representation used by
the Oshadha benchmark loader. The temporal encoder replaces its LSTM
baseline with a Mamba-style sequence model.
"""

from __future__ import annotations

import torch
from torch import nn


class MambaTemporalEncoder(nn.Module):
    """Encode the 40x7 weather sequence into a 128-D temporal feature."""

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)

        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)

        # Keep the sequence-model interface independent of the exact Mamba
        # package so the rest of the multimodal architecture stays stable.
        try:
            from mamba_ssm import Mamba

            self.layers = nn.ModuleList(
                [
                    Mamba(
                        d_model=self.hidden_dim,
                        d_state=16,
                        d_conv=4,
                        expand=2,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.backend = "mamba_ssm"
        except ImportError:
            # Fallback keeps the module importable on machines where
            # mamba-ssm is not installed yet. This can be replaced by the
            # real Mamba backend without changing the model interface.
            self.layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.hidden_dim, self.hidden_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    )
                    for _ in range(num_layers)
                ]
            )
            self.backend = "fallback"

        self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, weather_sequence: torch.Tensor) -> torch.Tensor:
        """Encode weather history.

        Args:
            weather_sequence: Tensor of shape (B, 40, 7).

        Returns:
            Tensor of shape (B, 128).
        """
        if weather_sequence.ndim != 3:
            raise ValueError(
                "Expected weather_sequence with shape (B, T, C), "
                f"got {tuple(weather_sequence.shape)}"
            )

        if weather_sequence.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} weather features, "
                f"got {weather_sequence.shape[-1]}"
            )

        x = self.input_projection(weather_sequence)

        for layer in self.layers:
            residual = x
            x = layer(x)
            x = self.norm(x + residual)

        # Use the final time step as the sequence representation.
        return x[:, -1, :]


__all__ = ["MambaTemporalEncoder"]
