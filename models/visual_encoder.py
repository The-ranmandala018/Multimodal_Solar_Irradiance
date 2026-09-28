"""CNN visual encoder for the Folsom 10-channel 3D projection.

Input:
    (B, 10, 180, 720)

Output:
    (B, 256) visual feature vector

The encoder treats the 10 Folsom projection features as channels and
preserves their spatial arrangement through convolutional blocks.
"""

from __future__ import annotations

import torch
from torch import nn


class FolsomCNNEncoder(nn.Module):
    """Small CNN baseline for the Folsom 3D projection representation."""

    def __init__(self, in_channels: int = 10, feature_dim: int = 256):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(256, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of Folsom projection tensors.

        Args:
            x: Tensor with shape (B, 10, 180, 720).

        Returns:
            Tensor with shape (B, feature_dim).
        """
        if x.ndim != 4:
            raise ValueError(
                f"Expected a 4D tensor (B, C, H, W), got shape {tuple(x.shape)}"
            )

        if x.shape[1] != 10:
            raise ValueError(
                f"Expected 10 Folsom projection channels, got {x.shape[1]}"
            )

        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        return self.projection(x)


__all__ = ["FolsomCNNEncoder"]
