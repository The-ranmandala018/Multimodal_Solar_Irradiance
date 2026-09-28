"""Complete multimodal solar irradiance forecasting model.

Architecture:
    Folsom 3D projection -> CNN -> 256-D
    Weather history (40 x 7) -> LSTM -> 128-D
    256-D + 128-D -> Fusion -> 384-D
    384-D -> Regression Head -> forecast
"""

from __future__ import annotations

import torch
from torch import nn

from .visual_encoder import FolsomCNNEncoder
from .temporal import BaselineTemporalEncoder
from .fusion import ConcatenationFusion
from .heads import BaselineRegressionHead


class MultimodalSolarForecastModel(nn.Module):
    """CNN + LSTM + concatenation fusion + regression head."""

    def __init__(
        self,
        visual_dim: int = 256,
        temporal_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.visual_encoder = FolsomCNNEncoder(
            in_channels=10,
            feature_dim=visual_dim,
        )

        self.temporal_encoder = BaselineTemporalEncoder(
            input_dim=7,
            hidden_dim=temporal_dim,
        )

        self.fusion = ConcatenationFusion(
            visual_dim=visual_dim,
            temporal_dim=temporal_dim,
        )

        self.forecasting_head = BaselineRegressionHead(
            input_dim=visual_dim + temporal_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
        )

    def forward(
        self,
        visual_input: torch.Tensor,
        temporal_input: torch.Tensor,
    ) -> torch.Tensor:
        """Run the complete multimodal forecasting pipeline.

        Args:
            visual_input:
                Folsom projection tensor with shape (B, 10, 180, 720).

            temporal_input:
                Historical weather/solar features with shape (B, 40, 7).

        Returns:
            Forecast tensor with shape (B, output_dim).
        """
        visual_features = self.visual_encoder(
            visual_input
        )

        temporal_features = self.temporal_encoder(
            temporal_input
        )

        fused_features = self.fusion(
            visual_features,
            temporal_features,
        )

        forecast = self.forecasting_head(
            fused_features
        )

        return forecast


__all__ = ["MultimodalSolarForecastModel"]
