"""Mamba + LSTM multimodal solar irradiance forecasting model.

Architecture:
    Folsom 3D projection (10 x 180 x 720) -> VMamba encoder -> 256-D
    Weather history (40 x 7) -> LSTM -> 128-D
    256-D + 128-D -> Fusion -> 384-D
    384-D -> Regression Head -> forecast

Identical to MultimodalSolarForecastModel except for the visual encoder: the
LSTM, fusion and regression head are the same classes with the same settings.
This model is deliberately NOT registered in models.multimodal_model.MODEL_TYPES,
so the existing CNN training/evaluation scripts are unaffected. It is built by
scripts/train_mamba_lstm.py and scripts/evaluate_mamba_lstm.py.
"""

from __future__ import annotations

import torch
from torch import nn

from .fusion import ConcatenationFusion
from .heads import BaselineRegressionHead
from .mamba_encoder import FolsomVMambaEncoder
from .temporal import BaselineTemporalEncoder


class MambaLSTMForecastModel(nn.Module):
    """VMamba + LSTM + concatenation fusion + regression head."""

    def __init__(
        self,
        visual_dim: int = 256,
        temporal_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 1,
        dropout: float = 0.3,
        scale: str = "tiny",
        pretrained: bool = True,
        weight_path: str | None = None,
    ):
        super().__init__()

        self.visual_encoder = FolsomVMambaEncoder(
            in_channels=10,
            feature_dim=visual_dim,
            scale=scale,
            pretrained=pretrained,
            weight_path=weight_path,
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
            visual_input: Folsom projection tensor with shape (B, 10, 180, 720).
            temporal_input: Weather/solar history with shape (B, 40, 7).

        Returns:
            Forecast tensor with shape (B, output_dim).
        """
        visual_features = self.visual_encoder(visual_input)
        temporal_features = self.temporal_encoder(temporal_input)
        fused_features = self.fusion(visual_features, temporal_features)
        return self.forecasting_head(fused_features)


def build_mamba_lstm(model_cfg: dict, pretrained: bool | None = None) -> MambaLSTMForecastModel:
    """Build the model from the `model:` section of experiments/mamba_lstm/config.yaml.

    `pretrained` overrides the config; evaluation passes False because the
    checkpoint replaces every weight anyway.
    """
    return MambaLSTMForecastModel(
        visual_dim=int(model_cfg.get("visual_dim", 256)),
        temporal_dim=int(model_cfg.get("temporal_dim", 128)),
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        dropout=float(model_cfg.get("dropout", 0.3)),
        scale=str(model_cfg.get("scale", "tiny")),
        pretrained=bool(model_cfg.get("pretrained", True)) if pretrained is None else pretrained,
        weight_path=model_cfg.get("weight_path"),
    )


__all__ = ["MambaLSTMForecastModel", "build_mamba_lstm"]
