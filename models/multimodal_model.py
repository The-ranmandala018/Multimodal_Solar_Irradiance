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


def build_visual_encoder(name: str, visual_dim: int = 256) -> nn.Module:
    """Build the image encoder: "cnn" (default) or "vim_tiny"."""
    if name == "cnn":
        return FolsomCNNEncoder(in_channels=10, feature_dim=visual_dim)
    if name == "vim_tiny":
        # Imported here so CNN models do not need mamba-ssm installed.
        from .visual_encoder_vim import FolsomVimTinyEncoder

        return FolsomVimTinyEncoder(in_channels=10, feature_dim=visual_dim)
    raise ValueError(f"Unknown visual encoder: {name!r}")


class MultimodalSolarForecastModel(nn.Module):
    """Image encoder (CNN by default) + LSTM + fusion + regression head."""

    def __init__(
        self,
        visual_dim: int = 256,
        temporal_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 1,
        dropout: float = 0.3,
        visual_encoder: str = "cnn",
    ):
        super().__init__()

        self.visual_encoder = build_visual_encoder(visual_encoder, visual_dim)

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


class WeatherOnlyForecastModel(nn.Module):
    """Weather-only baseline: the same LSTM and head, no sky image.

    Takes the same (visual_input, temporal_input) arguments as the multimodal
    model so the training and evaluation loops are unchanged; visual_input is
    ignored.
    """

    def __init__(
        self,
        temporal_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.temporal_encoder = BaselineTemporalEncoder(
            input_dim=7,
            hidden_dim=temporal_dim,
        )

        self.forecasting_head = BaselineRegressionHead(
            input_dim=temporal_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
        )

    def forward(
        self,
        visual_input: torch.Tensor,
        temporal_input: torch.Tensor,
    ) -> torch.Tensor:
        return self.forecasting_head(self.temporal_encoder(temporal_input))


class ImageOnlyForecastModel(nn.Module):
    """Image-only baseline: the same CNN encoder and head, no weather history.

    Takes the same (visual_input, temporal_input) arguments as the multimodal
    model so the training and evaluation loops are unchanged; temporal_input
    is ignored.
    """

    def __init__(
        self,
        visual_dim: int = 256,
        hidden_dim: int = 256,
        output_dim: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.visual_encoder = FolsomCNNEncoder(
            in_channels=10,
            feature_dim=visual_dim,
        )

        self.forecasting_head = BaselineRegressionHead(
            input_dim=visual_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
        )

    def forward(
        self,
        visual_input: torch.Tensor,
        temporal_input: torch.Tensor,
    ) -> torch.Tensor:
        return self.forecasting_head(self.visual_encoder(visual_input))


class VimTinyLSTMForecastModel(MultimodalSolarForecastModel):
    """Same as cnn_lstm, but the CNN is replaced by Vim-tiny."""

    def __init__(self, **kwargs):
        super().__init__(visual_encoder="vim_tiny", **kwargs)


MODEL_TYPES = {
    "cnn_lstm": MultimodalSolarForecastModel,
    "weather_only": WeatherOnlyForecastModel,
    "image_only": ImageOnlyForecastModel,
    "vim_tiny_lstm": VimTinyLSTMForecastModel,
}


def build_model(model_type: str = "cnn_lstm") -> nn.Module:
    """Build a model by name; see MODEL_TYPES."""
    return MODEL_TYPES[model_type]()


def uses_images(model_type: str) -> bool:
    """Whether the dataset must load sky images for this model."""
    return model_type != "weather_only"


__all__ = [
    "MultimodalSolarForecastModel",
    "WeatherOnlyForecastModel",
    "ImageOnlyForecastModel",
    "VimTinyLSTMForecastModel",
    "build_visual_encoder",
    "MODEL_TYPES",
    "build_model",
    "uses_images",
]
