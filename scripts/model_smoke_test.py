"""Quick smoke test for the complete multimodal model.

This test does not run Folsom calibration or full dataset training.
It verifies that the current CNN + LSTM + fusion + regression pipeline
accepts the expected tensor shapes, produces a forecast, computes a loss,
and supports backpropagation.

Expected inputs:
    visual   -> (B, 10, 180, 720)
    temporal -> (B, 40, 7)

Expected output:
    forecast -> (B, 1)
"""

from __future__ import annotations

import torch

from models.multimodal_model import MultimodalSolarForecastModel


def main() -> None:
    torch.manual_seed(42)

    batch_size = 2

    visual_input = torch.randn(
        batch_size, 10, 180, 720, dtype=torch.float32
    )
    temporal_input = torch.randn(
        batch_size, 40, 7, dtype=torch.float32
    )
    target = torch.randn(batch_size, 1, dtype=torch.float32)

    model = MultimodalSolarForecastModel()
    model.train()

    forecast = model(visual_input, temporal_input)

    assert forecast.shape == (batch_size, 1), (
        f"Unexpected forecast shape: {tuple(forecast.shape)}"
    )
    assert torch.isfinite(forecast).all(), "Forecast contains NaN or Inf."

    loss = torch.nn.functional.mse_loss(forecast, target)
    assert torch.isfinite(loss), "Loss is NaN or Inf."

    loss.backward()

    trainable_grads = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    assert all(grad is not None for grad in trainable_grads), (
        "At least one trainable parameter did not receive a gradient."
    )

    print("Model smoke test PASSED")
    print(f"Visual input : {tuple(visual_input.shape)}")
    print(f"Temporal input: {tuple(temporal_input.shape)}")
    print(f"Forecast     : {tuple(forecast.shape)}")
    print(f"Loss         : {loss.item():.6f}")
    print("Forward pass : OK")
    print("Backward pass: OK")


if __name__ == "__main__":
    main()
