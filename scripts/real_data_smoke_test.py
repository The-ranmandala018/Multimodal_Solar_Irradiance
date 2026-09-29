"""Smoke test the complete model with one real Folsom sample.

This is a pipeline/integration test, not a trained-model evaluation.
It loads one real Folsom image and its real 40 x 7 weather history,
passes both through the current CNN + LSTM + fusion + regression model,
and checks the resulting shapes and finite values.

The test uses the existing 2014 calibration results through
FolsomPreprocessor. It does not modify the original Folsom source.
"""

from __future__ import annotations

import torch

from datasets.folsom_dataset import FolsomDataset
from models.multimodal_model import MultimodalSolarForecastModel


CONFIG = {
    "data": {
        "csv_path": "/storage2/CV_Irradiance/datasets/1_Folsom/csv_files/Folsom_irradiance_weather.csv",
        "image_root": "/storage2/CV_Irradiance/datasets/1_Folsom",
        "sequence_length": 40,
        "sampling_rate_sec": 60,
        "image_tolerance_sec": 120,
        "latitude": 38.642,
        "longitude": -121.148,
        "altitude": 60.0,
        "batch_size": 1,
        "num_workers": 0,
    },
    "model": {
        "horizons": [10],
    },
}


def main() -> None:
    torch.manual_seed(42)

    print("Loading Folsom dataset metadata...")
    dataset = FolsomDataset(CONFIG)

    if len(dataset) == 0:
        raise RuntimeError("No valid Folsom samples were found.")

    print(f"Available samples: {len(dataset)}")

    # Use the first real sample. This test is intentionally small.
    image, weather, target, ghi_cs = dataset[0]

    print(f"Image tensor      : {tuple(image.shape)}")
    print(f"Weather sequence  : {tuple(weather.shape)}")
    print(f"Target            : {tuple(target.shape)}")
    print(f"GHI clear-sky     : {tuple(ghi_cs.shape)}")

    assert image.shape == (10, 180, 720)
    assert weather.shape == (40, 7)
    assert target.shape == (1,)
    assert ghi_cs.shape == (1,)

    assert torch.isfinite(image).all(), "Image contains NaN or Inf."
    assert torch.isfinite(weather).all(), "Weather contains NaN or Inf."
    assert torch.isfinite(target).all(), "Target contains NaN or Inf."

    model = MultimodalSolarForecastModel()
    model.eval()

    with torch.no_grad():
        prediction = model(
            image.unsqueeze(0),
            weather.unsqueeze(0),
        )

    print(f"Prediction        : {tuple(prediction.shape)}")
    print(f"Prediction value  : {prediction.item():.6f}")
    print(f"Actual target     : {target.item():.6f}")

    assert prediction.shape == (1, 1)
    assert torch.isfinite(prediction).all(), "Prediction contains NaN or Inf."

    print()
    print("Real-data smoke test PASSED")
    print("Real Folsom image -> preprocessing -> CNN ->")
    print("LSTM -> fusion -> regression -> prediction: OK")
    print()
    print("Note: the model is untrained, so the prediction value")
    print("is NOT a forecasting accuracy result.")


if __name__ == "__main__":
    main()
