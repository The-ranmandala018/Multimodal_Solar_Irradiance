"""GPU batch smoke test for the real Folsom multimodal pipeline.

This is NOT a training run. It loads one small real batch, measures CPU
DataLoader/preprocessing time, GPU transfer time, forward time, backward
time, GPU memory, and verifies that the model/loss/backpropagation work.

The original Folsom source is not modified.
"""

from __future__ import annotations

import time

import torch
from torch import nn
from torch.amp import GradScaler, autocast

from datasets.folsom_dataset import FolsomDataset
from models.multimodal_model import MultimodalSolarForecastModel


BATCH_SIZE = 2

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
    },
    "model": {"horizons": [10]},
}


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Stop before running the GPU test.")

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

    dataset = FolsomDataset(CONFIG)
    print(f"Available samples: {len(dataset)}")

    print(f"Loading {BATCH_SIZE} real samples...")
    cpu_start = time.perf_counter()

    samples = [dataset[i] for i in range(BATCH_SIZE)]
    images = torch.stack([sample[0] for sample in samples])
    weather = torch.stack([sample[1] for sample in samples])
    targets = torch.stack([sample[2] for sample in samples])

    cpu_time = time.perf_counter() - cpu_start

    print(f"CPU preprocessing time: {cpu_time:.3f} s")
    print(f"Images: {tuple(images.shape)}")
    print(f"Weather: {tuple(weather.shape)}")
    print(f"Targets: {tuple(targets.shape)}")

    assert images.shape == (BATCH_SIZE, 10, 180, 720)
    assert weather.shape == (BATCH_SIZE, 40, 7)
    assert targets.shape == (BATCH_SIZE, 1)

    model = MultimodalSolarForecastModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    scaler = GradScaler("cuda")

    torch.cuda.synchronize()
    transfer_start = time.perf_counter()

    images = images.to(device, non_blocking=True)
    weather = weather.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)

    torch.cuda.synchronize()
    transfer_time = time.perf_counter() - transfer_start

    optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    forward_start = time.perf_counter()

    with autocast(device_type="cuda"):
        predictions = model(images, weather)
        loss = criterion(predictions, targets)

    torch.cuda.synchronize()
    forward_time = time.perf_counter() - forward_start

    torch.cuda.synchronize()
    backward_start = time.perf_counter()

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    torch.cuda.synchronize()
    backward_time = time.perf_counter() - backward_start

    memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)

    print()
    print("GPU BATCH SMOKE TEST PASSED")
    print("--------------------------------")
    print(f"Batch size          : {BATCH_SIZE}")
    print(f"CPU preprocessing   : {cpu_time:.3f} s")
    print(f"CPU -> GPU transfer : {transfer_time:.3f} s")
    print(f"GPU forward         : {forward_time:.3f} s")
    print(f"GPU backward        : {backward_time:.3f} s")
    print(f"Loss                : {loss.item():.6f}")
    print(f"Prediction shape    : {tuple(predictions.shape)}")
    print(f"GPU memory allocated: {memory_allocated:.2f} GB")
    print(f"GPU memory reserved : {memory_reserved:.2f} GB")
    print("--------------------------------")
    print("Forward pass : OK")
    print("Backward pass: OK")
    print("Real Folsom batch -> GPU model pipeline: OK")
    print()
    print("This is only a performance/integration test.")
    print("It is NOT a trained forecasting result.")


if __name__ == "__main__":
    main()
