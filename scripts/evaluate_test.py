"""Evaluate the best Folsom CNN+LSTM checkpoint on the chronological 2016 test set."""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch
from torch import nn

from datasets.folsom_dataset import get_data_loaders
from models.multimodal_model import MultimodalSolarForecastModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        default="experiments/baseline_lstm_cnn/checkpoints/best_model.pt",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output-dir", default="experiments/baseline_lstm_cnn/evaluation")
    return p.parse_args()


def load_state(model, state):
    keys = list(state.keys())
    if keys and all(k.startswith("module.") for k in keys):
        state = {k[len("module."):]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return mae, rmse, r2


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    config = {
        "data": {
            "csv_path": "/storage2/CV_Irradiance/datasets/1_Folsom/csv_files/Folsom_irradiance_weather.csv",
            "image_root": "/storage2/CV_Irradiance/datasets/1_Folsom",
            "sequence_length": 40,
            "sampling_rate_sec": 60,
            "image_tolerance_sec": 120,
            "latitude": 38.642,
            "longitude": -121.148,
            "altitude": 60.0,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "prefetch_factor": 2,
            "distributed": False,
            "rank": 0,
            "world_size": 1,
        },
        "model": {"horizons": [10]},
        "training": {"val_split": 0.1},
    }

    print("Building chronological train/validation/test loaders...")
    _, _, test_loader = get_data_loaders(config)
    print(f"Test batches: {len(test_loader)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    model = MultimodalSolarForecastModel().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    load_state(model, checkpoint["model_state_dict"])
    model.eval()

    y_k_true, y_k_pred = [], []
    y_ghi_true, y_ghi_pred = [], []

    with torch.no_grad():
        for batch_idx, (images, weather, targets, ghi_cs) in enumerate(test_loader, start=1):
            images = images.to(device, non_blocking=True)
            weather = weather.to(device, non_blocking=True)

            pred = model(images, weather).detach().cpu().numpy().reshape(-1)
            true = targets.cpu().numpy().reshape(-1)
            cs = ghi_cs.cpu().numpy().reshape(-1)

            y_k_pred.extend(pred.tolist())
            y_k_true.extend(true.tolist())

            # The dataset target k_index is clipped to [0, 1.2].
            # Therefore this reconstructs the corresponding clipped-target GHI.
            y_ghi_pred.extend((pred * cs).tolist())
            y_ghi_true.extend((true * cs).tolist())

            if batch_idx == 1 or batch_idx % 100 == 0:
                print(f"Evaluated {batch_idx}/{len(test_loader)} batches")

    k_mae, k_rmse, k_r2 = regression_metrics(y_k_true, y_k_pred)
    ghi_mae, ghi_rmse, ghi_r2 = regression_metrics(y_ghi_true, y_ghi_pred)

    ghi_mean = float(np.mean(y_ghi_true))
    ghi_nmae = ghi_mae / ghi_mean if ghi_mean != 0 else float("nan")
    ghi_nrmse = ghi_rmse / ghi_mean if ghi_mean != 0 else float("nan")

    results = [
        ("checkpoint", args.checkpoint),
        ("test_samples", len(y_k_true)),
        ("k_index_MAE", k_mae),
        ("k_index_RMSE", k_rmse),
        ("k_index_R2", k_r2),
        ("GHI_clipped_target_MAE_Wm2", ghi_mae),
        ("GHI_clipped_target_RMSE_Wm2", ghi_rmse),
        ("GHI_clipped_target_R2", ghi_r2),
        ("GHI_clipped_target_mean_Wm2", ghi_mean),
        ("GHI_clipped_target_nMAE", ghi_nmae),
        ("GHI_clipped_target_nRMSE", ghi_nrmse),
    ]

    print("\n=== TEST RESULTS ===")
    for name, value in results:
        if isinstance(value, (float, np.floating)):
            print(f"{name}: {value:.6f}")
        else:
            print(f"{name}: {value}")

    csv_path = os.path.join(args.output_dir, "test_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(results)

    np.savez(
        os.path.join(args.output_dir, "test_predictions.npz"),
        k_index_true=np.asarray(y_k_true),
        k_index_pred=np.asarray(y_k_pred),
        ghi_clipped_true=np.asarray(y_ghi_true),
        ghi_clipped_pred=np.asarray(y_ghi_pred),
    )

    print(f"Saved metrics: {csv_path}")
    print(f"Saved predictions: {os.path.join(args.output_dir, 'test_predictions.npz')}")


if __name__ == "__main__":
    main()
