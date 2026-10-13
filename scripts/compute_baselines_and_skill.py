"""Compute persistence baselines and forecast skill for the Folsom CNN+LSTM model.

Two things this script does NOT need: a GPU, or re-running the model. It reads
the already-saved experiments/baseline_lstm_cnn/evaluation/test_predictions.npz
for the model's own numbers, and rebuilds only the lightweight tabular (CSV)
part of the dataset -- no images, no network -- to compute two persistence
baselines on the exact same chronological 2016 test split.

Baselines, both compared against the same clipped target GHI(t+10) the model
is evaluated against (k_index(t+10)_clipped * GHI_cs(t+10)):

  naive persistence       GHI(t+10)_pred = GHI(t)                      (raw)
  clear-sky persistence   GHI(t+10)_pred = k_index(t) * GHI_cs(t+10)   (clipped k at t)

Forecast skill (FS) of the trained model relative to a baseline:
  FS = 1 - RMSE_model / RMSE_baseline

Run on the server, from the repo root:
  python scripts/compute_baselines_and_skill.py
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from datasets.chrono import chronological_year_split
from datasets.folsom_dataset import FolsomDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-predictions",
        default="experiments/baseline_lstm_cnn/evaluation/test_predictions.npz",
    )
    p.add_argument(
        "--model-metrics",
        default="experiments/baseline_lstm_cnn/evaluation/test_metrics.csv",
    )
    p.add_argument(
        "--output",
        default="experiments/baseline_lstm_cnn/evaluation/full_metrics_summary.csv",
    )
    return p.parse_args()


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mbe = float(np.mean(err))  # signed: >0 means the model over-predicts on average
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return mae, rmse, mbe, r2


def read_csv_metric(path, name):
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if row and row[0] == name:
                return float(row[1])
    raise ValueError(f"{name!r} not found in {path}")


def build_raw_target_table():
    """Load only the CSV-derived table (no images) for the same split used in training."""
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
            "batch_size": 8,
            "num_workers": 0,
            "distributed": False,
        },
        "model": {"horizons": [10]},
        "training": {"val_split": 0.1},
    }
    print("Building dataset table (CSV + solar geometry only, no images)...")
    dataset = FolsomDataset(config)
    _, _, test_idx = chronological_year_split(
        dataset.samples, val_split=0.1, train_years=(2014, 2015), test_year=2016
    )
    print(f"Test samples (image-matched, SZA<=85): {len(test_idx)}")
    return dataset, test_idx


def compute_persistence(dataset, test_idx):
    target_df = dataset.target_df  # raw, unnormalized GHI / k_index / GHI_cs
    y_true, y_naive, y_clearsky = [], [], []
    skipped = 0

    for i in test_idx:
        _, dt = dataset.samples[i]
        try:
            row_t = target_df.loc[dt]
            row_tgt = target_df.loc[dt + pd.Timedelta(minutes=10)]
        except KeyError:
            skipped += 1
            continue
        if isinstance(row_t, pd.DataFrame):
            row_t = row_t.iloc[0]
        if isinstance(row_tgt, pd.DataFrame):
            row_tgt = row_tgt.iloc[0]

        ghi_true = float(row_tgt["k_index"]) * float(row_tgt["GHI_cs"])
        y_true.append(ghi_true)
        y_naive.append(float(row_t["GHI"]))
        y_clearsky.append(float(row_t["k_index"]) * float(row_tgt["GHI_cs"]))

    if skipped:
        print(f"WARNING: {skipped}/{len(test_idx)} test samples skipped "
              f"(t or t+10 not found in raw table). Investigate if this is large.")

    print(f"Scored samples: {len(y_true)}")
    return y_true, y_naive, y_clearsky


def main():
    args = parse_args()

    # --- Model's own numbers, from the already-saved evaluation run ---
    with np.load(args.model_predictions) as d:
        model_true = d["ghi_clipped_true"]
        model_pred = d["ghi_clipped_pred"]
    model_mae, model_rmse, model_mbe, model_r2 = regression_metrics(model_true, model_pred)
    model_rmse_saved = read_csv_metric(args.model_metrics, "GHI_clipped_target_RMSE_Wm2")
    if abs(model_rmse - model_rmse_saved) > 1e-3:
        print(f"WARNING: recomputed model RMSE ({model_rmse:.4f}) does not match "
              f"{args.model_metrics} ({model_rmse_saved:.4f}). Using the recomputed value.")

    # --- Persistence baselines, from the raw CSV/solar-geometry table ---
    dataset, test_idx = build_raw_target_table()
    y_true, y_naive, y_clearsky = compute_persistence(dataset, test_idx)

    if len(y_true) != len(model_true):
        print(f"NOTE: baseline sample count ({len(y_true)}) differs from the model's "
              f"saved sample count ({len(model_true)}). Both still cover the full "
              f"chronological 2016 test set; FS below uses aggregate RMSEs, which "
              f"does not require matching row order or count exactly, but a large "
              f"gap is worth investigating.")

    rows = [("set", "n", "MAE_Wm2", "RMSE_Wm2", "MBE_Wm2", "R2", "forecast_skill_vs_model")]
    rows.append(("model (CNN+LSTM)", len(model_true), model_mae, model_rmse, model_mbe, model_r2, ""))

    for name, y_pred in [("naive_persistence", y_naive), ("clearsky_persistence", y_clearsky)]:
        mae, rmse, mbe, r2 = regression_metrics(y_true, y_pred)
        fs = 1.0 - model_rmse / rmse
        rows.append((name, len(y_true), mae, rmse, mbe, r2, fs))
        print(f"{name}: n={len(y_true)} MAE={mae:.3f} RMSE={rmse:.3f} MBE={mbe:.3f} "
              f"R2={r2:.4f}  FS(model)={fs:.4f}")

    print(f"model: n={len(model_true)} MAE={model_mae:.3f} RMSE={model_rmse:.3f} "
          f"MBE={model_mbe:.3f} R2={model_r2:.4f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        for row in rows:
            w.writerow([
                f"{v:.6f}" if isinstance(v, float) else v
                for v in row
            ])
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
