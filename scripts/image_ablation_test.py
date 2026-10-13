"""Image ablation test: does the trained model actually use the sky images?

Runs the SAME trained checkpoint on the SAME chronological 2016 test split,
three ways, and reports RMSE/MAE/MBE/R2 for each:

  1. real      images paired with their own weather sequence (ground truth)
  2. shuffled  each sample gets the image of a random OTHER test sample, drawn
               from the whole 2016 test set (same weather, wrong sky)
  3. zero      every image replaced with a zero tensor (same weather, no sky)

Interpretation:
  * If shuffled/zero RMSE is close to real RMSE, the model is ignoring the
    images -- the weather branch is doing almost all the work.
  * If shuffled/zero RMSE is much worse than real, the images matter.

Needs a GPU (or will fall back to CPU and be slow). Does NOT retrain anything.

All modes use the same samples, so the comparison is paired. Use --subset to
score a fixed random subset of the test set (much faster, same seed every run).

Run on the server, from the repo root:
  python scripts/image_ablation_test.py --subset 20000       # quick, all modes
  python scripts/image_ablation_test.py                      # full test set
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    p.add_argument(
        "--output-dir",
        default="experiments/baseline_lstm_cnn/evaluation",
    )
    p.add_argument(
        "--mode",
        choices=["all", "real", "shuffled", "zero"],
        default="all",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="Seed for the subset choice and the shuffle permutation.",
    )
    p.add_argument(
        "--subset", type=int, default=0,
        help="Score this many randomly chosen test samples (0 = full test set).",
    )
    return p.parse_args()


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mbe = float(np.mean(err))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return mae, rmse, mbe, r2


def load_state(model, state):
    keys = list(state.keys())
    if keys and all(k.startswith("module.") for k in keys):
        state = {k[len("module."):]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)


class AblationView(Dataset):
    """Test samples with the image replaced according to `mode`.

    Weather, target and clear-sky values always come from the sample itself.
    Only the image changes:
      real      the sample's own image
      shuffled  the image of sample perm[i], a different test sample chosen
                from the whole test set (not a neighbour in the same batch,
                whose sky would be nearly identical one minute apart)
      zero      a zero tensor of the same shape
    """

    def __init__(self, base, positions, mode, perm):
        self.base = base
        self.positions = positions
        self.mode = mode
        self.perm = perm

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, j):
        i = self.positions[j]
        image, weather, target, ghi_cs = self.base[i]
        if self.mode == "zero":
            image = torch.zeros_like(image)
        elif self.mode == "shuffled":
            image = self.base[int(self.perm[i])][0]
        return image, weather, target, ghi_cs


def make_derangement(n, rng):
    """Random permutation of range(n) with no element left in place."""
    perm = rng.permutation(n)
    while True:
        fixed = np.flatnonzero(perm == np.arange(n))
        if fixed.size == 0:
            return perm
        k = int(fixed[0])
        other = (k + n // 2) % n
        perm[k], perm[other] = perm[other], perm[k]


def run_mode(model, base, positions, perm, mode, device, args):
    loader = DataLoader(
        AblationView(base, positions, mode, perm),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    y_true, y_pred = [], []
    t0 = time.time()

    with torch.no_grad():
        for batch_idx, (images, weather, targets, ghi_cs) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            weather = weather.to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                pred = model(images, weather)

            pred = pred.detach().float().cpu().numpy().reshape(-1)
            true = targets.cpu().numpy().reshape(-1)
            cs = ghi_cs.cpu().numpy().reshape(-1)

            y_true.extend((true * cs).tolist())
            y_pred.extend((pred * cs).tolist())

            if batch_idx == 1 or batch_idx % 200 == 0:
                print(f"  [{mode}] batch {batch_idx}/{len(loader)}  "
                      f"elapsed={(time.time()-t0)/60:.1f}min", flush=True)

    return y_true, y_pred


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  (CPU runs will be slow)")

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
        },
        "model": {"horizons": [10]},
        "training": {"val_split": 0.1},
    }
    print("Building DataLoaders...")
    _, _, test_loader = get_data_loaders(config)
    print(f"Test batches: {len(test_loader)}")

    model = MultimodalSolarForecastModel().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    load_state(model, ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch')}  "
          f"val_loss={ckpt.get('val_loss')}")

    base = test_loader.dataset  # the 2016 test split, in time order
    n = len(base)
    rng = np.random.default_rng(args.seed)
    perm = make_derangement(n, rng)
    if args.subset and args.subset < n:
        positions = np.sort(rng.choice(n, size=args.subset, replace=False))
    else:
        positions = np.arange(n)
    print(f"Scoring {len(positions)} of {n} test samples (seed={args.seed})")

    modes = ["real", "shuffled", "zero"] if args.mode == "all" else [args.mode]
    rows = [("mode", "n", "MAE_Wm2", "RMSE_Wm2", "MBE_Wm2", "R2")]
    results = {}
    for mode in modes:
        print(f"\n=== mode: {mode} ===")
        y_true, y_pred = run_mode(model, base, positions, perm, mode, device, args)
        mae, rmse, mbe, r2 = regression_metrics(y_true, y_pred)
        print(f"{mode}: n={len(y_true)} MAE={mae:.3f} RMSE={rmse:.3f} MBE={mbe:+.3f} R2={r2:.4f}")
        rows.append((mode, len(y_true), mae, rmse, mbe, r2))
        results[mode] = rmse

    if "real" in results and len(results) > 1:
        print("\n=== relative to real ===")
        for mode in modes:
            if mode == "real":
                continue
            dr = results[mode] - results["real"]
            pct = 100 * dr / results["real"]
            print(f"{mode}:  RMSE {dr:+.2f} W/m2  ({pct:+.2f}% vs real)")

    tag = f"subset{len(positions)}" if len(positions) < n else "full"
    out = Path(args.output_dir) / f"image_ablation_metrics_{tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        for row in rows:
            w.writerow([
                f"{v:.6f}" if isinstance(v, float) else v
                for v in row
            ])
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
