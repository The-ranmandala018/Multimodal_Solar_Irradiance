"""Evaluate a Mamba-LSTM (or CNN+LSTM) checkpoint on validation and test, raw and bias-corrected.

    python scripts/evaluate_mamba_lstm.py --checkpoint experiments/mamba_lstm/seed0/checkpoints/best.pt \
        --output-dir experiments/mamba_lstm/seed0/evaluation

    # Validation predictions of an existing CNN+LSTM checkpoint (read-only), for its bias correction:
    python scripts/evaluate_mamba_lstm.py --model-type cnn_lstm --splits val \
        --checkpoint experiments/cnn_lstm_seed1/checkpoints/best_model.pt \
        --output-dir experiments/mamba_lstm/comparison/cnn_validation/cnn_lstm_seed1

Data, split, target and test metrics follow scripts/evaluate_test.py (one GPU,
fp16 autocast, GHI = k * GHI_cs at t+10). Predictions are saved in
chronological order with each sample's timestamp.

Bias correction follows scripts/analysis_bias_and_sky_conditions.py: one
constant, the mean GHI error on the VALIDATION split (last 10 % of 2014-2015),
subtracted from the test predictions. The test set is never used to fit it.

Existing output files are never overwritten (use a new --output-dir).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from datasets.folsom_dataset import get_data_loaders  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-type", choices=["mamba_lstm", "cnn_lstm"], default="mamba_lstm")
    p.add_argument("--config", default="experiments/mamba_lstm/config.yaml",
                   help="Used for the data settings and, if the checkpoint has none, the model.")
    p.add_argument("--splits", default="val,test")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-batches", type=int, default=0, help="Smoke test only.")
    return p.parse_args()


def metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return {
        "n": len(y_true),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MBE": float(np.mean(err)),
        "R2": float(1 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
    }


def build(model_type, checkpoint, cfg):
    if model_type == "cnn_lstm":
        from models.multimodal_model import build_model
        model = build_model("cnn_lstm")
    else:
        from models.mamba_lstm_model import build_mamba_lstm
        model = build_mamba_lstm(checkpoint.get("config", cfg)["model"], pretrained=False)
    state = checkpoint["model_state_dict"]
    if state and all(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model


def predict(model, subset, device, args, label):
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        prefetch_factor=2 if args.num_workers else None)
    k_true, k_pred, cs_all = [], [], []
    t0 = time.time()
    total = min(len(loader), args.max_batches) if args.max_batches else len(loader)
    with torch.no_grad():
        for b, (images, weather, targets, ghi_cs) in enumerate(loader, start=1):
            if args.max_batches and b > args.max_batches:
                break
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = model(images.to(device, non_blocking=True), weather.to(device, non_blocking=True))
            k_pred.append(pred.float().cpu().numpy().reshape(-1))
            k_true.append(targets.numpy().reshape(-1))
            cs_all.append(ghi_cs.numpy().reshape(-1))
            if b == 1 or b % 500 == 0 or b == total:
                el = time.time() - t0
                print(f"  [{label}] {b}/{total} batches, {el / 60:.1f} min, "
                      f"ETA {el / b * (total - b) / 60:.1f} min", flush=True)
    k_true, k_pred, cs = map(np.concatenate, (k_true, k_pred, cs_all))
    return k_true, k_pred, cs, time.time() - t0


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA is required.")
    device = torch.device("cuda")
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = REPO / out
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    planned = [out / f"{s}_predictions.npz" for s in splits]
    planned += [out / "test_metrics.csv"] if "test" in splits else []
    planned += [out / "metrics_raw_and_bias_corrected.csv"] if set(splits) == {"val", "test"} else []
    existing = [str(p) for p in planned if p.exists()]
    if existing:
        sys.exit(f"Refusing to overwrite existing results: {existing}. Use a new --output-dir.")
    out.mkdir(parents=True, exist_ok=True)

    with open(REPO / args.config) as f:
        cfg = yaml.safe_load(f)
    d = cfg["data"]
    _, val_loader0, test_loader0 = get_data_loaders({
        "data": {**{k: d[k] for k in ("csv_path", "image_root", "sequence_length", "sampling_rate_sec",
                                      "image_tolerance_sec", "latitude", "longitude", "altitude")},
                 "batch_size": 1, "num_workers": 0, "distributed": False,
                 "rank": 0, "world_size": 1, "load_images": True},
        "model": {"horizons": d["horizons"]},
        "training": {"val_split": d["val_split"]},
    })
    subsets = {"val": val_loader0.dataset, "test": test_loader0.dataset}

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build(args.model_type, checkpoint, cfg).to(device).eval()
    print(f"Model {args.model_type} from {args.checkpoint} (epoch {checkpoint.get('epoch')}, "
          f"val_loss {checkpoint.get('val_loss')})")

    results = {}
    for split in splits:
        subset = subsets[split]
        k_true, k_pred, cs, seconds = predict(model, subset, device, args, split)
        times = np.array([subset.dataset.samples[i][1] for i in subset.indices[: len(k_true)]],
                         dtype="datetime64[ns]")
        if np.any(np.diff(times.astype(np.int64)) < 0):
            raise RuntimeError(f"{split} samples are not in time order")
        ghi_true, ghi_pred = k_true * cs, k_pred * cs
        tmp = out / f"{split}_predictions.tmp.npz"
        np.savez_compressed(tmp, sample_time=times,
                            k_index_true=k_true.astype(np.float32), k_index_pred=k_pred.astype(np.float32),
                            ghi_clipped_true=ghi_true.astype(np.float32),
                            ghi_clipped_pred=ghi_pred.astype(np.float32))
        os.replace(tmp, out / f"{split}_predictions.npz")
        results[split] = dict(k_true=k_true, k_pred=k_pred, ghi_true=ghi_true, ghi_pred=ghi_pred,
                              seconds=seconds)
        m = metrics(ghi_true, ghi_pred)
        print(f"{split}: n={m['n']} RMSE={m['RMSE']:.3f} MAE={m['MAE']:.3f} MBE={m['MBE']:+.3f} "
              f"R2={m['R2']:.4f} W/m2 ({seconds / 60:.1f} min)")

    if "test" in results:
        r = results["test"]
        km, gm = metrics(r["k_true"], r["k_pred"]), metrics(r["ghi_true"], r["ghi_pred"])
        ghi_mean = float(np.mean(r["ghi_true"]))
        rows = [  # same names as scripts/evaluate_test.py, plus MBE
            ("checkpoint", args.checkpoint), ("checkpoint_epoch", checkpoint.get("epoch")),
            ("checkpoint_val_loss", checkpoint.get("val_loss")), ("test_samples", gm["n"]),
            ("k_index_MAE", km["MAE"]), ("k_index_RMSE", km["RMSE"]), ("k_index_R2", km["R2"]),
            ("GHI_clipped_target_MAE_Wm2", gm["MAE"]), ("GHI_clipped_target_RMSE_Wm2", gm["RMSE"]),
            ("GHI_clipped_target_R2", gm["R2"]), ("GHI_clipped_target_mean_Wm2", ghi_mean),
            ("GHI_clipped_target_nMAE", gm["MAE"] / ghi_mean),
            ("GHI_clipped_target_nRMSE", gm["RMSE"] / ghi_mean),
            ("GHI_clipped_target_MBE_Wm2", gm["MBE"]),
            ("inference_seconds", r["seconds"]),
        ]
        with open(out / "test_metrics.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            w.writerows(rows)

    if "val" in results and "test" in results:
        v, t = results["val"], results["test"]
        val_bias = float(np.mean(v["ghi_pred"] - v["ghi_true"]))
        with open(out / "metrics_raw_and_bias_corrected.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["correction", "val_bias_Wm2", "val_n", "test_n", "MAE_Wm2", "RMSE_Wm2",
                        "MBE_Wm2", "R2"])
            for name, pred in (("raw", t["ghi_pred"]), ("val_bias_corrected", t["ghi_pred"] - val_bias)):
                m = metrics(t["ghi_true"], pred)
                w.writerow([name, f"{val_bias:.6f}", len(v["ghi_true"]), m["n"], f"{m['MAE']:.6f}",
                            f"{m['RMSE']:.6f}", f"{m['MBE']:.6f}", f"{m['R2']:.6f}"])
                print(f"test {name}: RMSE={m['RMSE']:.3f} MAE={m['MAE']:.3f} MBE={m['MBE']:+.3f} "
                      f"(validation bias {val_bias:+.3f} W/m2)")
    print(f"Saved results in {out}")


if __name__ == "__main__":
    main()
