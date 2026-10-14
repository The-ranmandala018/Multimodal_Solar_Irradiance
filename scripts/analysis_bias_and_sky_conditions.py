"""Bias correction and sky-condition breakdown: CNN+LSTM vs weather-only.

(a) Bias check
    Runs each trained model once on the VALIDATION samples (last 10 % of
    2014+2015 from chronological_year_split) on CPU, takes the mean bias
    (predicted GHI - true GHI) there, subtracts it from the model's saved
    2016 test predictions, and reports MAE/RMSE/MBE/R2 before and after.
    The bias is estimated on validation data only, never on the test set.

(b) Sky-condition breakdown
    Classifies each 2016 test sample from the trailing 30-min window of the
    raw clear-sky index k (t-29 min .. t, 60 s steps, from target_df):
        clear    : std(k) < 0.05 and mean(k) > 0.9
        overcast : std(k) < 0.10 and mean(k) < 0.5
        mixed    : everything else
    (simple k mean/std thresholds -- not the Stein et al. 2012 Variability
    Index, which compares GHI and clear-sky line lengths). Then reports
    n/MAE/RMSE/MBE/R2 per class for CNN+LSTM, weather-only, naive persistence
    and clear-sky persistence, from the ALREADY SAVED test predictions.

Saved predictions are matched to test samples by chronological position.
evaluate_test.py run on W GPUs saves rank 0's samples, then rank 1's, ...;
those files are put back into time order, and every sample's saved target is
checked against the target rebuilt from target_df before anything is compared.

No GPU is used. Outputs (experiments/analysis/):
    bias_correction.csv, sky_condition_breakdown.csv, sky_condition_rmse.png

Run from the repo root:
  python scripts/analysis_bias_and_sky_conditions.py
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only, by design

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from datasets.folsom_dataset import get_data_loaders  # noqa: E402
from models.multimodal_model import build_model  # noqa: E402

MODELS = [
    # key, label, checkpoint, saved test predictions
    ("cnn_lstm", "CNN+LSTM",
     "experiments/baseline_lstm_cnn/checkpoints/best_model.pt",
     "experiments/baseline_lstm_cnn/evaluation/test_predictions.npz"),
    ("weather_only", "Weather-only LSTM",
     "experiments/weather_only_seed0/checkpoints/best_model.pt",
     "experiments/weather_only_seed0/evaluation/test_predictions.npz"),
]
BASELINES = [("naive_persistence", "Naive persistence"),
             ("clearsky_persistence", "Clear-sky persistence")]
CLASSES = ["clear", "mixed", "overcast"]
WINDOW_MIN = 30

# Colours match plot_model_comparison.py; baselines are neutral greys.
COLORS = {"naive_persistence": "#c3c2b7", "clearsky_persistence": "#8a8984",
          "weather_only": "#eb6834", "cnn_lstm": "#2a78d6"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="experiments/analysis")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=12)
    p.add_argument("--threads", type=int, default=16, help="torch CPU threads")
    return p.parse_args()


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, np.float64)
    err = np.asarray(y_pred, np.float64) - y_true
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "n": int(len(y_true)),
        "MAE_Wm2": float(np.mean(np.abs(err))),
        "RMSE_Wm2": float(np.sqrt(np.mean(err ** 2))),
        "MBE_Wm2": float(np.mean(err)),
        "R2": 1.0 - float(np.sum(err ** 2)) / ss_tot if ss_tot > 0 else float("nan"),
    }


def build_data():
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
    # Same dataset, split and train-only normalisation as training/evaluation.
    _, val_loader, test_loader = get_data_loaders(config)
    return val_loader.dataset, test_loader.dataset


def predict_validation(key, checkpoint, val_subset, args):
    """Run one model on the validation samples (CPU). Returns GHI (true, pred)."""
    dataset = val_subset.dataset
    dataset.load_images = key != "weather_only"  # same switch as training
    loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        prefetch_factor=4 if args.num_workers else None)
    model = build_model(key)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]
    if all(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()

    y_true, y_pred = [], []
    t0 = time.time()
    with torch.no_grad():
        for b, (images, weather, targets, ghi_cs) in enumerate(loader, start=1):
            pred = model(images, weather).numpy().reshape(-1)
            cs = ghi_cs.numpy().reshape(-1)
            y_true.append(targets.numpy().reshape(-1) * cs)
            y_pred.append(pred * cs)
            if b == 1 or b % 100 == 0 or b == len(loader):
                print(f"  [{key} validation] batch {b}/{len(loader)}  "
                      f"elapsed={(time.time() - t0) / 60:.1f} min", flush=True)
    dataset.load_images = True
    return np.concatenate(y_true), np.concatenate(y_pred)


def test_table(test_subset):
    """Per test sample, in time order: time, target, persistence, sky class."""
    dataset = test_subset.dataset
    times = pd.DatetimeIndex([dataset.samples[i][1] for i in test_subset.indices])
    if not times.is_monotonic_increasing:
        raise RuntimeError("test samples are not in time order")

    tdf = dataset.target_df  # raw, unnormalised (copied before normalisation)
    tdf = tdf[~tdf.index.duplicated(keep="first")]
    k, cs, ghi = tdf["k_index"], tdf["GHI_cs"], tdf["GHI"]

    t10 = times + pd.Timedelta(minutes=10)
    k_t, ghi_t = k.reindex(times).to_numpy(), ghi.reindex(times).to_numpy()
    k_t10, cs_t10 = k.reindex(t10).to_numpy(), cs.reindex(t10).to_numpy()
    if np.isnan(k_t).any() or np.isnan(k_t10).any() or np.isnan(cs_t10).any():
        raise RuntimeError("some test times or t+10 targets are missing from target_df")

    # Trailing 30-min k window (t-29 .. t); missing minutes are ignored.
    win = np.column_stack([
        k.reindex(times - pd.Timedelta(minutes=m)).to_numpy()
        for m in range(WINDOW_MIN - 1, -1, -1)
    ])
    n_avail = np.sum(~np.isnan(win), axis=1)
    k_mean, k_std = np.nanmean(win, axis=1), np.nanstd(win, axis=1)
    sky = np.full(len(times), "mixed", dtype=object)
    sky[(k_std < 0.05) & (k_mean > 0.9)] = "clear"
    sky[(k_std < 0.10) & (k_mean < 0.5)] = "overcast"
    print(f"30-min windows: {np.mean(n_avail == WINDOW_MIN) * 100:.2f}% complete, "
          f"min {n_avail.min()} of {WINDOW_MIN} minutes present")

    return pd.DataFrame({
        "time": times,
        "y_true": k_t10 * cs_t10,                    # same target the models use
        "naive_persistence": ghi_t,                  # GHI(t)
        "clearsky_persistence": k_t * cs_t10,        # k(t) * GHI_cs(t+10)
        "k_mean_30min": k_mean, "k_std_30min": k_std,
        "window_minutes": n_avail, "sky": sky,
    })


def load_saved_predictions(path, y_ref, label):
    """Saved test predictions in time order, verified against y_ref."""
    d = np.load(path)
    y_true, y_pred = d["ghi_clipped_true"].astype(np.float64), d["ghi_clipped_pred"].astype(np.float64)
    n = len(y_ref)
    if len(y_true) != n:
        raise RuntimeError(f"{label}: {len(y_true)} saved test samples, expected {n}")
    for world in range(1, 5):
        saved_order = np.concatenate([np.arange(r, n, world) for r in range(world)])
        inverse = np.empty(n, dtype=np.int64)
        inverse[saved_order] = np.arange(n)
        if np.allclose(y_true[inverse], y_ref, atol=0.05):
            max_diff = np.max(np.abs(y_true[inverse] - y_ref))
            print(f"{label}: saved by a {world}-GPU run; matched all {n} targets "
                  f"(max |diff| {max_diff:.4f} W/m2)")
            return y_pred[inverse]
    raise RuntimeError(f"{label}: saved targets do not match target_df in any rank order")


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow([f"{v:.6f}" if isinstance(v, float) else v for v in row])
    print(f"Saved: {path}")


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building dataset (same split and normalisation as training)...")
    val_subset, test_subset = build_data()
    print(f"Validation samples: {len(val_subset)}   Test samples: {len(test_subset)}")

    table = test_table(test_subset)
    preds = {}
    for key, label, _, npz in MODELS:
        preds[key] = load_saved_predictions(npz, table["y_true"].to_numpy(), label)

    # (a) bias check
    bias_rows = []
    for key, label, ckpt, _ in MODELS:
        v_true, v_pred = predict_validation(key, ckpt, val_subset, args)
        val_bias = float(np.mean(v_pred - v_true))
        print(f"{label}: validation n={len(v_true)} mean bias {val_bias:+.3f} W/m2")
        for when, y_pred in (("before", preds[key]), ("after", preds[key] - val_bias)):
            m = metrics(table["y_true"], y_pred)
            bias_rows.append((key, when, val_bias, len(v_true), m["n"],
                              m["MAE_Wm2"], m["RMSE_Wm2"], m["MBE_Wm2"], m["R2"]))
            print(f"  test {when:6s}: MAE={m['MAE_Wm2']:.3f} RMSE={m['RMSE_Wm2']:.3f} "
                  f"MBE={m['MBE_Wm2']:+.3f} R2={m['R2']:.4f}")
    write_csv(out_dir / "bias_correction.csv",
              ["model", "correction", "val_bias_Wm2", "val_n", "test_n",
               "MAE_Wm2", "RMSE_Wm2", "MBE_Wm2", "R2"], bias_rows)

    # (b) sky-condition breakdown
    predictors = [(k, l) for k, l in BASELINES] + [(k, l) for k, l, *_ in MODELS]
    series = {k: table[k].to_numpy() for k, _ in BASELINES} | preds
    sky_rows, rmse = [], {}
    for cls in CLASSES:
        mask = (table["sky"] == cls).to_numpy()
        for key, _ in predictors:
            m = metrics(table["y_true"].to_numpy()[mask], series[key][mask])
            rmse[(cls, key)] = (m["RMSE_Wm2"], m["n"])
            sky_rows.append((cls, key, m["n"], m["MAE_Wm2"], m["RMSE_Wm2"], m["MBE_Wm2"], m["R2"]))
        print(f"{cls:8s} n={int(mask.sum()):6d} " + "  ".join(
            f"{key}={rmse[(cls, key)][0]:.1f}" for key, _ in predictors))
    write_csv(out_dir / "sky_condition_breakdown.csv",
              ["sky_class", "predictor", "n", "MAE_Wm2", "RMSE_Wm2", "MBE_Wm2", "R2"], sky_rows)

    # Bar chart: RMSE, 4 bars per sky class
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.2
    for j, (key, label) in enumerate(predictors):
        xs = np.arange(len(CLASSES)) + (j - 1.5) * width
        vals = [rmse[(c, key)][0] for c in CLASSES]
        ax.bar(xs, vals, width=width * 0.92, color=COLORS[key], label=label,
               edgecolor="white", linewidth=1)
        for x, v in zip(xs, vals):
            ax.text(x, v + 1.5, f"{v:.0f}", ha="center", va="bottom", color=INK, fontsize=8.5)
    n_total = len(table)
    ax.set_xticks(range(len(CLASSES)), [
        f"{c.capitalize()}\nn = {rmse[(c, 'cnn_lstm')][1]:,} "
        f"({100 * rmse[(c, 'cnn_lstm')][1] / n_total:.0f}%)" for c in CLASSES])
    ax.set_ylabel("Test RMSE (W/m²)", color=INK)
    ax.set_ylim(0, max(v for v, _ in rmse.values()) * 1.15)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelcolor=INK)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=4, loc="upper left", fontsize=9)
    fig.suptitle("Test RMSE by sky condition, 10-min GHI forecast (Folsom 2016)",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.text(0.01, 0.01,
             "Sky class from the trailing 30-min clear-sky index k: clear = std<0.05 & mean>0.9; "
             "overcast = std<0.10 & mean<0.5; mixed = rest.",
             color=INK_2, fontsize=8, ha="left", va="bottom")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.2)
    fig.savefig(out_dir / "sky_condition_rmse.png", dpi=200, facecolor="white")
    print(f"Saved: {out_dir / 'sky_condition_rmse.png'}")


if __name__ == "__main__":
    main()
