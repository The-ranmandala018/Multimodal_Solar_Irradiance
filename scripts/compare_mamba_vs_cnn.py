"""Compare Mamba-LSTM runs with the CNN+LSTM baseline runs on the 2016 test set.

    python scripts/compare_mamba_vs_cnn.py

Reads saved predictions only (no model is run) and writes to
experiments/mamba_lstm/comparison/:
    per_run_metrics.csv   RMSE/MAE/MBE/R2 (W/m2) and forecast skill per run, raw and
                          validation-bias-corrected, plus training time and best epoch
    summary.csv           mean and sample SD over seeds per model and correction
    bootstrap.csv         day-block bootstrap of the seed-mean RMSE/MAE difference
                          (Mamba - CNN), paired on the same test samples
    rmse_by_seed.png

Every prediction file is aligned to one chronological test order and its saved
targets are checked against the reference before any metric is computed
(baseline_lstm_cnn was evaluated on 2 GPUs and is saved rank-interleaved).
The bias correction subtracts each run's own validation mean error, as in
scripts/analysis_bias_and_sky_conditions.py; CNN validation predictions come
from scripts/evaluate_mamba_lstm.py --model-type cnn_lstm --splits val.
Existing CNN files are only read.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments"
OUT = EXP / "mamba_lstm" / "comparison"

CNN_RUNS = [  # label, experiment folder (read-only)
    ("cnn_lstm_seed0", EXP / "baseline_lstm_cnn"),
    ("cnn_lstm_seed1", EXP / "cnn_lstm_seed1"),
    ("cnn_lstm_seed2", EXP / "cnn_lstm_seed2"),
]
MAMBA_SEEDS = [0, 1, 2]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def metrics(y, p):
    e = p - y
    return {"RMSE": float(np.sqrt(np.mean(e ** 2))), "MAE": float(np.mean(np.abs(e))),
            "MBE": float(np.mean(e)), "R2": float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2))}


def align(path, y_ref, label):
    """Predictions from `path` in reference order; checks saved targets."""
    d = np.load(path)
    y, p = d["ghi_clipped_true"].astype(np.float64), d["ghi_clipped_pred"].astype(np.float64)
    n = len(y_ref)
    if len(y) != n:
        raise RuntimeError(f"{label}: {len(y)} samples, expected {n}")
    for world in range(1, 5):
        order = np.concatenate([np.arange(r, n, world) for r in range(world)])
        inv = np.empty(n, dtype=np.int64)
        inv[order] = np.arange(n)
        if np.allclose(y[inv], y_ref, atol=0.05):
            print(f"{label}: aligned ({world}-GPU save order), max |target diff| "
                  f"{np.max(np.abs(y[inv] - y_ref)):.4f} W/m2")
            return p[inv]
    raise RuntimeError(f"{label}: saved targets do not match the reference order")


def training_info(run_dir: Path):
    hist = run_dir / "logs" / "training_history.csv"
    if not hist.exists():
        return {}
    rows = list(csv.DictReader(open(hist)))
    best = min(rows, key=lambda r: float(r["val_loss"]))
    return {"epochs_run": len(rows), "best_epoch": int(best["epoch"]),
            "train_hours": sum(float(r["epoch_seconds"]) for r in rows) / 3600}


def clearsky_persistence_rmse():
    for row in csv.DictReader(open(EXP / "comparison" / "model_comparison.csv")):
        if row["model"] == "clearsky_persistence":
            return float(row["RMSE_Wm2"])
    raise RuntimeError("clearsky_persistence row missing")


def main():
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    mamba = []
    for s in MAMBA_SEEDS:
        ev = EXP / "mamba_lstm" / f"seed{s}" / "evaluation"
        if (ev / "test_predictions.npz").exists():
            mamba.append((f"mamba_lstm_seed{s}", EXP / "mamba_lstm" / f"seed{s}", ev))
    if mamba:
        ref = np.load(mamba[0][2] / "test_predictions.npz")
        y_ref, times = ref["ghi_clipped_true"].astype(np.float64), ref["sample_time"]
    else:  # no Mamba run evaluated yet: a 1-GPU CNN evaluation is in time order
        y_ref = np.load(EXP / "cnn_lstm_seed1/evaluation/test_predictions.npz")["ghi_clipped_true"].astype(np.float64)
        times = None
        print("No evaluated Mamba run yet; reporting the CNN runs only.")

    cs_rmse = clearsky_persistence_rmse()
    runs = []  # (model, label, test_pred, val_bias or None, info)
    for label, run_dir in CNN_RUNS:
        pred = align(run_dir / "evaluation" / "test_predictions.npz", y_ref, label)
        vpath = OUT / "cnn_validation" / label / "val_predictions.npz"
        bias = None
        if vpath.exists():
            v = np.load(vpath)
            bias = float(np.mean(v["ghi_clipped_pred"].astype(np.float64) - v["ghi_clipped_true"]))
        runs.append(("cnn_lstm", label, pred, bias, training_info(run_dir)))
    for label, run_dir, ev in mamba:
        pred = align(ev / "test_predictions.npz", y_ref, label)
        vpath = ev / "val_predictions.npz"
        bias = None
        if vpath.exists():
            v = np.load(vpath)
            bias = float(np.mean(v["ghi_clipped_pred"].astype(np.float64) - v["ghi_clipped_true"]))
        runs.append(("mamba_lstm", label, pred, bias, training_info(run_dir)))

    rows = []
    for model, label, pred, bias, info in runs:
        for corr, p in (("raw", pred), ("val_bias_corrected", None if bias is None else pred - bias)):
            if p is None:
                continue
            m = metrics(y_ref, p)
            rows.append({"model": model, "run": label, "correction": corr,
                         "val_bias_Wm2": "" if bias is None else f"{bias:.4f}", "n": len(y_ref),
                         **{k: f"{v:.4f}" for k, v in m.items()},
                         "FS_vs_clearsky_persistence": f"{1 - m['RMSE'] / cs_rmse:.4f}",
                         **{k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in info.items()}})
    cols = ["model", "run", "correction", "val_bias_Wm2", "n", "RMSE", "MAE", "MBE", "R2",
            "FS_vs_clearsky_persistence", "epochs_run", "best_epoch", "train_hours"]
    with open(OUT / "per_run_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(rows)

    summary = []
    for model in ("cnn_lstm", "mamba_lstm"):
        for corr in ("raw", "val_bias_corrected"):
            sel = [r for r in rows if r["model"] == model and r["correction"] == corr]
            if not sel:
                continue
            entry = {"model": model, "correction": corr, "n_seeds": len(sel)}
            for k in ("RMSE", "MAE", "MBE", "R2"):
                vals = np.array([float(r[k]) for r in sel])
                entry[f"{k}_mean"] = f"{vals.mean():.4f}"
                entry[f"{k}_sd"] = f"{vals.std(ddof=1):.4f}" if len(vals) > 1 else ""
            summary.append(entry)
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    for s in summary:
        print(s)

    # Day-block bootstrap of the paired difference in seed-mean RMSE / MAE.
    if mamba and times is not None:
        days = times.astype("datetime64[D]")
        _, day_idx = np.unique(days, return_inverse=True)
        n_days = day_idx.max() + 1
        count = np.bincount(day_idx, minlength=n_days).astype(np.float64)
        rng = np.random.default_rng(args.seed)
        draws = rng.integers(0, n_days, size=(args.bootstrap, n_days))
        weights = np.stack([np.bincount(d, minlength=n_days) for d in draws]).astype(np.float64)
        n_w = weights @ count
        boot_rows = []
        for corr in ("raw", "val_bias_corrected"):
            groups = {}
            for model, label, pred, bias, _ in runs:
                if corr == "val_bias_corrected":
                    if bias is None:
                        continue
                    pred = pred - bias
                err = pred - y_ref
                groups.setdefault(model, []).append(
                    (np.bincount(day_idx, err ** 2, n_days), np.bincount(day_idx, np.abs(err), n_days)))
            if len(groups) < 2:
                continue

            def stat(group, w=None):
                rm, ma = [], []
                for sq, ab in group:
                    if w is None:
                        rm.append(np.sqrt(sq.sum() / count.sum()))
                        ma.append(ab.sum() / count.sum())
                    else:
                        rm.append(np.sqrt((w @ sq) / n_w))
                        ma.append((w @ ab) / n_w)
                return np.mean(rm, axis=0), np.mean(ma, axis=0)

            point = np.subtract(stat(groups["mamba_lstm"]), stat(groups["cnn_lstm"]))
            boot = np.subtract(stat(groups["mamba_lstm"], weights), stat(groups["cnn_lstm"], weights))
            for i, name in enumerate(("RMSE", "MAE")):
                lo, hi = np.percentile(boot[i], [2.5, 97.5])
                boot_rows.append({"correction": corr, "metric": name,
                                  "n_mamba_seeds": len(groups["mamba_lstm"]),
                                  "n_cnn_seeds": len(groups["cnn_lstm"]), "n_days": n_days,
                                  "diff_mamba_minus_cnn": f"{point[i]:.4f}",
                                  "ci95_low": f"{lo:.4f}", "ci95_high": f"{hi:.4f}",
                                  "share_of_resamples_mamba_better": f"{np.mean(boot[i] < 0):.4f}"})
        if boot_rows:
            with open(OUT / "bootstrap.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(boot_rows[0].keys()))
                w.writeheader()
                w.writerows(boot_rows)
            for r in boot_rows:
                print(r)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.5))
        raw = [r for r in rows if r["correction"] == "raw"]
        colors = ["#2a78d6" if r["model"] == "cnn_lstm" else "#1b9e77" for r in raw]
        ax.bar([r["run"].replace("_lstm_", "\n") for r in raw], [float(r["RMSE"]) for r in raw], color=colors)
        ax.axhline(cs_rmse, color="#9a9893", ls="--", label=f"clear-sky persistence {cs_rmse:.2f}")
        lo = min(float(r["RMSE"]) for r in raw)
        ax.set_ylim(lo - 3, cs_rmse + 2)
        ax.set_ylabel("Test RMSE (W/m2), raw")
        ax.set_title("2016 test RMSE by run: CNN+LSTM (blue) vs Mamba-LSTM (green)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / "rmse_by_seed.png", dpi=160)
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: plot failed: {exc}")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
