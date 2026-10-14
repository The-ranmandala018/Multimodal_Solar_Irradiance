"""Compare the CNN+LSTM (sky image + weather) with the weather-only model.

Reads saved training histories, saved test predictions and the persistence
baselines. Does not run any model.

  (a) validation loss per epoch for both models, best epoch marked
  (b) test RMSE: naive persistence, clear-sky persistence, weather-only,
      CNN+LSTM, with forecast skill vs clear-sky persistence on each model

Also writes a CSV with every number shown.

Run from the repo root:
  python scripts/plot_model_comparison.py
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Colour follows the model in both panels.
MODELS = [
    # key, label, experiment dir, colour
    ("weather_only", "Weather-only LSTM", "experiments/weather_only_seed0", "#eb6834"),
    ("cnn_lstm", "CNN+LSTM (image + weather)", "experiments/baseline_lstm_cnn", "#2a78d6"),
]
BAR_LABELS = {"weather_only": "Weather-only\nLSTM", "cnn_lstm": "CNN+LSTM\n(image+weather)"}
# Best-epoch label offset (points): weather-only above its curve, CNN+LSTM below.
BEST_LABEL_OFFSET = {"weather_only": (0, 14, "bottom"), "cnn_lstm": (0, -30, "top")}

BASELINE_GREY = "#9a9893"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--baselines-csv",
        default="experiments/baseline_lstm_cnn/evaluation/full_metrics_summary.csv",
    )
    p.add_argument("--output-dir", default="experiments/comparison")
    return p.parse_args()


def style_axis(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelcolor=INK)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def to_time_order(y_true, y_pred, y_ref, label):
    """Return (y_true, y_pred) in the same sample order as y_ref.

    evaluate_test.py run on W GPUs saves rank 0's samples (0, W, 2W, ...),
    then rank 1's (1, W+1, ...), etc. -- not time order. A single-GPU run is
    already in time order. Try W = 1..4 and keep the one whose targets match.
    """
    n = len(y_ref)
    if len(y_true) != n:
        raise RuntimeError(f"{label}: {len(y_true)} test samples, expected {n}")
    for world in range(1, 5):
        saved_order = np.concatenate([np.arange(r, n, world) for r in range(world)])
        inverse = np.empty(n, dtype=np.int64)
        inverse[saved_order] = np.arange(n)
        if np.allclose(y_true[inverse], y_ref, atol=1e-3):
            if world > 1:
                print(f"{label}: predictions were saved by a {world}-GPU run; "
                      "reordered to time order")
            return y_true[inverse], y_pred[inverse]
    raise RuntimeError(f"{label}: test targets do not match the reference in any rank order")


def metrics(y_true, y_pred):
    err = np.asarray(y_pred, np.float64) - np.asarray(y_true, np.float64)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return {
        "MAE_Wm2": float(np.mean(np.abs(err))),
        "RMSE_Wm2": float(np.sqrt(np.mean(err ** 2))),
        "MBE_Wm2": float(np.mean(err)),
        "R2": float(1 - np.sum(err ** 2) / ss_tot),
    }


def main():
    args = parse_args()
    base = pd.read_csv(args.baselines_csv).set_index("set")
    cs_rmse = base.loc["clearsky_persistence", "RMSE_Wm2"]
    naive_rmse = base.loc["naive_persistence", "RMSE_Wm2"]

    histories, results, y_ref = {}, {}, None
    for key, label, exp_dir, _ in MODELS:
        histories[key] = pd.read_csv(Path(exp_dir) / "logs" / "training_history.csv")
        d = np.load(Path(exp_dir) / "evaluation" / "test_predictions.npz")
        y_true, y_pred = d["ghi_clipped_true"], d["ghi_clipped_pred"]
        # Both models must be scored on the same samples; put them in the
        # same (time) order. The weather-only run is single-GPU, so it is first.
        if y_ref is None:
            y_ref = y_true
        else:
            y_true, y_pred = to_time_order(y_true, y_pred, y_ref, key)
        m = metrics(y_true, y_pred)
        m["n"] = len(y_true)
        m["FS_vs_clearsky_persistence"] = 1 - m["RMSE_Wm2"] / cs_rmse
        m["FS_vs_naive_persistence"] = 1 - m["RMSE_Wm2"] / naive_rmse
        h = histories[key]
        best = h.loc[h["val_loss"].idxmin()]
        m["best_epoch"] = int(best["epoch"])
        m["best_val_loss"] = float(best["val_loss"])
        results[key] = m
        print(f"{label}: n={m['n']} RMSE={m['RMSE_Wm2']:.2f} MAE={m['MAE_Wm2']:.2f} "
              f"MBE={m['MBE_Wm2']:+.2f} R2={m['R2']:.4f} "
              f"FS(clear-sky)={m['FS_vs_clearsky_persistence']:.3f} "
              f"best epoch {m['best_epoch']} val {m['best_val_loss']:.5f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["n", "MAE_Wm2", "RMSE_Wm2", "MBE_Wm2", "R2", "FS_vs_clearsky_persistence",
            "FS_vs_naive_persistence", "best_epoch", "best_val_loss"]
    with open(out_dir / "model_comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model"] + cols)
        for key in ("naive_persistence", "clearsky_persistence"):
            r = base.loc[key]
            w.writerow([key, int(r["n"]), r["MAE_Wm2"], r["RMSE_Wm2"], r["MBE_Wm2"], r["R2"],
                        1 - r["RMSE_Wm2"] / cs_rmse, 1 - r["RMSE_Wm2"] / naive_rmse, "", ""])
        for key, *_ in MODELS:
            w.writerow([key] + [results[key][c] for c in cols])
    print(f"Saved: {out_dir / 'model_comparison.csv'}")

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.42},
    )

    # (a) validation loss per epoch
    for key, label, _, color in MODELS:
        h = histories[key]
        ax1.plot(h["epoch"], h["val_loss"], color=color, linewidth=2,
                 marker="o", markersize=5, label=label)
        b = results[key]
        ax1.plot(b["best_epoch"], b["best_val_loss"], marker="o", markersize=11,
                 markerfacecolor="none", markeredgecolor=color, markeredgewidth=2)
        dx, dy, va = BEST_LABEL_OFFSET[key]
        ax1.annotate(f"best {b['best_val_loss']:.4f}\n(epoch {b['best_epoch']})",
                     (b["best_epoch"], b["best_val_loss"]), textcoords="offset points",
                     xytext=(dx, dy), ha="center", va=va, color=INK, fontsize=9)
    ax1.set_xlabel("Epoch", color=INK)
    ax1.set_ylabel("Validation loss (MSE on clear-sky index k)", color=INK)
    ax1.set_title("(a) Validation loss during training", loc="left", color=INK, fontsize=11)
    lo = min(h["val_loss"].min() for h in histories.values())
    hi = max(h["val_loss"].max() for h in histories.values())
    ax1.set_ylim(lo - 0.25 * (hi - lo) - 0.001, hi + 0.08 * (hi - lo))
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax1.legend(frameon=False, loc="upper right", fontsize=9)
    style_axis(ax1)

    # (b) test RMSE
    bars = [("Naive\npersistence", naive_rmse, BASELINE_GREY, None),
            ("Clear-sky\npersistence", cs_rmse, BASELINE_GREY, None)]
    for key, label, _, color in MODELS:
        bars.append((BAR_LABELS[key], results[key]["RMSE_Wm2"], color,
                     results[key]["FS_vs_clearsky_persistence"]))
    for i, (name, v, color, fs) in enumerate(bars):
        ax2.bar(i, v, width=0.6, color=color, edgecolor="white", linewidth=2)
        text = f"{v:.1f}" if fs is None else f"{v:.1f}\nskill {100 * fs:+.1f}%"
        ax2.text(i, v + 1.5, text, ha="center", va="bottom", color=INK, fontsize=10)
    ax2.set_xticks(range(len(bars)), [b[0] for b in bars], fontsize=9)
    ax2.set_ylabel("Test RMSE (W/m²)", color=INK)
    ax2.set_ylim(0, max(b[1] for b in bars) * 1.25)
    ax2.set_title("(b) Test error, 2016", loc="left", color=INK, fontsize=11)
    style_axis(ax2)

    n = results[MODELS[0][0]]["n"]
    fig.suptitle("Does the sky image help? Same LSTM, same data, with and without the image",
                 color=INK, fontsize=12, x=0.01, y=0.99, ha="left")
    fig.text(0.01, 0.005,
             f"10-min GHI forecast, Folsom. Test: full 2016 set ({n:,} samples), same samples "
             "for every bar. Skill = 1 - RMSE / RMSE(clear-sky persistence).",
             color=INK_2, fontsize=8, ha="left", va="bottom")
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.17)

    for ext in ("png", "pdf"):
        out = out_dir / f"model_comparison.{ext}"
        fig.savefig(out, dpi=200, facecolor="white")
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
