"""Plot how the trained CNN+LSTM's error changes when only the sky image changes.

Reads the saved image-ablation metrics (real / shuffled / zero image) and the
persistence baselines. Does not run the model.

Outputs (next to the existing baseline plots):
    - image_ablation_<tag>.png
    - image_ablation_<tag>.pdf

Run from the repo root:
  python scripts/plot_image_ablation.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

LABELS = {
    "real": "Real image",
    "shuffled": "Shuffled image\n(another time)",
    "zero": "Blank image\n(all zeros)",
}
BAR = "#2a78d6"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ablation-csv",
        default="experiments/baseline_lstm_cnn/evaluation/image_ablation_metrics_subset20000.csv",
    )
    p.add_argument(
        "--baselines-csv",
        default="experiments/baseline_lstm_cnn/evaluation/full_metrics_summary.csv",
    )
    p.add_argument("--output-dir", default="experiments/baseline_lstm_cnn/plots")
    return p.parse_args()


def style_axis(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelcolor=INK)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    args = parse_args()
    abl = pd.read_csv(args.ablation_csv).set_index("mode").loc[["real", "shuffled", "zero"]]
    base = pd.read_csv(args.baselines_csv).set_index("set")
    n = int(abl["n"].iloc[0])
    tag = Path(args.ablation_csv).stem.replace("image_ablation_metrics_", "")

    x = range(len(abl))
    names = [LABELS[m] for m in abl.index]
    rmse_real = abl.loc["real", "RMSE_Wm2"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.55},
    )

    # (a) RMSE with persistence baselines as reference lines
    rmse = abl["RMSE_Wm2"]
    ax1.bar(x, rmse, width=0.55, color=BAR, edgecolor="white", linewidth=2)
    for i, (mode, v) in enumerate(rmse.items()):
        text = f"{v:.1f}" if mode == "real" else f"{v:.1f}\n(+{100 * (v - rmse_real) / rmse_real:.0f}%)"
        ax1.text(i, v + 3, text, ha="center", va="bottom", color=INK, fontsize=10)
    for key, name in (("naive_persistence", "Naive persistence"),
                      ("clearsky_persistence", "Clear-sky persistence")):
        y = base.loc[key, "RMSE_Wm2"]
        ax1.axhline(y, color=INK_2, linestyle="--", linewidth=1)
        # Label outside the plot, right of the axis, so it never sits on a bar.
        ax1.text(1.01, y, f"{name}\n{y:.1f}", transform=ax1.get_yaxis_transform(),
                 ha="left", va="bottom" if key == "naive_persistence" else "top",
                 color=INK_2, fontsize=8.5, linespacing=1.1)
    ax1.set_xticks(list(x), names)
    ax1.set_ylabel("RMSE (W/m²)", color=INK)
    ax1.set_ylim(0, rmse.max() * 1.22)
    ax1.set_title("(a) Forecast error", loc="left", color=INK, fontsize=11)
    style_axis(ax1)

    # (b) Bias
    mbe = abl["MBE_Wm2"]
    ax2.bar(x, mbe, width=0.55, color=BAR, edgecolor="white", linewidth=2)
    for i, v in enumerate(mbe):
        ax2.text(i, v - 2, f"{v:+.1f}", ha="center", va="top", color=INK, fontsize=10)
    ax2.axhline(0, color=INK_2, linewidth=1)
    ax2.set_xticks(list(x), names)
    ax2.set_ylabel("Mean bias error (W/m²)", color=INK)
    ax2.set_ylim(mbe.min() * 1.25, 10)
    ax2.set_title("(b) Bias (negative = under-prediction)", loc="left", color=INK, fontsize=11)
    style_axis(ax2)

    fig.suptitle(
        "Same trained CNN+LSTM, same weather input — only the sky image changes",
        color=INK, fontsize=12, x=0.01, y=0.985, ha="left",
    )
    fig.text(
        0.01, 0.005,
        f"10-min GHI forecast, Folsom 2016 test set, {n:,} random samples (seed 0). "
        "Persistence baselines are on the full test set (224,022 samples).",
        color=INK_2, fontsize=8, ha="left", va="bottom",
    )
    fig.subplots_adjust(left=0.065, right=0.985, top=0.83, bottom=0.2)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = out_dir / f"image_ablation_{tag}.{ext}"
        fig.savefig(out, dpi=200, facecolor="white")
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
