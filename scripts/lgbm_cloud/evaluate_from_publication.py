"""Reproduce the reported metrics from the PUBLISHED files only (no raw data, no feature tables).

Inputs:
    experiments/lightgbm_cloud_features/publication/targets_<split>.parquet
    experiments/lightgbm_cloud_features/publication/predictions_<split>_<model>.parquet
    experiments/<existing run>/evaluation/test_predictions.npz   (already in the repository)
Checks the overall validation and test metrics and the test bootstrap of LGBM-B vs LGBM-A
and LGBM-B vs CNN+LSTM against summary/*.csv, and prints the comparison. Comparisons whose
inputs are not present in this repository version (e.g. CNN+LSTM seeds 1-2 / image-only
predictions before they are published) are reported as SKIP, never silently passed.

Usage (from the repository root; needs numpy, pandas, pyarrow):
    python scripts/lgbm_cloud/evaluate_from_publication.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "lightgbm_cloud_features"
PUB, SUM = EXP / "publication", EXP / "summary"
MODELS = {"LGBM-A": "lgbm_A", "LGBM-A0": "lgbm_A0", "LGBM-B": "lgbm_B", "LGBM-C": "lgbm_C",
          "LGBM-B-parity": "lgbm_B_parity", "Ridge-A": "ridge_A", "Ridge-B": "ridge_B"}
EXISTING = {"Weather-only LSTM": ["weather_only_seed0", "weather_only_seed1", "weather_only_seed2"],
            "CNN+LSTM": ["baseline_lstm_cnn", "cnn_lstm_seed1", "cnn_lstm_seed2"], "Image-only CNN": ["image_only_seed0"]}
TOL = 1e-9


def metrics(y, p):
    e = p - y
    return {"RMSE_Wm2": float(np.sqrt(np.mean(e ** 2))), "MAE_Wm2": float(np.mean(np.abs(e))),
            "MBE_Wm2": float(np.mean(e)), "R2": 1 - float(np.sum(e ** 2)) / float(np.sum((y - y.mean()) ** 2))}


def ghi(split, key):
    t = pd.read_parquet(PUB / f"targets_{split}.parquet")
    k = pd.read_parquet(PUB / f"predictions_{split}_{key}.parquet")
    j = t.merge(k, on="split_pos", validate="one_to_one").sort_values("split_pos").reset_index(drop=True)
    cols = [c for c in j.columns if c.startswith("k_pred__")]
    g = np.mean([j[c].to_numpy() * j["ghi_cs_t10"].to_numpy() for c in cols], axis=0)
    return j, g


def existing(y, run):
    d = np.load(ROOT / "experiments" / run / "evaluation" / "test_predictions.npz")
    yt, yp = d["ghi_clipped_true"].astype(np.float64), d["ghi_clipped_pred"].astype(np.float64)
    n = len(y)
    for world in range(1, 5):  # undo multi-GPU rank interleaving, as analysis_bias_and_sky_conditions.py
        order = np.concatenate([np.arange(r, n, world) for r in range(world)])
        inv = np.empty(n, dtype=np.int64)
        inv[order] = np.arange(n)
        if np.allclose(yt[inv], y, atol=0.05):
            return yp[inv]
    raise RuntimeError(f"{run}: targets do not match")


def boot(a, b, y, days, n=2000, seed=0):
    ea, eb = a - y, b - y
    g = pd.DataFrame({"d": days, "aa": np.abs(ea), "ab": np.abs(eb), "sa": ea ** 2, "sb": eb ** 2})
    per = g.groupby("d").agg(n=("aa", "size"), aa=("aa", "sum"), ab=("ab", "sum"), sa=("sa", "sum"), sb=("sb", "sum"))
    idx = np.random.default_rng(seed).integers(0, len(per), size=(n, len(per)))
    s = per.to_numpy()[idx].sum(axis=1)
    mae = (s[:, 1] - s[:, 2]) / s[:, 0]
    rmse = np.sqrt(s[:, 3] / s[:, 0]) - np.sqrt(s[:, 4] / s[:, 0])
    return {"dMAE": float(np.abs(ea).mean() - np.abs(eb).mean()), "dMAE_lo": float(np.quantile(mae, .025)),
            "dMAE_hi": float(np.quantile(mae, .975)), "dRMSE": float(np.sqrt((ea ** 2).mean()) - np.sqrt((eb ** 2).mean())),
            "dRMSE_lo": float(np.quantile(rmse, .025)), "dRMSE_hi": float(np.quantile(rmse, .975))}


def check(label, got, ref, out):
    d = max(abs(got[k] - ref[k]) for k in got)
    out.append((label, d <= TOL, d))


def main():
    out = []
    val_rep = pd.read_csv(SUM / "val_metrics_overall.csv").set_index("run")
    for name, key in MODELS.items():
        j, g = ghi("val", key)
        m = j["in_purged_val"].to_numpy()
        check(f"val {name}", metrics(j["ghi_true"].to_numpy()[m], g[m]), val_rep.loc[name], out)

    test_rep = pd.read_csv(SUM / "test_metrics_overall.csv").set_index("model")
    preds = {}
    for name, key in MODELS.items():
        j, g = ghi("test", key)
        preds[name] = g
        check(f"test {name}", metrics(j["ghi_true"].to_numpy(), g), test_rep.loc[name], out)
    y = j["ghi_true"].to_numpy()
    days = pd.to_datetime(j["datetime"]).dt.date.to_numpy()
    skipped = []
    for name, runs in EXISTING.items():
        missing = [r for r in runs if not (ROOT / "experiments" / r / "evaluation" / "test_predictions.npz").exists()]
        if missing:  # e.g. results not yet published in this repository version
            skipped.append(f"test {name} (missing {', '.join(missing)})")
            continue
        preds[name] = np.mean([existing(y, r) for r in runs], axis=0)
        check(f"test {name}", metrics(y, preds[name]), test_rep.loc[name], out)
    check("test Clear-sky persistence", metrics(y, j["clearsky_persistence"].to_numpy()), test_rep.loc["Clear-sky persistence"], out)

    b_rep = pd.read_csv(SUM / "test_paired_bootstrap.csv")
    for a, b in (("LGBM-B", "LGBM-A"), ("LGBM-B", "CNN+LSTM"), ("LGBM-B-parity", "LGBM-B")):
        if a not in preds or b not in preds:
            skipped.append(f"bootstrap {a} - {b} (inputs missing)")
            continue
        ref = b_rep[(b_rep["comparison"] == f"{a} - {b}") & (b_rep["subset"] == "all")].iloc[0]
        check(f"bootstrap {a} - {b}", boot(preds[a], preds[b], y, days), ref, out)

    for label, ok, d in out:
        print(f"{'PASS' if ok else 'FAIL'}  {label}  (max |diff| {d:.2e})")
    for sk in skipped:
        print(f"SKIP  {sk}")
    print(f"{sum(o[1] for o in out)}/{len(out)} reproduced; {len(skipped)} skipped (inputs not in this repository version)")
    sys.exit(0 if all(o[1] for o in out) else 1)


if __name__ == "__main__":
    main()
