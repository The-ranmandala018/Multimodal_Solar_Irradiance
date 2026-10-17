"""Step 2: sample table and feature set A (no images) for every train/val/test sample.

The samples, split and 40 x 7 history window come from the existing code:
FolsomDataset (load_images=False) + chronological_year_split, and the window is
taken from FolsomDataset.__getitem__ BEFORE set_normalization, so the gap filling
(reindex nearest 10 min -> ffill -> bfill -> 0) is exactly the one the LSTM sees.

Image columns for the later cloud features:
    img_le_t   latest non-corrupted image with t_img <= t and t_img >= t - 120 s
               (t_img = corrected image time from build_image_index.py)
    img_cnn    the image FolsomDataset matched for the CNN (file-name time, +-120 s)

Outputs (experiments/lightgbm_cloud_features/features/):
    samples_setA.parquet, windows_40x7.npy (raw, unnormalised), samples_setA_summary.json

Usage:
    PYTHONPATH=. python scripts/lgbm_cloud/build_tabular_features.py [--workers 40]
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import (EXPECTED_SPLITS, FEATURES_DIR, HORIZON_MIN, IMAGE_MAX_AGE_S, PROJECT_DIR,
                    SEQ_LEN, WINDOW_COLS, dataset_config, git_head, refuse_existing, versions,
                    write_json)

OUT_TABLE = FEATURES_DIR / "samples_setA.parquet"
OUT_WINDOWS = FEATURES_DIR / "windows_40x7.npy"
OUT_SUMMARY = FEATURES_DIR / "samples_setA_summary.json"
IMAGE_INDEX = FEATURES_DIR / "image_index.parquet"
SKY_WINDOW_MIN = 30  # same as scripts/analysis_bias_and_sky_conditions.py

_DS = None  # dataset shared with forked workers


def _window(i):
    _, weather, target, cs = _DS[i]
    return i, weather.numpy(), float(target[0]), float(cs[0])


def build_windows(n, workers):
    import torch
    torch.set_num_threads(1)
    windows = np.empty((n, SEQ_LEN, len(WINDOW_COLS)), np.float32)
    k_t10 = np.empty(n, np.float64)
    cs_t10 = np.empty(n, np.float64)
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        for i, w, k, c in pool.imap_unordered(_window, range(n), chunksize=2000):
            windows[i], k_t10[i], cs_t10[i] = w, k, c
    return windows, k_t10, cs_t10


def set_a_features(w: np.ndarray) -> pd.DataFrame:
    """Feature set A from the raw 40 x 7 window (row 39 = t, row 0 = t - 39 min)."""
    k, temp, pres = w[:, :, 0], w[:, :, 1], w[:, :, 2]
    f = {}
    for lag in (0, 1, 2, 5, 10, 20, 39):
        f[f"k_lag{lag}"] = k[:, 39 - lag]
    for span in (10, 20, 40):
        f[f"k_mean{span}"] = k[:, 40 - span:].mean(axis=1)
        f[f"k_std{span}"] = k[:, 40 - span:].std(axis=1)
    f["k_diff10"] = k[:, 39] - k[:, 29]
    f["temp_t"] = temp[:, 39]
    f["pres_t"] = pres[:, 39]
    f["temp_d30"] = temp[:, 39] - temp[:, 9]
    f["pres_d30"] = pres[:, 39] - pres[:, 9]
    f["sza_t"] = w[:, 39, 3]
    az = np.deg2rad(w[:, 39, 4])
    f["az_sin"] = np.sin(az)
    f["az_cos"] = np.cos(az)
    f["sin_hour"] = w[:, 39, 5]
    f["cos_hour"] = w[:, 39, 6]
    return pd.DataFrame(f).astype(np.float32)


def sky_and_persistence(times: pd.DatetimeIndex, target_df: pd.DataFrame) -> pd.DataFrame:
    """Same rule as analysis_bias_and_sky_conditions.test_table, for every sample."""
    tdf = target_df[~target_df.index.duplicated(keep="first")]
    k, cs, ghi, sza = tdf["k_index"], tdf["GHI_cs"], tdf["GHI"], tdf["SZA"]
    t10 = times + pd.Timedelta(minutes=HORIZON_MIN)
    win = np.column_stack([k.reindex(times - pd.Timedelta(minutes=m)).to_numpy()
                           for m in range(SKY_WINDOW_MIN - 1, -1, -1)])
    k_mean, k_std = np.nanmean(win, axis=1), np.nanstd(win, axis=1)
    sky = np.full(len(times), "mixed", dtype=object)
    sky[(k_std < 0.05) & (k_mean > 0.9)] = "clear"
    sky[(k_std < 0.10) & (k_mean < 0.5)] = "overcast"
    return pd.DataFrame({
        "k_t_raw": k.reindex(times).to_numpy(),
        "ghi_t": ghi.reindex(times).to_numpy(),
        "k_true_ref": k.reindex(t10).to_numpy(),
        "ghi_cs_t10_ref": cs.reindex(t10).to_numpy(),
        "sza_t10": sza.reindex(t10).to_numpy(),
        "k_mean_30min": k_mean, "k_std_30min": k_std,
        "sky_window_minutes": np.sum(~np.isnan(win), axis=1),
        "sky_class": sky,
    })


def latest_image_at_or_before(times: pd.DatetimeIndex, index: pd.DataFrame):
    ok = index[~index["corrupted"]].sort_values("t_img")
    t_img = ok["t_img"].to_numpy()
    pos = np.searchsorted(t_img, times.to_numpy(), side="right") - 1
    valid = pos >= 0
    age = np.full(len(times), np.nan)
    age[valid] = (times.to_numpy()[valid] - t_img[pos[valid]]) / np.timedelta64(1, "s")
    valid &= age <= IMAGE_MAX_AGE_S
    path = np.full(len(times), None, dtype=object)
    path[valid] = ok["path"].to_numpy()[pos[valid]]
    age[~valid] = np.nan
    return path, age


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=40)
    args = parser.parse_args()

    for p in (OUT_TABLE, OUT_WINDOWS, OUT_SUMMARY):
        refuse_existing(p)
    if not IMAGE_INDEX.exists():
        raise FileNotFoundError(f"{IMAGE_INDEX} missing; run build_image_index.py first")
    t0 = time.time()

    global _DS
    from datasets.chrono import chronological_year_split
    from datasets.folsom_dataset import FolsomDataset

    _DS = FolsomDataset(dataset_config(load_images=False))
    assert _DS.mean is None, "window must be taken before normalisation"
    samples = _DS.samples
    n = len(samples)
    train_idx, val_idx, test_idx = chronological_year_split(samples, 0.1, (2014, 2015), 2016)
    sizes = {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)}
    if sizes != EXPECTED_SPLITS:
        raise RuntimeError(f"split sizes {sizes} != expected {EXPECTED_SPLITS}")

    times = pd.DatetimeIndex([s[1] for s in samples])
    if not times.is_monotonic_increasing:
        raise RuntimeError("samples are not in time order")
    split = np.empty(n, dtype=object)
    split_pos = np.empty(n, np.int64)
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        split[idx] = name
        split_pos[idx] = np.arange(len(idx))

    print(f"Building {n} windows with {args.workers} workers ...", flush=True)
    windows, k_t10, cs_t10 = build_windows(n, args.workers)
    print(f"  done in {time.time() - t0:.0f} s", flush=True)

    table = pd.DataFrame({"datetime": times, "split": split, "split_pos": split_pos})
    table = pd.concat([table, set_a_features(windows), sky_and_persistence(times, _DS.target_df)], axis=1)
    table["ghi_cs_t10"] = cs_t10
    table["k_true"] = k_t10
    table["ghi_true"] = k_t10 * cs_t10           # = evaluate_test.py's ghi_clipped_true
    table["clearsky_persistence"] = table["k_t_raw"] * cs_t10
    table["test_period"] = ""
    is_test = table["split"] == "test"
    table.loc[is_test, "test_period"] = np.where(table.loc[is_test, "datetime"].dt.month <= 8, "jan_aug", "sep_dec")

    # Image columns.
    index = pd.read_parquet(IMAGE_INDEX)
    table["img_le_t"], table["img_le_t_age_s"] = latest_image_at_or_before(times, index)
    table["has_image"] = table["img_le_t"].notna()
    table["img_cnn"] = [s[0] for s in samples]
    t_img_of = index.set_index("path")["t_img"]
    table["img_cnn_true_offset_s"] = (table["img_cnn"].map(t_img_of) - table["datetime"]).dt.total_seconds()

    # ---- checks -----------------------------------------------------------
    checks = {}
    checks["k_true_eq_target_df"] = float(np.max(np.abs(table["k_true"] - table["k_true_ref"])))
    checks["cs_t10_eq_target_df"] = float(np.max(np.abs(table["ghi_cs_t10"] - table["ghi_cs_t10_ref"])))
    checks["k_lag0_eq_k_t_raw"] = float(np.nanmax(np.abs(table["k_lag0"] - table["k_t_raw"])))
    checks["img_le_t_age_max_s"] = float(table["img_le_t_age_s"].max())
    checks["img_le_t_age_min_s"] = float(table["img_le_t_age_s"].min())
    checks["img_cnn_unmapped"] = int(table["img_cnn_true_offset_s"].isna().sum())
    test = table[is_test]
    saved = np.load(PROJECT_DIR / "experiments/weather_only_seed0/evaluation/test_predictions.npz")
    checks["test_k_true_eq_saved_weather_only_seed0"] = float(np.max(np.abs(test["k_true"].to_numpy() - saved["k_index_true"])))
    checks["test_ghi_true_eq_saved_weather_only_seed0"] = float(np.max(np.abs(test["ghi_true"].to_numpy() - saved["ghi_clipped_true"])))
    for name, tol in (("k_true_eq_target_df", 1e-6), ("cs_t10_eq_target_df", 1e-4), ("k_lag0_eq_k_t_raw", 1e-5),
                      ("test_k_true_eq_saved_weather_only_seed0", 1e-5), ("test_ghi_true_eq_saved_weather_only_seed0", 0.05)):
        if not checks[name] <= tol:
            raise RuntimeError(f"check failed: {name} = {checks[name]} > {tol}")
    if checks["img_le_t_age_min_s"] < 0:
        raise RuntimeError("an img_le_t image is after t")
    table = table.drop(columns=["k_true_ref", "ghi_cs_t10_ref"])

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUT_TABLE, index=False)
    np.save(OUT_WINDOWS, windows)

    per_split = {}
    for name, g in table.groupby("split"):
        per_split[name] = {
            "n": int(len(g)), "first": str(g["datetime"].min()), "last": str(g["datetime"].max()),
            "has_image_frac": float(g["has_image"].mean()),
            "img_le_t_age_median_s": float(g["img_le_t_age_s"].median()),
            "img_cnn_true_offset_median_s": float(g["img_cnn_true_offset_s"].median()),
            "sky_class_frac": g["sky_class"].value_counts(normalize=True).round(4).to_dict(),
        }
    write_json(OUT_SUMMARY, {
        "git_head": git_head(), "versions": versions(), "checks": checks, "splits": per_split,
        "columns": list(table.columns), "seconds": round(time.time() - t0, 1),
    })
    print(f"Saved {OUT_TABLE}, {OUT_WINDOWS}, {OUT_SUMMARY}")
    print("checks:", checks)
    for name, s in per_split.items():
        print(name, s)


if __name__ == "__main__":
    main()
