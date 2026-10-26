"""Step 1: corrected image index for the LightGBM cloud-feature experiment.

Folsom image file names are stamped by a drifting camera clock (Phase 1;
Varaschin & Silva, arXiv:2503.21966). The file modified time (st_mtime, UTC epoch)
agrees with the irradiance clock. Per-file mtime has some jitter, so the image
time used here is

    t_img = file-name time - median of (file-name time - mtime) over the image's clock segment

A clock segment is a day's folder, split where the camera clock was reset mid-day.

Output: experiments/lightgbm_cloud_features/features/image_index.parquet
    path, day_dir, ts_name, ts_mtime, day_median_offset_s, clock_segment, offset_s, t_img,
    resid_vs_mtime_s, sza_t_img, corrupted, day_offset_jump_flag

Usage:
    PYTHONPATH=. python scripts/lgbm_cloud/build_image_index.py
"""

from __future__ import annotations

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pvlib

from common import (ALTITUDE, CORRUPTED_LIST, DATA_ROOT, FEATURES_DIR, LATITUDE,
                    LONGITUDE, git_head, refuse_existing, versions, write_json)

OUT_FILE = FEATURES_DIR / "image_index.parquet"
SUMMARY_FILE = FEATURES_DIR / "image_index_summary.json"
RUN_MEDIAN_N = 15      # images (~15 min) in the running median
SEGMENT_JUMP_S = 60    # running-median jump that starts a new clock segment


def scan() -> pd.DataFrame:
    rows = []
    for year in ("2014", "2015", "2016"):
        for day_dir in sorted(DATA_ROOT.glob(f"{year}/[0-1][0-9]/[0-3][0-9]")):
            with os.scandir(day_dir) as it:
                for e in it:
                    if not e.name.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue
                    try:
                        ts = datetime.strptime(os.path.splitext(e.name)[0], "%Y%m%d_%H%M%S")
                    except ValueError:
                        continue
                    rows.append((str(day_dir / e.name), str(day_dir.relative_to(DATA_ROOT)),
                                 ts, e.stat().st_mtime))
    df = pd.DataFrame(rows, columns=["path", "day_dir", "ts_name", "mtime_epoch"])
    df["ts_mtime"] = pd.to_datetime(df["mtime_epoch"], unit="s")  # UTC, tz-naive
    return df.drop(columns="mtime_epoch")


def main():
    refuse_existing(OUT_FILE)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    img = scan()
    img = img.sort_values(["day_dir", "ts_name"]).reset_index(drop=True)
    img["name_minus_mtime_s"] = (img["ts_name"] - img["ts_mtime"]).dt.total_seconds()
    img["day_median_offset_s"] = img.groupby("day_dir")["name_minus_mtime_s"].transform("median")

    # The camera clock is occasionally reset in the middle of a day (e.g. 2016-09-13
    # 20:34 UTC, +637 s). Split each day at points where the running median of
    # (name - mtime) jumps by more than SEGMENT_JUMP_S, and correct each segment with
    # its own median. A running median ignores single files with a late mtime.
    run_med = img.groupby("day_dir")["name_minus_mtime_s"].transform(
        lambda s: s.rolling(RUN_MEDIAN_N, center=True, min_periods=1).median())
    new_seg = (run_med.diff().abs() > SEGMENT_JUMP_S) | img["day_dir"].ne(img["day_dir"].shift())
    img["clock_segment"] = new_seg.cumsum()
    img["offset_s"] = img.groupby("clock_segment")["name_minus_mtime_s"].transform("median")
    img["t_img"] = img["ts_name"] - pd.to_timedelta(img["offset_s"], unit="s")
    img["resid_vs_mtime_s"] = (img["t_img"] - img["ts_mtime"]).dt.total_seconds()

    # Flag days whose median offset jumps away from the neighbouring days
    # (a whole-day mtime problem would show up here, not in the per-file residual).
    day = img.groupby("day_dir")["day_median_offset_s"].first().sort_index()
    rolling = day.rolling(7, center=True, min_periods=3).median()
    jump = (day - rolling).abs() > 60
    img["day_offset_jump_flag"] = img["day_dir"].map(jump).astype(bool)

    corrupted = set()
    if CORRUPTED_LIST.exists():
        corrupted = {line.strip() for line in CORRUPTED_LIST.read_text().splitlines() if line.strip()}
    img["corrupted"] = img["path"].isin(corrupted)

    site = pvlib.location.Location(LATITUDE, LONGITUDE, altitude=ALTITUDE)
    img["sza_t_img"] = site.get_solarposition(pd.DatetimeIndex(img["t_img"]))["zenith"].to_numpy()

    img = img.sort_values("t_img").reset_index(drop=True)
    img = img[["path", "day_dir", "ts_name", "ts_mtime", "day_median_offset_s", "clock_segment", "offset_s", "t_img",
               "resid_vs_mtime_s", "sza_t_img", "corrupted", "day_offset_jump_flag"]]

    t_sorted = img["t_img"].is_monotonic_increasing
    dup_t = int(img["t_img"].duplicated().sum())
    img.to_parquet(OUT_FILE, index=False)

    resid = img["resid_vs_mtime_s"].abs()
    summary = {
        "git_head": git_head(),
        "versions": versions(),
        "n_images": int(len(img)),
        "n_days": int(img["day_dir"].nunique()),
        "n_corrupted_listed": len(corrupted),
        "n_corrupted_matched": int(img["corrupted"].sum()),
        "n_clock_segments": int(img["clock_segment"].nunique()),
        "days_with_midday_reset": sorted(img.groupby("day_dir")["clock_segment"].nunique().loc[lambda x: x > 1].index.tolist()),
        "n_days_offset_jump_flag": int(jump.sum()),
        "days_offset_jump_flag": sorted(jump[jump].index.tolist()),
        "t_img_duplicates": dup_t,
        "t_img_sorted": bool(t_sorted),
        "abs_resid_vs_mtime_quantiles_s": {str(q): float(resid.quantile(q)) for q in (0.5, 0.9, 0.99, 0.999, 1.0)},
        "day_median_offset_by_month_s": (
            img.groupby(img["day_dir"].str[:7])["day_median_offset_s"].median().round(1).to_dict()
        ),
        "seconds": round(time.time() - t0, 1),
    }
    write_json(SUMMARY_FILE, summary)
    print(f"Saved {OUT_FILE} ({len(img)} images) and {SUMMARY_FILE}")
    for k in ("n_images", "n_corrupted_matched", "n_clock_segments", "days_with_midday_reset", "n_days_offset_jump_flag", "t_img_duplicates",
              "abs_resid_vs_mtime_quantiles_s", "seconds"):
        print(f"  {k}: {summary[k]}")


if __name__ == "__main__":
    main()
