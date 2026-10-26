"""Step 4: extract cloud features for every daytime image (all splits), once per image.

Images: all non-corrupted images with SZA <= 87 deg at t_img, plus every image the
CNN pipeline matched (img_cnn), so both the new rule (img_le_t) and the parity run
(img_cnn) can be joined later. Work is split by day folder; within a day images are
processed in t_img order so motion can use the previous frame.

Full runs are checkpointed per day folder (checkpoint.py): each verified day is saved to
features/cloud_parts_v1/ with a ledger record, and a rerun resumes, skipping verified days
and redoing missing, partial or corrupted ones. The final file is assembled only when every
day is complete, then verified as a whole (verify_cloud_features.py).

Outputs:
    full : features/cloud_features.parquet   (one row per image, keyed by path)
    trial: checks/cloud_trial.parquet + checks/cloud_trial_timing.json
           (--trial: 10 random TRAIN days x first 100 daytime images = 1000 images)

Usage:
    PYTHONPATH=. python scripts/lgbm_cloud/extract_cloud_features.py --trial
    PYTHONPATH=. python scripts/lgbm_cloud/extract_cloud_features.py --workers 48
    (rerun the same command after an interruption to resume)
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

from checkpoint import ChunkStore, chunk_key, frozen_config_id
from cloud_features import ALL_FEATURES, MonthContext, compute, library
from common import (ALTITUDE, CHECKS_DIR, FEATURES_DIR, LATITUDE, LONGITUDE, git_head, refuse_existing,
                    versions, write_json)
from sky_geometry import SIZE, read_small_rgb

OUT_FULL = FEATURES_DIR / "cloud_features.parquet"
OUT_TRIAL = CHECKS_DIR / "cloud_trial.parquet"
PARTS_DIR = FEATURES_DIR / "cloud_parts_v1"
MAX_SZA = 87.0
TRAIN_END = pd.Timestamp("2015-10-04 20:09:00")  # last train sample (Phase 1)

_SITE = pvlib.location.Location(LATITUDE, LONGITUDE, altitude=ALTITUDE)
_CTX = {}


def _ctx(year, month):
    if (year, month) not in _CTX:
        _CTX[(year, month)] = MonthContext(year, month)
    return _CTX[(year, month)]


def process_day(day: pd.DataFrame) -> pd.DataFrame:
    import cv2
    cv2.setNumThreads(1)  # one thread per worker: 64 workers x OpenCV's own pool overloaded the server
    library()  # load once per worker
    day = day.sort_values("t_img")
    sp = _SITE.get_solarposition(pd.DatetimeIndex(day["t_img"]))
    rows, prev = [], None
    for path, t_img, zen, az in zip(day["path"], day["t_img"], sp["apparent_zenith"], sp["azimuth"]):
        t0 = time.perf_counter()
        rgb = read_small_rgb(path)
        row = {"path": path, "t_img": t_img, "sun_zen": float(zen), "sun_az": float(az), "read_ok": rgb is not None}
        if rgb is None or rgb.shape[:2] != (SIZE, SIZE):
            row["read_ok"] = False
            rows.append(row)
            prev = None
            continue
        feats, prev = compute(rgb, _ctx(t_img.year, t_img.month), float(zen), float(az), t_img, prev)
        row.update(feats)
        row["ms"] = (time.perf_counter() - t0) * 1000
        rows.append(row)
    return pd.DataFrame(rows)


def select_images(trial: bool) -> pd.DataFrame:
    idx = pd.read_parquet(FEATURES_DIR / "image_index.parquet",
                          columns=["path", "day_dir", "t_img", "sza_t_img", "corrupted"])
    cnn = set(pd.read_parquet(FEATURES_DIR / "samples_setA.parquet", columns=["img_cnn"])["img_cnn"])
    keep = ((idx["sza_t_img"] <= MAX_SZA) & ~idx["corrupted"]) | idx["path"].isin(cnn)
    idx = idx[keep]
    if trial:
        train_days = idx.loc[idx["t_img"] < TRAIN_END - pd.Timedelta(days=1), "day_dir"].unique()
        days = np.random.default_rng(21).choice(train_days, size=10, replace=False)
        idx = (idx[idx["day_dir"].isin(days)].sort_values("t_img")
               .groupby("day_dir", group_keys=False).head(100))
    return idx


_STORE = None


def _validate_chunk(df: pd.DataFrame) -> list[str]:
    from verify_cloud_features import verify
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        return [f"missing columns {missing}"]
    return [f"{k}: {v[1]}" for k, v in verify(df).items() if not v[0]]


def process_and_commit(day: pd.DataFrame) -> dict:
    key = chunk_key(day["day_dir"].iloc[0])
    t0 = time.time()
    df = process_day(day)
    return _STORE.commit(key, df, day["path"].tolist(), _validate_chunk,
                         {"seconds": round(time.time() - t0, 1), "backend": "cpu", "pid": os.getpid()})


def run_trial(args):
    out = refuse_existing(OUT_TRIAL)
    idx = select_images(True)
    days = [g for _, g in idx.groupby("day_dir")]
    t0 = time.time()
    with mp.get_context("fork").Pool(args.workers) as pool:
        parts = list(pool.imap_unordered(process_day, days))
    df = pd.concat(parts, ignore_index=True).sort_values("t_img").reset_index(drop=True)
    df.to_parquet(out, index=False)
    ok = df[df["read_ok"]]
    n_full = len(select_images(False))
    timing = {
        "git_head": git_head(), "versions": versions(), "n_images": int(len(df)), "n_read_ok": int(len(ok)),
        "wall_seconds": round(time.time() - t0, 1), "workers": args.workers,
        "ms_per_image_median": float(ok["ms"].median()), "ms_per_image_p95": float(ok["ms"].quantile(0.95)),
        "n_full_images": int(n_full),
        "projected_full_minutes_at_64_workers": round(n_full * float(ok["ms"].median()) / 1000 / 64 / 60, 1),
        "missing_frac": {c: float(ok[c].isna().mean()) for c in ALL_FEATURES},
        "ranges": {c: [float(ok[c].min()), float(ok[c].max())] for c in ALL_FEATURES if ok[c].notna().any()},
        "motion_valid_frac": float(ok["mo_valid"].mean()),
    }
    write_json(CHECKS_DIR / "cloud_trial_timing.json", timing)
    print({k: v for k, v in timing.items() if k not in ("missing_frac", "ranges", "versions")})


def main():
    global _STORE
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", action="store_true")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--parts-dir", default=str(PARTS_DIR))
    ap.add_argument("--out", default=str(OUT_FULL))
    ap.add_argument("--days", default="", help="comma-separated day folders (testing only); no assembly")
    ap.add_argument("--stop-after", type=int, default=0, help="stop after N new chunks (resume testing)")
    args = ap.parse_args()
    if args.trial:
        return run_trial(args)

    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"{out} already exists; refusing to overwrite")
    config_id = frozen_config_id()
    _STORE = ChunkStore(Path(args.parts_dir), config_id)
    _STORE.clean_tmp()

    idx = select_images(False)
    if args.days:
        idx = idx[idx["day_dir"].isin(args.days.split(","))]
    days = {chunk_key(k): g for k, g in idx.groupby("day_dir")}
    todo, reasons = [], {}
    for key, g in days.items():
        ok, why = _STORE.is_complete(key, len(g))
        if not ok:
            if why != "no ledger record":
                reasons[key] = why
            _STORE.discard(key)
            todo.append(g)
    todo.sort(key=len, reverse=True)  # long days first for load balance
    print(f"config {config_id[:12]} | {len(days)} chunks: {len(days) - len(todo)} verified, {len(todo)} to do "
          f"({sum(len(g) for g in todo)} images), {args.workers} workers", flush=True)
    for key, why in sorted(reasons.items()):
        print(f"  redo {key}: {why}", flush=True)

    t0 = time.time()
    if todo:
        pool = mp.get_context("fork").Pool(args.workers)
        try:
            for i, rec in enumerate(pool.imap_unordered(process_and_commit, todo), start=1):
                if i % 50 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)} chunks committed, {time.time() - t0:.0f} s", flush=True)
                if args.stop_after and i >= args.stop_after:
                    print(f"  stopping after {i} chunks (--stop-after)", flush=True)
                    pool.terminate()
                    return
            pool.close()
        finally:
            pool.join()

    if args.days:
        print("subset run complete; no assembly", flush=True)
        return
    incomplete = [k for k, g in days.items() if not _STORE.is_complete(k, len(g))[0]]
    if incomplete:
        raise RuntimeError(f"{len(incomplete)} chunks incomplete after run, e.g. {incomplete[:5]}; rerun to resume")
    df = pd.concat([pd.read_parquet(_STORE.part(k)) for k in sorted(days)], ignore_index=True)
    df = df.sort_values("t_img").reset_index(drop=True)
    tmp = out.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out)
    ok = df[df["read_ok"]]
    write_json(out.with_name(out.stem + "_summary.json"), {
        "git_head": git_head(), "versions": versions(), "config_id": config_id, "n_images": int(len(df)),
        "n_read_ok": int(len(ok)), "wall_seconds_this_run": round(time.time() - t0, 1), "workers": args.workers,
        "chunks": len(days), "ms_per_image_median": float(ok["ms"].median()),
        "motion_valid_frac": float(ok["mo_valid"].mean())})
    print(f"assembled {out} ({len(df)} rows); run verify_cloud_features.py on it", flush=True)


if __name__ == "__main__":
    main()
