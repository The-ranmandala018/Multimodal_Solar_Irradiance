"""Verify a cloud-feature extraction output against the frozen v1 configuration.

Checks (all must pass):
  coverage      every image selected for extraction is present exactly once, including
                every img_le_t and img_cnn image used by any sample
  timestamps    t_img equals the corrected time in image_index.parquet
  reads         read_ok for every row (failed reads would need NaN features)
  finiteness    no +-inf anywhere
  ranges        fractions in [0, 1]; flags in {0, 1}; angles in [0, 180]; brightness in [0, 255];
                entropy in [0, 6]; blob area in [0, 384*384]
  motion NaNs   mo_valid == 0  -> vx, vy, speed, resid_ratio, upwind_cf, time_to_sun all NaN
                mo_valid == 1  -> vx, vy, speed finite, response >= MOTION_MIN_RESPONSE
Writes <output>.verify.json next to the file and prints a summary.

Usage:
    PYTHONPATH=. python scripts/lgbm_cloud/verify_cloud_features.py [path/to/cloud_features.parquet]
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cloud_features import ALL_FEATURES, FEATURE_GROUPS, MOTION_MIN_RESPONSE
from common import FEATURES_DIR, write_json

FRACTIONS = [c for c in ALL_FEATURES if c.startswith(("cf", "pos_cf"))] + [
    "sun_sat_frac_all", "sun_sat_frac_5", "sun_sat_frac_15", "mo_upwind_cf"]
FLAGS = ["sun_detected", "mo_valid"]
ANGLES = ["pos_min_angle_cloud", "sun_blob_offset_deg"]
MOTION_VALUES = ["mo_vx", "mo_vy", "mo_speed", "mo_resid_ratio", "mo_upwind_cf", "mo_time_to_sun_min"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def verify(df: pd.DataFrame, expected_paths: set | None = None) -> dict:
    """Checks that do not need the sample table; returns {check: (ok, detail)}."""
    out = {}
    out["unique_paths"] = (not df["path"].duplicated().any(), int(df["path"].duplicated().sum()))
    if expected_paths is not None:
        got = set(df["path"])
        out["coverage"] = (got == expected_paths, {"missing": len(expected_paths - got), "extra": len(got - expected_paths)})
    out["read_ok"] = (bool(df["read_ok"].all()), int((~df["read_ok"]).sum()))
    num = df[ALL_FEATURES].to_numpy(np.float64)
    out["no_inf"] = (not np.isinf(num).any(), int(np.isinf(num).sum()))

    def in_range(cols, lo, hi):
        v = df[cols].to_numpy(np.float64)
        bad = ~np.isnan(v) & ((v < lo) | (v > hi))
        return (not bad.any(), int(bad.sum()))

    out["fractions_0_1"] = in_range(FRACTIONS, 0, 1)
    flags = df[FLAGS].to_numpy(np.float64)
    out["flags_0_1"] = (bool(np.isin(flags, [0, 1]).all()), int((~np.isin(flags, [0, 1])).sum()))
    out["angles_0_180"] = in_range(ANGLES, 0, 180)
    out["gray_0_255"] = in_range(["tx_gray_mean"], 0, 255)
    out["entropy_0_6"] = in_range(["tx_entropy"], 0, 6)
    out["blob_area"] = in_range(["sun_blob_area"], 0, 384 * 384)

    inval = df["mo_valid"] == 0
    leak = df.loc[inval, MOTION_VALUES].notna().sum()
    out["motion_invalid_all_nan"] = (int(leak.sum()) == 0, leak[leak > 0].to_dict())
    val = df["mo_valid"] == 1
    miss = df.loc[val, ["mo_vx", "mo_vy", "mo_speed"]].isna().sum()
    out["motion_valid_finite"] = (int(miss.sum()) == 0, miss[miss > 0].to_dict())
    low = int((df.loc[val, "mo_response"] < MOTION_MIN_RESPONSE).sum())
    out["motion_valid_response"] = (low == 0, low)
    return out


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else FEATURES_DIR / "cloud_features.parquet"
    df = pd.read_parquet(path)
    idx = pd.read_parquet(FEATURES_DIR / "image_index.parquet",
                          columns=["path", "t_img", "sza_t_img", "corrupted"]).set_index("path")
    s = pd.read_parquet(FEATURES_DIR / "samples_setA.parquet", columns=["split", "img_le_t", "img_cnn"])
    from extract_cloud_features import select_images
    expected = set(select_images(False)["path"])
    res = verify(df, expected)

    used_le = set(s["img_le_t"].dropna())
    used_cnn = set(s["img_cnn"])
    got = set(df["path"])
    res["all_img_le_t_present"] = (used_le <= got, len(used_le - got))
    res["all_img_cnn_present"] = (used_cnn <= got, len(used_cnn - got))
    t_ref = idx.loc[df["path"], "t_img"].to_numpy()
    res["t_img_matches_index"] = (bool((df["t_img"].to_numpy() == t_ref).all()),
                                  int((df["t_img"].to_numpy() != t_ref).sum()))

    ok = all(v[0] for v in res.values())
    summary = {
        "file": str(path), "sha256": sha256(path), "rows": int(len(df)), "all_checks_pass": ok,
        "checks": {k: {"ok": bool(v[0]), "detail": v[1]} for k, v in res.items()},
        "missing_frac_by_group": {g: {c: round(float(df[c].isna().mean()), 4) for c in cols}
                                  for g, cols in FEATURE_GROUPS.items()},
        "motion_valid_frac": float(df["mo_valid"].mean()),
        "motion_valid_but_resid_nan": int(df.loc[df["mo_valid"] == 1, "mo_resid_ratio"].isna().sum()),
        "ranges": {c: [float(df[c].min()), float(df[c].max())] for c in ALL_FEATURES if df[c].notna().any()},
    }
    write_json(path.with_suffix(".verify.json"), summary)
    for k, v in summary["checks"].items():
        print(f"{'PASS' if v['ok'] else 'FAIL'}  {k}: {v['detail']}")
    print("all_checks_pass:", ok, "| sha256:", summary["sha256"][:16])
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
