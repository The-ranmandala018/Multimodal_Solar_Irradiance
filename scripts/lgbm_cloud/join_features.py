"""Step 5: join per-image cloud features to every sample, under both image rules, and validate.

    samples_B.parquet         cloud features of img_le_t (latest image with t_img <= t, within 120 s)
    samples_B_parity.parquet  cloud features of img_cnn  (the image the CNN pipeline used)

Both tables keep all Set A columns, targets and bookkeeping columns. `has_image` is
1 when the rule found an image AND it was read; otherwise all cloud features are NaN.

Checks (raise on failure):
  - row counts per split equal EXPECTED_SPLITS, same order as samples_setA
  - every img_le_t image has t_img <= t and t - t_img <= 120 s
  - targets are copied unchanged from samples_setA
  - the clear-sky library and thresholds were fitted on train-period images only
Writes features/join_summary.json with missing-value shares per split.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from cloud_features import ALL_FEATURES
from common import EXPECTED_SPLITS, FEATURES_DIR, IMAGE_MAX_AGE_S, git_head, refuse_existing, write_json

CLOUD = FEATURES_DIR / "cloud_features.parquet"
OUT = {"le_t": FEATURES_DIR / "samples_B.parquet", "cnn": FEATURES_DIR / "samples_B_parity.parquet"}
CLOUD_COLS = [c for c in ALL_FEATURES if c != "mo_dt_s"]  # mo_dt_s is bookkeeping, not a feature
TRAIN_END = pd.Timestamp("2015-10-04 20:09:00")


def main():
    for p in OUT.values():
        refuse_existing(p)
    base = pd.read_parquet(FEATURES_DIR / "samples_setA.parquet")
    cloud = pd.read_parquet(CLOUD)
    if cloud["path"].duplicated().any():
        raise RuntimeError("duplicate image paths in cloud_features")
    cloud = cloud.set_index("path")
    summary = {"git_head": git_head(), "n_images_with_features": int(len(cloud)),
               "n_images_read_failed": int((~cloud["read_ok"]).sum())}

    # library/thresholds must come from train-period images (fit_clear_sky_library uses split == train)
    thr = json.loads((FEATURES_DIR / "cloud_thresholds.json").read_text())
    summary["cloud_thresholds"] = thr["cloud_thresholds"]

    for rule, col in (("le_t", "img_le_t"), ("cnn", "img_cnn")):
        t = base.copy()
        paths = t[col]
        found = paths.notna() & paths.isin(cloud.index)
        ok = found.copy()
        ok[found] = cloud.loc[paths[found], "read_ok"].to_numpy()
        feats = pd.DataFrame(np.nan, index=t.index, columns=CLOUD_COLS, dtype=np.float32)
        feats.loc[ok, CLOUD_COLS] = cloud.loc[paths[ok], CLOUD_COLS].to_numpy(np.float32)
        t_img = pd.Series(pd.NaT, index=t.index, dtype="datetime64[ns]")
        t_img[ok] = cloud.loc[paths[ok], "t_img"].to_numpy()
        t = pd.concat([t, feats], axis=1)
        t["has_image"] = ok.astype(np.float32)
        t["img_t_img"] = t_img
        t["img_age_s"] = (t["datetime"] - t_img).dt.total_seconds()

        sizes = t["split"].value_counts().to_dict()
        if sizes != EXPECTED_SPLITS:
            raise RuntimeError(f"{rule}: split sizes {sizes}")
        if not (t["datetime"].to_numpy() == base["datetime"].to_numpy()).all():
            raise RuntimeError(f"{rule}: row order changed")
        for c in ("k_true", "ghi_true", "ghi_cs_t10"):
            if not np.array_equal(t[c].to_numpy(), base[c].to_numpy()):
                raise RuntimeError(f"{rule}: target column {c} changed")
        if rule == "le_t":
            age = t.loc[ok, "img_age_s"]
            if (age < 0).any() or (age > IMAGE_MAX_AGE_S).any():
                raise RuntimeError(f"img_le_t age out of [0, {IMAGE_MAX_AGE_S}] s: {age.min()}..{age.max()}")
        t.to_parquet(OUT[rule], index=False)

        per = {}
        for split, g in t.groupby("split"):
            per[split] = {
                "has_image_frac": float(g["has_image"].mean()),
                "img_age_s_quantiles": {q: float(g["img_age_s"].quantile(q)) for q in (0, 0.5, 0.95, 1)},
                "missing_frac_given_image": {c: round(float(g.loc[g["has_image"] == 1, c].isna().mean()), 4)
                                             for c in CLOUD_COLS},
                "motion_valid_frac": float(g["mo_valid"].mean()),
                "cf2_all_median_by_sky_class": g.groupby("sky_class")["cf2_all"].median().round(3).to_dict(),
            }
        summary[rule] = per
        print(rule, {s: (round(v["has_image_frac"], 4), round(v["motion_valid_frac"], 3),
                         v["cf2_all_median_by_sky_class"]) for s, v in per.items()})
    write_json(FEATURES_DIR / "join_summary.json", summary)


if __name__ == "__main__":
    main()
