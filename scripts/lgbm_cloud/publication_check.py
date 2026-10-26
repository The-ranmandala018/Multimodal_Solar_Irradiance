"""Publication check for the LightGBM cloud-feature experiment (reads originals, never modifies them).

1. Recompute the reported metrics from the ORIGINAL prediction files and compare them with
   summary/val_metrics_overall.csv and summary/test_metrics_overall.csv (tolerance 1e-9).
2. Build compact prediction files for publication:
       publication/targets_<split>.parquet               targets and grouping columns (split_pos key)
       publication/predictions_<split>_<model>.parquet   split_pos + one k_pred column per seed (float64, verbatim)
   and verify that metrics recomputed from the joined published files equal the originals exactly.
(The artifact inventory is written separately by build_inventory.py.)
Refuses to overwrite any output.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import EXP_DIR, FEATURES_DIR, PROJECT_DIR, RUNS_DIR, SUMMARY_DIR, refuse_existing, regression_metrics, write_json
from evaluate_test_final import EXISTING, LGBM, RIDGE, reorder_saved

PUB_DIR = EXP_DIR / "publication"
DOC_DIR = EXP_DIR / "documentation"
TOL = 1e-9


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def run_dirs(prefix, seeded):
    return [RUNS_DIR / f"{prefix}_seed{k}" for k in range(5)] if seeded else [RUNS_DIR / prefix]


def consolidate(split):
    cols = ["split_pos", "datetime", "k_true", "ghi_true", "ghi_cs_t10", "sky_class", "test_period", "has_image"]
    base = pd.read_parquet(FEATURES_DIR / "samples_B.parquet",
                           columns=cols + ["sza_t", "cf2_all", "clearsky_persistence", "ghi_t", "split"],
                           filters=[("split", "==", split)]).sort_values("split_pos").reset_index(drop=True)
    base = base.drop(columns="split")
    sources = {}
    for name, prefix in {**LGBM, **RIDGE}.items():
        for d in run_dirs(prefix, name in LGBM):
            p = pd.read_parquet(d / f"predictions_{split}.parquet").sort_values("split_pos").reset_index(drop=True)
            if not (np.array_equal(p["split_pos"].to_numpy(), base["split_pos"].to_numpy())
                    and np.array_equal(p["k_true"].to_numpy(), base["k_true"].to_numpy())):
                raise RuntimeError(f"{d.name}/{split}: rows or targets differ from the feature table")
            if split == "val" and "in_purged_val" not in base:
                base["in_purged_val"] = p["in_purged_val"].to_numpy()
            base[f"k_pred__{d.name}"] = p["k_pred"].to_numpy()
            # stored GHI must equal k_pred * GHI_cs exactly, so GHI can be recomputed from the consolidated file
            if not np.array_equal(p["ghi_pred"].to_numpy(), p["k_pred"].to_numpy() * p["ghi_cs_t10"].to_numpy()):
                raise RuntimeError(f"{d.name}/{split}: ghi_pred != k_pred * ghi_cs_t10")
            sources[d.name] = {"file": str((d / f"predictions_{split}.parquet").relative_to(PROJECT_DIR)),
                               "sha256": sha256(d / f"predictions_{split}.parquet")}
    return base, sources


def metrics_from(df, name, prefix, seeded, mask=None):
    cols = [f"k_pred__{d.name}" for d in run_dirs(prefix, seeded)]
    cs = df["ghi_cs_t10"].to_numpy()
    g = np.mean([df[c].to_numpy() * cs for c in cols], axis=0)  # same order as the evaluation scripts
    y = df["ghi_true"].to_numpy()
    if mask is not None:
        g, y = g[mask], y[mask]
    return regression_metrics(y, g)


def original_seed_avg(split, prefix, seeded, mask_col=None):
    ghis, y = [], None
    for d in run_dirs(prefix, seeded):
        p = pd.read_parquet(d / f"predictions_{split}.parquet").sort_values("split_pos")
        m = p[mask_col].to_numpy() if mask_col else np.ones(len(p), bool)
        ghis.append(p["ghi_pred"].to_numpy()[m])
        y = p["ghi_true"].to_numpy()[m]
    return regression_metrics(y, np.mean(ghis, axis=0))


def compare(reported: pd.DataFrame, key, name, m, out):
    r = reported[reported[key] == name]
    if r.empty:
        out.append({"check": f"{key}={name}", "ok": False, "detail": "not in reported table"})
        return
    r = r.iloc[0]
    diffs = {k: abs(float(r[k]) - m[k]) for k in ("RMSE_Wm2", "MAE_Wm2", "MBE_Wm2", "R2")}
    out.append({"check": f"{key}={name}", "ok": max(diffs.values()) <= TOL, "detail": max(diffs.values())})


def main():
    for p in (PUB_DIR / "predictions_sources.json", DOC_DIR / "publication_check.json"):
        refuse_existing(p)
    PUB_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    checks = []

    # ---- 1 + 2: validation ---------------------------------------------------
    val_rep = pd.read_csv(SUMMARY_DIR / "val_metrics_overall.csv")
    val, val_src = consolidate("val")
    purged = val["in_purged_val"].to_numpy()
    for name, prefix in {**LGBM, **RIDGE}.items():
        seeded = name in LGBM
        orig = original_seed_avg("val", prefix, seeded, "in_purged_val")
        compare(val_rep, "run", name, orig, checks)
        cons = metrics_from(val, name, prefix, seeded, purged)
        checks.append({"check": f"val consolidated==original {name}", "ok": cons == orig, "detail": ""})

    # ---- 1 + 2: test -----------------------------------------------------------
    test_rep = pd.read_csv(SUMMARY_DIR / "test_metrics_overall.csv")
    test, test_src = consolidate("test")
    for name, prefix in {**LGBM, **RIDGE}.items():
        seeded = name in LGBM
        orig = original_seed_avg("test", prefix, seeded)
        compare(test_rep, "model", name, orig, checks)
        cons = metrics_from(test, name, prefix, seeded)
        checks.append({"check": f"test consolidated==original {name}", "ok": cons == orig, "detail": ""})
    y = test["ghi_true"].to_numpy()
    for name, runs in EXISTING.items():
        g = np.mean([reorder_saved(PROJECT_DIR / "experiments" / r / "evaluation" / "test_predictions.npz", y)[0]
                     for r in runs], axis=0)
        compare(test_rep, "model", name, regression_metrics(y, g), checks)
    compare(test_rep, "model", "Clear-sky persistence", regression_metrics(y, test["clearsky_persistence"].to_numpy()), checks)

    # One shared targets/grouping file per split, plus one k_pred file per model (split_pos + seed columns).
    pub_files = []
    for split, df in (("val", val), ("test", test)):
        common_cols = [c for c in df.columns if not c.startswith("k_pred__")]
        parts = [(PUB_DIR / f"targets_{split}.parquet", common_cols)]
        for name, prefix in {**LGBM, **RIDGE}.items():
            parts.append((PUB_DIR / f"predictions_{split}_{prefix.replace('final_', '')}.parquet",
                          ["split_pos"] + [f"k_pred__{d.name}" for d in run_dirs(prefix, name in LGBM)]))
        for f, cols in parts:
            refuse_existing(f)
            df[cols].to_parquet(f, index=False, compression="zstd")
            back = pd.read_parquet(f)
            checks.append({"check": f"{f.name} round-trip identical", "ok": bool(back.equals(df[cols])), "detail": ""})
            pub_files.append(f)
        # metrics recomputed by joining the published files back together
        joined = pd.read_parquet(PUB_DIR / f"targets_{split}.parquet")
        for name, prefix in {**LGBM, **RIDGE}.items():
            k = pd.read_parquet(PUB_DIR / f"predictions_{split}_{prefix.replace('final_', '')}.parquet")
            j = joined.merge(k, on="split_pos", how="left", validate="one_to_one")
            mask = j["in_purged_val"].to_numpy() if split == "val" else None
            orig = original_seed_avg(split, prefix, name in LGBM, "in_purged_val" if split == "val" else None)
            checks.append({"check": f"{split} published files reproduce {name}",
                           "ok": metrics_from(j, name, prefix, name in LGBM, mask) == orig, "detail": ""})
    write_json(PUB_DIR / "predictions_sources.json", {"val": val_src, "test": test_src,
               "note": "k_pred__<run> columns are copied unchanged from these original files (sha256 at copy time)."})

    ok = all(c["ok"] for c in checks)
    write_json(DOC_DIR / "publication_check.json", {"all_ok": ok, "tolerance": TOL, "checks": checks,
               "publication_files": {f.name: {"bytes": f.stat().st_size, "sha256": sha256(f)} for f in pub_files}})
    for c in checks:
        if not c["ok"]:
            print("FAIL", c)
    print(f"{sum(c['ok'] for c in checks)}/{len(checks)} checks pass")
    print({f.name: round(f.stat().st_size / 1e6, 1) for f in pub_files}, "MB")


if __name__ == "__main__":
    main()
