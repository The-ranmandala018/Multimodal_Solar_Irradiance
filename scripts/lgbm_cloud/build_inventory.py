"""Artifact inventory of the LightGBM cloud-feature experiment -> documentation/ARTIFACT_INVENTORY.csv

Every file under experiments/lightgbm_cloud_features/ and scripts/lgbm_cloud/ with size, sha256,
category, description and the proposed publication decision:
    YES       proposed for the repository
    NO        kept locally / external archive only (too large, or superseded by a published copy)
    HELD      held out pending the authors' approval (images derived from sky photographs)
Refuses to overwrite.
"""

from __future__ import annotations

import fnmatch
import hashlib

import pandas as pd

from common import EXP_DIR, PROJECT_DIR, refuse_existing

OUT = EXP_DIR / "documentation" / "ARTIFACT_INVENTORY.csv"
E = "experiments/lightgbm_cloud_features/"
RULES = [  # first match wins: (pattern, category, publish, description)
    ("scripts/lgbm_cloud/benchmarks/*", "code", "YES", "benchmark / profiling script (not used for results)"),
    ("scripts/lgbm_cloud/phase1/*", "code", "YES", "Phase 1 time-alignment checks"),
    ("scripts/lgbm_cloud/*", "code", "YES", "experiment code"),
    (E + "README.md", "documentation", "YES", "experiment README"),
    (E + "FROZEN_V1*", "documentation", "YES", "frozen cloud-feature v1 specification / hashes"),
    (E + "documentation/*", "documentation", "YES", "journal documentation"),
    (E + "summary/val_*", "results-validation", "YES", "validation result table"),
    (E + "summary/test_*", "results-test", "YES", "final test result table"),
    (E + "publication/*", "predictions", "YES", "compact verbatim prediction copies for publication"),
    (E + "runs/search_*", "tuning", "YES", "hyperparameter search / stability"),
    (E + "runs/*/config.json", "run-config", "YES", "run configuration, seed, fit time, versions"),
    (E + "runs/*/metrics_val.json", "results-validation", "YES", "per-run validation metrics"),
    (E + "runs/*/feature_importance.csv", "results-validation", "YES", "LightGBM gain/split importance"),
    (E + "runs/predict_test_log.json", "results-test", "YES", "test prediction log"),
    (E + "runs/*/model.txt", "model", "YES", "saved LightGBM model (best iteration); verified in model_verification.json"),
    (E + "runs/*/model.pkl", "model", "YES", "saved Ridge pipeline (pickle, scikit-learn 1.7.2); verified"),
    (E + "runs/*/model_params.json", "model", "YES", "pickle-free Ridge parameters (reproduce the pickle exactly)"),
    (E + "runs/*/predictions_*.parquet", "predictions-original", "NO",
     "original per-run predictions (published verbatim in publication/; sha256 in predictions_sources.json)"),
    (E + "logs/*", "logs", "YES", "run log (*.log is git-ignored: needs git add -f)"),
    (E + "checks/gate2/*.png", "figure-sky-images", "HELD", "Gate 2 overlays of 50 downsampled sky images (held: attribution to be approved)"),
    (E + "checks/sky_mask_preview.png", "figure-sky-images", "HELD", "median sky image with obstruction mask (held: attribution to be approved)"),
    (E + "checks/*", "evidence", "YES", "alignment / calibration / benchmark evidence"),
    (E + "features/windows_40x7.npy", "derived-data", "NO", "40x7 history windows for all samples (735 MB)"),
    (E + "features/*.parquet", "derived-data", "NO", "derived per-image / per-sample feature tables (large)"),
    (E + "features/*", "fitted-parameters", "YES", "fitted geometry / masks / clear-sky library / thresholds / summaries"),
]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def classify(rel):
    for pat, cat, pub, desc in RULES:
        if fnmatch.fnmatch(rel, pat):
            return cat, pub, desc
    return "unclassified", "DECISION", ""


def main():
    refuse_existing(OUT)
    rows = []
    for root in (EXP_DIR, PROJECT_DIR / "scripts" / "lgbm_cloud"):
        for p in sorted(root.rglob("*")):
            if not p.is_file() or "__pycache__" in p.parts:
                continue
            rel = p.relative_to(PROJECT_DIR).as_posix()
            if rel == OUT.relative_to(PROJECT_DIR).as_posix():
                continue
            cat, pub, desc = classify(rel)
            rows.append({"path": rel, "bytes": p.stat().st_size, "sha256": sha256(p), "category": cat,
                         "publish": pub, "description": desc})
    inv = pd.DataFrame(rows)
    inv.to_csv(OUT, index=False)
    s = inv.groupby(["publish", "category"]).agg(files=("path", "size"), MB=("bytes", lambda b: round(b.sum() / 1e6, 2)))
    print(s.to_string())
    print(inv.groupby("publish")["bytes"].sum().div(1e6).round(1).to_string())
    print("largest YES file:", inv[inv.publish == "YES"].nlargest(3, "bytes")[["path", "bytes"]].to_string(index=False))


if __name__ == "__main__":
    main()
