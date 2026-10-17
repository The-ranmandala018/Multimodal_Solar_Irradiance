"""Verify the 30 saved final models for publication: integrity, compatibility, exact reproduction.

For every runs/final_*/ model (25 LightGBM model.txt, 2 Ridge model.pkl; plus the 3 Step 3 check models):
  - sha256 equals documentation/ARTIFACT_INVENTORY.csv
  - loads with the recorded library versions (lightgbm 4.6.0 / scikit-learn 1.7.2)
  - feature names and count equal config.json; LightGBM tree count equals best_iteration
  - re-predicting the validation (and, for final runs, test) rows reproduces the saved
    predictions: exactly for LightGBM (max |diff| == 0), within 1e-12 for Ridge (1-ulp BLAS order)
  - no absolute paths or user names inside the model files
Ridge pipelines are also exported to a pickle-free JSON (imputer medians, indicator columns,
scaler mean/scale, coefficients, intercept) next to the pickle as model_params.json, and the
JSON is checked to reproduce the pickle's predictions (|diff| <= 1e-9).
Writes documentation/model_verification.json. Refuses to overwrite.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn

from common import EXP_DIR, FEATURES_DIR, RUNS_DIR, refuse_existing, write_json

OUT = EXP_DIR / "documentation" / "model_verification.json"
SENSITIVE = re.compile(rb"/storage2|/userhomes|/home/|shehan|keshawa|e20187|[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}")
RIDGE_TOL = 1e-12  # Ridge re-prediction can differ by 1 ulp (BLAS summation order); LightGBM must be exact


def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ridge_json(pipe, features):
    imp, sc, rg = pipe.named_steps["simpleimputer"], pipe.named_steps["standardscaler"], pipe.named_steps["ridge"]
    ind = imp.indicator_.features_ if imp.indicator_ is not None else np.array([], int)
    return {"features": features, "imputer_strategy": imp.strategy, "imputer_statistics": imp.statistics_.tolist(),
            "missing_indicator_feature_index": ind.tolist(), "scaler_mean": sc.mean_.tolist(),
            "scaler_scale": sc.scale_.tolist(), "coef": rg.coef_.tolist(), "intercept": float(rg.intercept_),
            "alpha": float(rg.alpha), "sklearn_version": sklearn.__version__,
            "predict": "x_imp = where(isnan(x), imputer_statistics, x); z = concat(x_imp, isnan(x)[:, missing_indicator_feature_index]); "
                       "k = ((z - scaler_mean) / scaler_scale) @ coef + intercept"}


def ridge_predict_json(p, X):
    x = X.to_numpy(np.float64)
    nan = np.isnan(x)
    xi = np.where(nan, np.asarray(p["imputer_statistics"]), x)
    z = np.hstack([xi, nan[:, p["missing_indicator_feature_index"]].astype(np.float64)])
    return ((z - np.asarray(p["scaler_mean"])) / np.asarray(p["scaler_scale"])) @ np.asarray(p["coef"]) + p["intercept"]


def main():
    refuse_existing(OUT)
    inv = pd.read_csv(EXP_DIR / "documentation" / "ARTIFACT_INVENTORY.csv").set_index("path")
    tables, results = {}, {}
    for d in sorted(RUNS_DIR.glob("*_*")):
        cfg_p = d / "config.json"
        if not cfg_p.exists():
            continue
        cfg = json.loads(cfg_p.read_text())
        mfile = d / ("model.txt" if cfg["model"] == "lgbm" else "model.pkl")
        rel = mfile.relative_to(EXP_DIR.parents[1]).as_posix()
        r = {"file": rel, "bytes": mfile.stat().st_size, "sha256_matches_inventory": sha256(mfile) == inv.loc[rel, "sha256"],
             "no_paths_or_names_inside": SENSITIVE.search(mfile.read_bytes()) is None}
        feats = cfg["features"]
        splits = ["val", "test"] if (d / "predictions_test.parquet").exists() else ["val"]
        if cfg["model"] == "lgbm":
            m = lgb.Booster(model_file=str(mfile))
            r["features_match_config"] = m.feature_name() == feats
            r["n_trees"] = m.num_trees()
            r["trees_equal_best_iteration"] = m.num_trees() == cfg["fit"]["best_iteration"]
            predict = m.predict
        else:
            with open(mfile, "rb") as f:
                m = pickle.load(f)
            r["features_match_config"] = list(m.feature_names_in_) == feats
            params = ridge_json(m, feats)
            pj = d / "model_params.json"
            refuse_existing(pj)
            write_json(pj, params)
            predict = m.predict
        for split in splits:
            key = (cfg.get("table", "samples_setA.parquet"), split)  # Step 3 check configs predate the field
            if key not in tables:
                tables[key] = pd.read_parquet(FEATURES_DIR / key[0], filters=[("split", "==", split)]).sort_values("split_pos")
            t = tables[key]
            saved = pd.read_parquet(d / f"predictions_{split}.parquet").sort_values("split_pos")["k_pred"].to_numpy()
            k = predict(t[feats])
            r[f"{split}_max_abs_diff_vs_saved"] = float(np.max(np.abs(k - saved)))
            if cfg["model"] == "ridge":
                r[f"{split}_json_max_abs_diff_vs_pickle"] = float(np.max(np.abs(ridge_predict_json(params, t[feats]) - k)))
        r["ok"] = bool(r["sha256_matches_inventory"] and r["no_paths_or_names_inside"] and r["features_match_config"]
                       and r.get("trees_equal_best_iteration", True)
                       and all(v <= (RIDGE_TOL if cfg["model"] == "ridge" else 0.0) for k_, v in r.items() if k_.endswith("_vs_saved"))
                       and all(v <= 1e-9 for k_, v in r.items() if k_.endswith("_vs_pickle")))
        results[d.name] = r
        print(d.name, "OK" if r["ok"] else "FAIL", {k_: v for k_, v in r.items() if "diff" in k_})
    write_json(OUT, {"lightgbm": lgb.__version__, "sklearn": sklearn.__version__, "all_ok": all(v["ok"] for v in results.values()),
                     "n_models": len(results), "models": results,
                     "note": "Ridge model.pkl requires scikit-learn 1.7.2 to unpickle; model_params.json is a pickle-free equivalent."})
    print("all_ok:", all(v["ok"] for v in results.values()), "| models:", len(results))


if __name__ == "__main__":
    main()
