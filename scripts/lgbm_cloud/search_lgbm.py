"""Step 6a: LightGBM random search on validation (purged first day), same budget for every set.

The same N_TRIALS settings (same RNG seed) are tried for every feature set; trial 0 is
LGBM_DEFAULTS. Fit on train, early stopping and selection on the purged validation k-MSE.
Test rows are never loaded.

Output: experiments/lightgbm_cloud_features/runs/search_<set>/trials.csv, best.json
Resumable: trials already in trials.csv (same space seed) are skipped on a rerun.

Usage:
    PYTHONPATH=. python scripts/lgbm_cloud/search_lgbm.py --features A --threads 24
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from common import RUNS_DIR, git_head, versions, write_json
from feature_sets import TABLE_OF, get_feature_set
from train_tabular import LGBM_DEFAULTS, fit_lgbm, load, val_metrics

N_TRIALS = 30
SPACE_SEED = 2026


def sample_space(n, seed=SPACE_SEED):
    rng = np.random.default_rng(seed)
    out = [{}]  # trial 0: defaults
    for _ in range(n - 1):
        out.append({
            "num_leaves": int(rng.integers(15, 128)),
            "min_data_in_leaf": int(round(np.exp(rng.uniform(np.log(200), np.log(5000))))),
            "learning_rate": float(np.exp(rng.uniform(np.log(0.02), np.log(0.1)))),
            "feature_fraction": float(rng.uniform(0.6, 0.9)),
            "bagging_fraction": float(rng.uniform(0.6, 0.9)),
            "lambda_l2": float(rng.uniform(0.0, 10.0)),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--n-trials", type=int, default=N_TRIALS)
    args = ap.parse_args()
    out_dir = RUNS_DIR / f"search_{args.features}"
    if (out_dir / "best.json").exists():
        raise FileExistsError(f"{out_dir} already finished; refusing to overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = out_dir / "space.json"
    if meta.exists() and json.loads(meta.read_text()) != {"space_seed": SPACE_SEED, "n_trials": args.n_trials}:
        raise RuntimeError(f"{meta} was written with a different search space")
    write_json(meta, {"space_seed": SPACE_SEED, "n_trials": args.n_trials})
    rows = pd.read_csv(out_dir / "trials.csv").to_dict("records") if (out_dir / "trials.csv").exists() else []
    done = {int(r["trial"]) for r in rows}
    if done:
        print(f"[{args.features}] resuming: {len(done)} trials already done", flush=True)
    features = get_feature_set(args.features)
    train, val, val_sel, _ = load(features, TABLE_OF[args.features])

    for i, over in enumerate(sample_space(args.n_trials)):
        if i in done:
            continue
        t0 = time.time()
        params = {**over, "num_threads": args.threads}
        model, info = fit_lgbm(train, val_sel, features, seed=0, overrides=params,
                               num_boost_round=5000, early_stopping=200)
        k_sel = model.predict(val_sel[features], num_iteration=model.best_iteration)
        m = val_metrics(val_sel, k_sel)
        rows.append({"trial": i, **{k: info["params"][k] for k in ("num_leaves", "min_data_in_leaf", "learning_rate",
                                                                    "feature_fraction", "bagging_fraction", "lambda_l2")},
                     "best_iteration": info["best_iteration"], "val_purged_k_MSE": m["k_MSE"],
                     "val_purged_RMSE_Wm2": m["RMSE_Wm2"], "val_purged_MAE_Wm2": m["MAE_Wm2"],
                     "seconds": round(time.time() - t0, 1)})
        tmp = out_dir / "trials.csv.tmp"
        pd.DataFrame(rows).to_csv(tmp, index=False)
        os.replace(tmp, out_dir / "trials.csv")
        print(f"[{args.features}] trial {i}: k_MSE {m['k_MSE']:.6f} RMSE {m['RMSE_Wm2']:.2f} "
              f"iters {info['best_iteration']} ({rows[-1]['seconds']} s)", flush=True)

    df = pd.DataFrame(rows).sort_values("val_purged_k_MSE")
    best = df.iloc[0]
    best_params = {k: (int(best[k]) if k in ("num_leaves", "min_data_in_leaf") else float(best[k]))
                   for k in ("num_leaves", "min_data_in_leaf", "learning_rate", "feature_fraction",
                             "bagging_fraction", "lambda_l2")}
    write_json(out_dir / "best.json", {
        "feature_set": args.features, "best_trial": int(best["trial"]), "params": best_params,
        "best_iteration": int(best["best_iteration"]), "val_purged_k_MSE": float(best["val_purged_k_MSE"]),
        "top5_trials": df.head(5)["trial"].astype(int).tolist(), "n_trials": args.n_trials,
        "space_seed": SPACE_SEED, "defaults": LGBM_DEFAULTS, "git_head": git_head(), "versions": versions(),
    })
    print(json.dumps({"best_trial": int(best["trial"]), "params": best_params,
                      "val_purged_k_MSE": float(best["val_purged_k_MSE"])}, indent=1))


if __name__ == "__main__":
    main()
