"""Step 6b: stability check of the chosen LightGBM settings on month-blocked folds INSIDE train.

Train period is cut into N_FOLDS contiguous time blocks. For each block, fit on the rest
of train minus GAP on both sides of the block, with the number of rounds fixed to the
trial's best_iteration from the validation search (no early stopping on the block), and
score the block. Used only to check that the chosen settings are stable, not to pick new ones.

Output: runs/search_<set>/stability_folds.csv and stability_summary.json
"""

from __future__ import annotations

import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import RUNS_DIR, refuse_existing, write_json
from feature_sets import TABLE_OF, get_feature_set
from train_tabular import LGBM_DEFAULTS, load

N_FOLDS = 4
GAP = pd.Timedelta(days=1)
TOP = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--threads", type=int, default=24)
    args = ap.parse_args()
    d = RUNS_DIR / f"search_{args.features}"
    out = refuse_existing(d / "stability_folds.csv")
    trials = pd.read_csv(d / "trials.csv").sort_values("val_purged_k_MSE")
    cand = pd.concat([trials.head(TOP), trials[trials["trial"] == 0]]).drop_duplicates("trial")
    features = get_feature_set(args.features)
    train, _, _, _ = load(features, TABLE_OF[args.features])
    t = train["datetime"]
    edges = t.quantile(np.linspace(0, 1, N_FOLDS + 1)).to_numpy()

    rows = []
    for _, tr in cand.iterrows():
        params = dict(LGBM_DEFAULTS)
        params.update({k: tr[k] for k in ("num_leaves", "min_data_in_leaf", "learning_rate", "feature_fraction",
                                           "bagging_fraction", "lambda_l2")})
        params.update({"num_leaves": int(params["num_leaves"]), "min_data_in_leaf": int(params["min_data_in_leaf"]),
                       "num_threads": args.threads, "seed": 0, "bagging_seed": 0, "feature_fraction_seed": 0,
                       "data_random_seed": 0})
        for f in range(N_FOLDS):
            lo, hi = edges[f], edges[f + 1]
            block = (t >= lo) & ((t <= hi) if f == N_FOLDS - 1 else (t < hi))
            fit = ~((t >= lo - GAP) & (t <= hi + GAP))
            model = lgb.train(params, lgb.Dataset(train.loc[fit, features], train.loc[fit, "k_true"]),
                              num_boost_round=int(tr["best_iteration"]))
            pred = model.predict(train.loc[block, features])
            mse = float(np.mean((pred - train.loc[block, "k_true"].to_numpy()) ** 2))
            rows.append({"trial": int(tr["trial"]), "fold": f, "block_start": str(lo)[:10], "block_end": str(hi)[:10],
                         "n_block": int(block.sum()), "k_MSE": mse, "val_purged_k_MSE": float(tr["val_purged_k_MSE"])})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    piv = df.pivot(index="trial", columns="fold", values="k_MSE")
    ranks = piv.rank(axis=0)
    chosen = int(trials.iloc[0]["trial"])
    write_json(d / "stability_summary.json", {
        "chosen_trial": chosen, "fold_k_MSE": piv.round(6).to_dict(orient="index"),
        "mean_fold_k_MSE": piv.mean(axis=1).round(6).to_dict(),
        "chosen_rank_per_fold": ranks.loc[chosen].astype(int).tolist(),
        "chosen_vs_default_mean_diff": float(piv.loc[chosen].mean() - piv.loc[0].mean()) if 0 in piv.index else None,
    })
    print(piv)


if __name__ == "__main__":
    main()
