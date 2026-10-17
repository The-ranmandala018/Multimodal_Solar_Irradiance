"""Train one Ridge or LightGBM run on a tabular feature set.

Train split only for fitting. The validation split, minus its first day (purge gap
after the last training sample), is used for early stopping and model selection.
The test split is NOT predicted unless --predict-test is given (Step 6 only).

Target: k(t+10) (clipped k_index, as in FolsomDataset). GHI = k_pred * GHI_cs(t+10),
not clipped, the same as scripts/evaluate_test.py.

Output: experiments/lightgbm_cloud_features/runs/<run_name>/
    config.json, metrics_val.json, predictions_val.parquet, model file, train.log
    (+ predictions_test.parquet with --predict-test; test metrics are computed only
     by the evaluation script)

Usage (Step 3 check):
    PYTHONPATH=. python scripts/lgbm_cloud/train_tabular.py --model ridge --features A --run-name check_ridge_A
    PYTHONPATH=. python scripts/lgbm_cloud/train_tabular.py --model lgbm  --features A --run-name check_lgbm_A
"""

from __future__ import annotations

import argparse
import json
import pickle
import time

import numpy as np
import pandas as pd

from common import (FEATURES_DIR, RUNS_DIR, git_head, refuse_existing, regression_metrics, versions,
                    write_json)
from feature_sets import TABLE_OF, get_feature_set

PURGE = pd.Timedelta(days=1)
PRED_COLS = ["datetime", "split", "split_pos", "k_true", "ghi_true", "ghi_cs_t10", "has_image",
             "sky_class", "test_period"]

LGBM_DEFAULTS = {
    "objective": "regression",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 500,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 255,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 32,
    "verbosity": -1,
}
RIDGE_ALPHAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["ridge", "lgbm"], required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--params", default="{}", help="JSON overrides for LightGBM parameters")
    p.add_argument("--num-boost-round", type=int, default=5000)
    p.add_argument("--early-stopping", type=int, default=200)
    p.add_argument("--predict-test", action="store_true")
    return p.parse_args()


def load(features, table, with_test=False):
    """Train and validation rows; test rows are only read when with_test is True (final evaluation)."""
    cols = list(dict.fromkeys(PRED_COLS + features + ["clearsky_persistence"]))
    keep = ["train", "val", "test"] if with_test else ["train", "val"]
    df = pd.read_parquet(FEATURES_DIR / table, columns=cols, filters=[("split", "in", keep)])
    train = df[df["split"] == "train"].reset_index(drop=True)
    val = df[df["split"] == "val"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True) if with_test else None
    purge_end = val["datetime"].min() + PURGE
    val_sel = val[val["datetime"] >= purge_end].reset_index(drop=True)
    return train, val, val_sel, test


def val_metrics(frame, k_pred):
    out = {"k_MSE": float(np.mean((k_pred - frame["k_true"].to_numpy()) ** 2))}
    out.update(regression_metrics(frame["ghi_true"], k_pred * frame["ghi_cs_t10"].to_numpy()))
    out["clearsky_persistence_RMSE_Wm2"] = regression_metrics(frame["ghi_true"], frame["clearsky_persistence"])["RMSE_Wm2"]
    out["skill_vs_clearsky_persistence"] = 1.0 - out["RMSE_Wm2"] / out["clearsky_persistence_RMSE_Wm2"]
    return out


def fit_ridge(train, val_sel, features):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    best = None
    trials = []
    for alpha in RIDGE_ALPHAS:
        model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                              Ridge(alpha=alpha))
        model.fit(train[features], train["k_true"])
        mse = float(np.mean((model.predict(val_sel[features]) - val_sel["k_true"]) ** 2))
        trials.append({"alpha": alpha, "val_purged_k_MSE": mse})
        if best is None or mse < best[1]:
            best = (model, mse, alpha)
    return best[0], {"alpha": best[2], "alpha_search": trials}


def fit_lgbm(train, val_sel, features, seed, overrides, num_boost_round, early_stopping):
    import lightgbm as lgb

    params = dict(LGBM_DEFAULTS)
    params.update(overrides)
    params.update({"seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed, "data_random_seed": seed})
    dtrain = lgb.Dataset(train[features], train["k_true"], free_raw_data=False)
    dval = lgb.Dataset(val_sel[features], val_sel["k_true"], reference=dtrain)
    log = {}
    model = lgb.train(params, dtrain, num_boost_round=num_boost_round, valid_sets=[dval], valid_names=["val"],
                      callbacks=[lgb.early_stopping(early_stopping, verbose=False), lgb.record_evaluation(log)])
    return model, {"params": params, "best_iteration": int(model.best_iteration),
                   "val_l2_curve_every_100": log["val"]["l2"][::100]}


def predict(model, kind, frame, features):
    if kind == "lgbm":
        return model.predict(frame[features], num_iteration=model.best_iteration)
    return model.predict(frame[features])


def main():
    args = parse_args()
    features = get_feature_set(args.features)
    out_dir = refuse_existing(RUNS_DIR / args.run_name)
    t0 = time.time()

    train, val, val_sel, test = load(features, TABLE_OF[args.features], with_test=args.predict_test)
    if args.model == "ridge":
        model, fit_info = fit_ridge(train, val_sel, features)
    else:
        model, fit_info = fit_lgbm(train, val_sel, features, args.seed, json.loads(args.params),
                                   args.num_boost_round, args.early_stopping)
    fit_seconds = time.time() - t0

    out_dir.mkdir(parents=True)
    k_val = predict(model, args.model, val, features)
    pred_val = val[PRED_COLS].copy()
    pred_val["k_pred"] = k_val
    pred_val["ghi_pred"] = k_val * pred_val["ghi_cs_t10"]
    pred_val["in_purged_val"] = pred_val["datetime"] >= val_sel["datetime"].min()
    pred_val.to_parquet(out_dir / "predictions_val.parquet", index=False)

    purged = pred_val["in_purged_val"].to_numpy()
    metrics = {"val_full": val_metrics(val, k_val), "val_purged": val_metrics(val_sel, k_val[purged])}
    by_month = {}
    for month, idx in val.groupby(val["datetime"].dt.to_period("M").astype(str)).groups.items():
        g = val.loc[idx]
        by_month[month] = val_metrics(g, k_val[idx])
    metrics["val_full_by_month"] = by_month
    write_json(out_dir / "metrics_val.json", metrics)

    if args.predict_test:
        k_test = predict(model, args.model, test, features)
        pred_test = test[PRED_COLS].copy()
        pred_test["k_pred"] = k_test
        pred_test["ghi_pred"] = k_test * pred_test["ghi_cs_t10"]
        pred_test.to_parquet(out_dir / "predictions_test.parquet", index=False)

    if args.model == "lgbm":
        model.save_model(str(out_dir / "model.txt"), num_iteration=model.best_iteration)
        imp = pd.DataFrame({"feature": features,
                            "gain": model.feature_importance("gain"),
                            "split": model.feature_importance("split")}).sort_values("gain", ascending=False)
        imp.to_csv(out_dir / "feature_importance.csv", index=False)
    else:
        with open(out_dir / "model.pkl", "wb") as f:
            pickle.dump(model, f)

    write_json(out_dir / "config.json", {
        "run_name": args.run_name, "model": args.model, "feature_set": args.features, "table": TABLE_OF[args.features],
        "features": features,
        "seed": args.seed, "fit": fit_info, "predict_test": args.predict_test,
        "n_train": len(train), "n_val_full": len(val), "n_val_purged": len(val_sel),
        "purge": str(PURGE), "val_purged_start": str(val_sel["datetime"].min()),
        "fit_seconds": round(fit_seconds, 1), "git_head": git_head(), "versions": versions(),
    })
    print(json.dumps({"run": args.run_name, "val_full": metrics["val_full"], "val_purged": metrics["val_purged"],
                      "fit_seconds": round(fit_seconds, 1),
                      **({"best_iteration": fit_info["best_iteration"]} if args.model == "lgbm" else {"alpha": fit_info["alpha"]})},
                     indent=1))


if __name__ == "__main__":
    main()
