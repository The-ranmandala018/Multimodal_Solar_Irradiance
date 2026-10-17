"""Step 7 analysis on VALIDATION only (the 2016 test set is not loaded).

Inputs: runs/final_<model>_<set>_seed<k>/predictions_val.parquet (+ model.txt for SHAP).
Scored on the purged validation set (first day removed), with GHI = k_pred * GHI_cs(t+10)
as in evaluate_test.py.

Outputs (experiments/lightgbm_cloud_features/summary/):
    val_metrics_overall.csv      per run: seed-averaged metrics and mean +- sd over seeds
    val_metrics_by_sky.csv       seed-averaged, by k-based sky class
    val_metrics_by_month.csv     seed-averaged, by month
    val_paired_bootstrap.csv     day-block bootstrap (95 %) of MAE / RMSE differences
    val_shap_groups.csv          mean |SHAP| by feature group (LGBM-B, seed 0, 30k val rows)
    val_report.json
"""

from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import PROJECT_DIR, RUNS_DIR, SUMMARY_DIR, git_head, regression_metrics, write_json
from feature_sets import GROUP_OF, TABLE_OF, get_feature_set
from common import FEATURES_DIR

N_BOOT = 2000
RUNS = {  # name -> (run-dir prefix, number of seeds)
    "Ridge-A": ("final_ridge_A", 1), "Ridge-B": ("final_ridge_B", 1),
    "LGBM-A0": ("final_lgbm_A0", 5), "LGBM-A": ("final_lgbm_A", 5), "LGBM-B": ("final_lgbm_B", 5),
    "LGBM-C": ("final_lgbm_C", 5), "LGBM-B-parity": ("final_lgbm_B_parity", 5),
}
PAIRS = [("LGBM-B", "LGBM-A"), ("LGBM-B", "Ridge-B"), ("Ridge-B", "Ridge-A"), ("LGBM-A0", "LGBM-A"),
         ("LGBM-C", "LGBM-A"), ("LGBM-B-parity", "LGBM-B")]


def load_runs():
    preds, per_seed = {}, {}
    for name, (prefix, n) in RUNS.items():
        dirs = [RUNS_DIR / (f"{prefix}_seed{k}" if n > 1 else prefix) for k in range(n)]
        frames = [pd.read_parquet(d / "predictions_val.parquet") for d in dirs if d.exists()]
        if not frames:
            continue
        base = frames[0]
        for f in frames[1:]:
            if not (f["datetime"].to_numpy() == base["datetime"].to_numpy()).all():
                raise RuntimeError(f"{name}: seed predictions not aligned")
        k = np.mean([f["k_pred"].to_numpy() for f in frames], axis=0)
        df = base[["datetime", "k_true", "ghi_true", "ghi_cs_t10", "sky_class", "in_purged_val"]].copy()
        df["ghi_pred"] = k * df["ghi_cs_t10"]
        preds[name] = df[df["in_purged_val"]].reset_index(drop=True)
        per_seed[name] = [regression_metrics(f.loc[f["in_purged_val"], "ghi_true"],
                                             f.loc[f["in_purged_val"], "ghi_pred"]) for f in frames]
    return preds, per_seed


def boot_diff(a, b, days, n=N_BOOT, seed=0):
    """Day-block bootstrap of MAE(a) - MAE(b) and RMSE(a) - RMSE(b)."""
    ea, eb = a["ghi_pred"] - a["ghi_true"], b["ghi_pred"] - b["ghi_true"]
    g = pd.DataFrame({"d": days, "aa": ea.abs(), "ab": eb.abs(), "sa": ea ** 2, "sb": eb ** 2})
    per = g.groupby("d").agg(n=("aa", "size"), aa=("aa", "sum"), ab=("ab", "sum"), sa=("sa", "sum"), sb=("sb", "sum"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per), size=(n, len(per)))
    arr = per.to_numpy()
    s = arr[idx].sum(axis=1)  # n x 5
    mae = (s[:, 1] - s[:, 2]) / s[:, 0]
    rmse = np.sqrt(s[:, 3] / s[:, 0]) - np.sqrt(s[:, 4] / s[:, 0])
    point_mae = float(ea.abs().mean() - eb.abs().mean())
    point_rmse = float(np.sqrt((ea ** 2).mean()) - np.sqrt((eb ** 2).mean()))
    return {"dMAE": point_mae, "dMAE_lo": float(np.quantile(mae, 0.025)), "dMAE_hi": float(np.quantile(mae, 0.975)),
            "dRMSE": point_rmse, "dRMSE_lo": float(np.quantile(rmse, 0.025)),
            "dRMSE_hi": float(np.quantile(rmse, 0.975)), "n_days": int(len(per))}


def main():
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    preds, per_seed = load_runs()
    any_df = next(iter(preds.values()))
    cs = regression_metrics(any_df["ghi_true"], (
        pd.read_parquet(FEATURES_DIR / "samples_setA.parquet", columns=["datetime", "clearsky_persistence"],
                        filters=[("split", "==", "val")]).set_index("datetime")
        .loc[any_df["datetime"], "clearsky_persistence"].to_numpy()))

    rows = []
    for name, df in preds.items():
        m = regression_metrics(df["ghi_true"], df["ghi_pred"])
        m["k_MSE"] = float(np.mean((df["ghi_pred"] / df["ghi_cs_t10"] - df["k_true"]) ** 2))
        m["skill_vs_clearsky_persistence_RMSE"] = 1 - m["RMSE_Wm2"] / cs["RMSE_Wm2"]
        seeds = per_seed[name]
        m["seeds"] = len(seeds)
        m["RMSE_seed_mean"] = float(np.mean([s["RMSE_Wm2"] for s in seeds]))
        m["RMSE_seed_sd"] = float(np.std([s["RMSE_Wm2"] for s in seeds], ddof=1)) if len(seeds) > 1 else 0.0
        m["MAE_seed_mean"] = float(np.mean([s["MAE_Wm2"] for s in seeds]))
        m["MAE_seed_sd"] = float(np.std([s["MAE_Wm2"] for s in seeds], ddof=1)) if len(seeds) > 1 else 0.0
        rows.append({"run": name, **m})
    rows.append({"run": "Clear-sky persistence", **cs})
    overall = pd.DataFrame(rows)
    overall.to_csv(SUMMARY_DIR / "val_metrics_overall.csv", index=False)

    by_sky, by_month = [], []
    for name, df in preds.items():
        for cls, g in df.groupby("sky_class"):
            by_sky.append({"run": name, "sky_class": cls, **regression_metrics(g["ghi_true"], g["ghi_pred"])})
        for mon, g in df.groupby(df["datetime"].dt.to_period("M").astype(str)):
            by_month.append({"run": name, "month": mon, **regression_metrics(g["ghi_true"], g["ghi_pred"])})
    pd.DataFrame(by_sky).to_csv(SUMMARY_DIR / "val_metrics_by_sky.csv", index=False)
    pd.DataFrame(by_month).to_csv(SUMMARY_DIR / "val_metrics_by_month.csv", index=False)

    boots = []
    for a, b in PAIRS:
        if a not in preds or b not in preds:
            continue
        A, B = preds[a], preds[b]
        days = A["datetime"].dt.date.to_numpy()
        boots.append({"comparison": f"{a} - {b}", "subset": "all", **boot_diff(A, B, days)})
        for cls in ("clear", "mixed", "overcast"):
            m = (A["sky_class"] == cls).to_numpy()
            boots.append({"comparison": f"{a} - {b}", "subset": cls,
                          **boot_diff(A[m].reset_index(drop=True), B[m].reset_index(drop=True), days[m])})
    boot = pd.DataFrame(boots)
    boot.to_csv(SUMMARY_DIR / "val_paired_bootstrap.csv", index=False)

    shap_rows = []
    model_dir = RUNS_DIR / "final_lgbm_B_seed0"
    if (model_dir / "model.txt").exists():
        feats = get_feature_set("B")
        model = lgb.Booster(model_file=str(model_dir / "model.txt"))
        v = pd.read_parquet(FEATURES_DIR / TABLE_OF["B"], columns=["datetime", "split", "sky_class"] + feats,
                            filters=[("split", "==", "val")])
        v = v[v["datetime"] >= v["datetime"].min() + pd.Timedelta(days=1)].sample(30000, random_state=0)
        contrib = model.predict(v[feats], pred_contrib=True)[:, :-1]
        for subset, mask in (("all", np.ones(len(v), bool)), *[(c, (v["sky_class"] == c).to_numpy())
                                                                for c in ("clear", "mixed", "overcast")]):
            mabs = np.abs(contrib[mask]).mean(axis=0)
            s = pd.Series(mabs, index=feats).groupby(lambda c: GROUP_OF[c]).sum()
            for grp, val in s.items():
                shap_rows.append({"subset": subset, "group": grp, "mean_abs_shap_k": float(val),
                                  "share": float(val / s.sum())})
        pd.DataFrame(shap_rows).to_csv(SUMMARY_DIR / "val_shap_groups.csv", index=False)

    lstm = {}
    for r in ("weather_only_seed0", "weather_only_seed1", "weather_only_seed2", "baseline_lstm_cnn",
              "cnn_lstm_seed1", "cnn_lstm_seed2"):
        f = PROJECT_DIR / "experiments" / r / "evaluation" / "test_metrics.csv"
        if f.exists():
            t = pd.read_csv(f).set_index("metric")["value"]
            lstm[r] = float(t["checkpoint_val_loss"])
    write_json(SUMMARY_DIR / "val_report.json", {
        "git_head": git_head(), "note": "validation only (purged first day); test not loaded",
        "existing_models_val_k_MSE_full_val_best_checkpoint": lstm,
        "overall": overall.to_dict(orient="records"), "bootstrap": boot.to_dict(orient="records"),
        "shap_groups": shap_rows})
    pd.set_option("display.width", 200)
    print(overall[["run", "RMSE_Wm2", "MAE_Wm2", "MBE_Wm2", "k_MSE", "RMSE_seed_sd", "MAE_seed_sd"]].round(4))
    print(boot.round(3).to_string(index=False))
    if shap_rows:
        print(pd.DataFrame(shap_rows).pivot(index="group", columns="subset", values="share").round(3))
    print("existing models, val k-MSE (full val):", lstm)


if __name__ == "__main__":
    main()
