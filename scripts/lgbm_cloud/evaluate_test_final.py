"""Final evaluation, part 2: one-time scoring on the 2016 test set (protocol fixed before test).

Protocol
  - Models: the saved final runs (selected on validation only). LightGBM predictions are
    averaged over 5 seeds; per-seed metrics are also reported. Ridge has one run.
  - Existing models: saved test predictions of weather_only_seed0-2, CNN+LSTM
    (baseline_lstm_cnn, cnn_lstm_seed1, cnn_lstm_seed2) and image_only_seed0, reordered to
    time order (multi-GPU rank order handled as in analysis_bias_and_sky_conditions.py)
    and verified sample-by-sample against our targets (|diff| <= 0.05 W/m2). Each family is
    also averaged over its runs; per-run metrics reported.
  - Same 224,022 test samples for every model; GHI = k * GHI_cs(t+10), predictions not
    clipped (as evaluate_test.py). Metrics: MAE, RMSE, MBE, R2, nMAE, nRMSE; forecast skill
    FS = 1 - RMSE / RMSE_baseline vs clear-sky and naive persistence.
  - Breakdowns: Jan-Aug vs Sep-Dec, k-based sky class, solar-zenith band x sky, month.
  - Paired day-block bootstrap (2000 resamples over test days) of MAE/RMSE differences.
  - Summer low-sun haze check: Jun-Aug clear sky by SZA band, and clear-sky samples whose
    image shows cf2 > 0.15 (likely false cloud).
  Nothing here feeds back into any model, mask, threshold or calibration.

Outputs: experiments/lightgbm_cloud_features/summary/test_*.csv, test_report.json (refuses to overwrite)
"""

from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from analyze_val import boot_diff
from common import FEATURES_DIR, PROJECT_DIR, RUNS_DIR, SUMMARY_DIR, git_head, refuse_existing, regression_metrics, write_json

LGBM = {"LGBM-A": "final_lgbm_A", "LGBM-A0": "final_lgbm_A0", "LGBM-B": "final_lgbm_B", "LGBM-C": "final_lgbm_C",
        "LGBM-B-parity": "final_lgbm_B_parity"}
RIDGE = {"Ridge-A": "final_ridge_A", "Ridge-B": "final_ridge_B"}
EXISTING = {"Weather-only LSTM": ["weather_only_seed0", "weather_only_seed1", "weather_only_seed2"],
            "CNN+LSTM": ["baseline_lstm_cnn", "cnn_lstm_seed1", "cnn_lstm_seed2"],
            "Image-only CNN": ["image_only_seed0"]}
PAIRS = [("LGBM-B", "LGBM-A"), ("LGBM-B", "Ridge-B"), ("LGBM-B-parity", "LGBM-B"), ("LGBM-C", "LGBM-A"),
         ("LGBM-A0", "LGBM-A"), ("LGBM-A", "Weather-only LSTM"), ("LGBM-B", "CNN+LSTM"),
         ("LGBM-B-parity", "CNN+LSTM"), ("LGBM-B", "Weather-only LSTM")]
OUT_FILES = ["test_metrics_overall.csv", "test_metrics_per_run.csv", "test_metrics_by_period.csv",
             "test_metrics_by_sky.csv", "test_metrics_by_sza_sky.csv", "test_metrics_by_month.csv",
             "test_paired_bootstrap.csv", "test_haze_check.csv", "test_report.json"]
SZA_BANDS = [0, 30, 50, 60, 70, 85]


def reorder_saved(path, y_ref):
    """Saved existing-model predictions in time order (same logic as analysis_bias_and_sky_conditions)."""
    d = np.load(path)
    y_true, y_pred = d["ghi_clipped_true"].astype(np.float64), d["ghi_clipped_pred"].astype(np.float64)
    n = len(y_ref)
    if len(y_true) != n:
        raise RuntimeError(f"{path}: {len(y_true)} samples, expected {n}")
    for world in range(1, 5):
        order = np.concatenate([np.arange(r, n, world) for r in range(world)])
        inv = np.empty(n, dtype=np.int64)
        inv[order] = np.arange(n)
        if np.allclose(y_true[inv], y_ref, atol=0.05):
            return y_pred[inv], world, float(np.max(np.abs(y_true[inv] - y_ref)))
    raise RuntimeError(f"{path}: targets do not match in any rank order")


def check_saved_models_reproduce_val():
    """Each saved LightGBM model must reproduce its own validation predictions."""
    worst = 0.0
    feats_cache = {}
    for d in sorted(RUNS_DIR.glob("final_lgbm_*")):
        cfg = json.loads((d / "config.json").read_text())
        key = cfg["table"]
        if key not in feats_cache:
            feats_cache[key] = pd.read_parquet(FEATURES_DIR / key, filters=[("split", "==", "val")]).sort_values("split_pos")
        v = feats_cache[key]
        k = lgb.Booster(model_file=str(d / "model.txt")).predict(v[cfg["features"]])
        saved = pd.read_parquet(d / "predictions_val.parquet").sort_values("split_pos")["k_pred"].to_numpy()
        worst = max(worst, float(np.max(np.abs(k - saved))))
    if worst > 1e-9:
        raise RuntimeError(f"saved models do not reproduce validation predictions (max diff {worst})")
    return worst


def main():
    for f in OUT_FILES:
        refuse_existing(SUMMARY_DIR / f)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    val_repro = check_saved_models_reproduce_val()

    base = pd.read_parquet(FEATURES_DIR / "samples_B.parquet",
                           columns=["datetime", "split", "split_pos", "ghi_true", "ghi_cs_t10", "ghi_t", "sky_class",
                                    "sza_t", "test_period", "clearsky_persistence", "has_image", "cf2_all"],
                           filters=[("split", "==", "test")]).sort_values("split_pos").reset_index(drop=True)
    n = len(base)
    y = base["ghi_true"].to_numpy()

    preds, per_run, checks = {}, [], {"saved_models_reproduce_val_max_abs_diff": val_repro}
    for name, prefix in {**LGBM, **RIDGE}.items():
        dirs = [RUNS_DIR / f"{prefix}_seed{k}" for k in range(5)] if name in LGBM else [RUNS_DIR / prefix]
        ghis = []
        for d in dirs:
            p = pd.read_parquet(d / "predictions_test.parquet").sort_values("split_pos")
            if not np.array_equal(p["ghi_true"].to_numpy(), y):
                raise RuntimeError(f"{d.name}: test targets differ")
            ghis.append(p["ghi_pred"].to_numpy())
            per_run.append({"model": name, "run": d.name, **regression_metrics(y, ghis[-1])})
        preds[name] = np.mean(ghis, axis=0)
    for name, runs in EXISTING.items():
        ghis = []
        for r in runs:
            g, world, diff = reorder_saved(PROJECT_DIR / "experiments" / r / "evaluation" / "test_predictions.npz", y)
            checks[f"{r}_rank_world"] = world
            checks[f"{r}_target_max_abs_diff"] = diff
            ghis.append(g)
            per_run.append({"model": name, "run": r, **regression_metrics(y, g)})
        preds[name] = np.mean(ghis, axis=0)
    preds["Clear-sky persistence"] = base["clearsky_persistence"].to_numpy()
    preds["Naive persistence"] = base["ghi_t"].to_numpy()
    per_run = pd.DataFrame(per_run)
    per_run.to_csv(SUMMARY_DIR / "test_metrics_per_run.csv", index=False)

    def table(mask, extra):
        rows = []
        cs = regression_metrics(y[mask], preds["Clear-sky persistence"][mask])["RMSE_Wm2"]
        nv = regression_metrics(y[mask], preds["Naive persistence"][mask])["RMSE_Wm2"]
        for name, g in preds.items():
            m = regression_metrics(y[mask], g[mask])
            m["FS_vs_clearsky_persistence"] = 1 - m["RMSE_Wm2"] / cs
            m["FS_vs_naive_persistence"] = 1 - m["RMSE_Wm2"] / nv
            rows.append({**extra, "model": name, **m})
        return rows

    allm = np.ones(n, bool)
    overall = pd.DataFrame(table(allm, {}))
    spread = per_run.groupby("model").agg(runs=("run", "size"), RMSE_run_mean=("RMSE_Wm2", "mean"),
                                          RMSE_run_sd=("RMSE_Wm2", "std"), MAE_run_mean=("MAE_Wm2", "mean"),
                                          MAE_run_sd=("MAE_Wm2", "std")).reset_index()
    overall = overall.merge(spread, on="model", how="left")
    overall.to_csv(SUMMARY_DIR / "test_metrics_overall.csv", index=False)

    period = pd.DataFrame([r for p in ("jan_aug", "sep_dec") for r in table((base["test_period"] == p).to_numpy(), {"period": p})])
    period.to_csv(SUMMARY_DIR / "test_metrics_by_period.csv", index=False)
    sky = pd.DataFrame([r for c in ("clear", "mixed", "overcast") for r in table((base["sky_class"] == c).to_numpy(), {"sky_class": c})])
    sky.to_csv(SUMMARY_DIR / "test_metrics_by_sky.csv", index=False)
    band = pd.cut(base["sza_t"], SZA_BANDS).astype(str)
    sza = pd.DataFrame([r for c in ("clear", "mixed", "overcast") for b in sorted(band.unique())
                        for r in table(((base["sky_class"] == c) & (band == b)).to_numpy(), {"sky_class": c, "sza_band": b})])
    sza.to_csv(SUMMARY_DIR / "test_metrics_by_sza_sky.csv", index=False)
    month = base["datetime"].dt.to_period("M").astype(str)
    mon = pd.DataFrame([r for mm in sorted(month.unique()) for r in table((month == mm).to_numpy(), {"month": mm})])
    mon.to_csv(SUMMARY_DIR / "test_metrics_by_month.csv", index=False)

    days = base["datetime"].dt.date.to_numpy()
    subsets = {"all": allm, "jan_aug": (base["test_period"] == "jan_aug").to_numpy(),
               "sep_dec": (base["test_period"] == "sep_dec").to_numpy(),
               **{c: (base["sky_class"] == c).to_numpy() for c in ("clear", "mixed", "overcast")}}
    boots = []
    for a, b in PAIRS:
        for sname, m in subsets.items():
            A = pd.DataFrame({"ghi_pred": preds[a][m], "ghi_true": y[m]})
            B = pd.DataFrame({"ghi_pred": preds[b][m], "ghi_true": y[m]})
            boots.append({"comparison": f"{a} - {b}", "subset": sname, "n": int(m.sum()), **boot_diff(A, B, days[m])})
    boot = pd.DataFrame(boots)
    boot.to_csv(SUMMARY_DIR / "test_paired_bootstrap.csv", index=False)

    # summer low-sun haze check (Jun-Aug, clear sky)
    mo = base["datetime"].dt.month
    haze = []
    clear = (base["sky_class"] == "clear").to_numpy()
    for season, sm in (("summer_jun_aug", mo.isin([6, 7, 8]).to_numpy()), ("other_months", ~mo.isin([6, 7, 8]).to_numpy())):
        for b in ["(0, 50]", "(50, 60]", "(60, 70]", "(70, 85]"]:
            bb = pd.cut(base["sza_t"], [0, 50, 60, 70, 85]).astype(str).to_numpy() == b
            m = clear & sm & bb
            if m.sum() == 0:
                continue
            eA, eB = preds["LGBM-A"][m] - y[m], preds["LGBM-B"][m] - y[m]
            haze.append({"season": season, "sza_band": b, "subset": "clear", "n": int(m.sum()),
                         "cf2_median": float(base.loc[m, "cf2_all"].median()),
                         "frac_cf2_gt_0.15": float((base.loc[m, "cf2_all"] > 0.15).mean()),
                         "MAE_A": float(np.abs(eA).mean()), "MAE_B": float(np.abs(eB).mean()),
                         "MBE_A": float(eA.mean()), "MBE_B": float(eB.mean()),
                         **{f"boot_{k}": v for k, v in boot_diff(pd.DataFrame({"ghi_pred": preds["LGBM-B"][m], "ghi_true": y[m]}),
                                                                   pd.DataFrame({"ghi_pred": preds["LGBM-A"][m], "ghi_true": y[m]}),
                                                                   days[m]).items()}})
    for label, m in (("clear_cf2_gt_0.15", clear & (base["cf2_all"] > 0.15).to_numpy()),
                     ("clear_cf2_le_0.15", clear & (base["cf2_all"] <= 0.15).to_numpy()),
                     ("summer_clear_sza_gt60_cf2_gt_0.15", clear & mo.isin([6, 7, 8]).to_numpy() & (base["sza_t"] > 60).to_numpy()
                      & (base["cf2_all"] > 0.15).to_numpy())):
        eA, eB = preds["LGBM-A"][m] - y[m], preds["LGBM-B"][m] - y[m]
        haze.append({"season": "all" if not label.startswith("summer") else "summer_jun_aug", "sza_band": "all" if "sza" not in label else "(60, 85]",
                     "subset": label, "n": int(m.sum()), "cf2_median": float(base.loc[m, "cf2_all"].median()),
                     "frac_cf2_gt_0.15": float((base.loc[m, "cf2_all"] > 0.15).mean()),
                     "MAE_A": float(np.abs(eA).mean()), "MAE_B": float(np.abs(eB).mean()),
                     "MBE_A": float(eA.mean()), "MBE_B": float(eB.mean()),
                     **{f"boot_{k}": v for k, v in boot_diff(pd.DataFrame({"ghi_pred": preds["LGBM-B"][m], "ghi_true": y[m]}),
                                                               pd.DataFrame({"ghi_pred": preds["LGBM-A"][m], "ghi_true": y[m]}),
                                                               days[m]).items()}})
    haze = pd.DataFrame(haze)
    haze.to_csv(SUMMARY_DIR / "test_haze_check.csv", index=False)

    counts = {"test_samples": n, "jan_aug": int((base["test_period"] == "jan_aug").sum()),
              "sep_dec": int((base["test_period"] == "sep_dec").sum()),
              "sky_class": base["sky_class"].value_counts().to_dict(), "test_days": int(len(np.unique(days))),
              "has_image_le_t": int(base["has_image"].sum())}
    write_json(SUMMARY_DIR / "test_report.json", {
        "git_head": git_head(), "protocol": __doc__, "counts": counts, "checks": checks,
        "overall": overall.to_dict(orient="records"), "bootstrap": boot.to_dict(orient="records"),
        "haze": haze.to_dict(orient="records")})
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("counts:", counts)
    print("checks:", checks)
    print(overall[["model", "RMSE_Wm2", "MAE_Wm2", "MBE_Wm2", "R2", "FS_vs_clearsky_persistence", "FS_vs_naive_persistence",
                   "runs", "RMSE_run_mean", "RMSE_run_sd"]].round(3).to_string(index=False))
    print(period.pivot(index="model", columns="period", values="RMSE_Wm2").round(2))
    print(sky.pivot(index="model", columns="sky_class", values="MAE_Wm2").round(2))
    print(boot[boot["subset"].isin(["all", "jan_aug", "sep_dec", "clear", "mixed", "overcast"])].round(3).to_string(index=False))
    print(haze.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
