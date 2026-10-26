"""Final evaluation, part 1: test predictions from the SAVED final models (no refitting).

For every runs/final_* directory with config.json, load the saved model (model.txt for
LightGBM, saved at best_iteration; model.pkl for Ridge), read only the test rows of the
run's table, and write predictions_test.parquet in that run directory. Refuses to
overwrite an existing predictions_test.parquet. Nothing about the models is changed.

Usage:
    PYTHONPATH=. python scripts/lgbm_cloud/predict_test.py
"""

from __future__ import annotations

import json
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import EXPECTED_SPLITS, FEATURES_DIR, RUNS_DIR, write_json
from train_tabular import PRED_COLS


def main():
    runs = sorted(d for d in RUNS_DIR.glob("final_*") if (d / "config.json").exists())
    tables = {}
    log = {}
    for d in runs:
        out = d / "predictions_test.parquet"
        if out.exists():
            raise FileExistsError(f"{out} exists; refusing to overwrite")
        cfg = json.loads((d / "config.json").read_text())
        feats, table = cfg["features"], cfg["table"]
        if table not in tables:
            cols = list(dict.fromkeys(PRED_COLS + [c for r in runs for c in json.loads((r / "config.json").read_text())["features"]
                                                  if json.loads((r / "config.json").read_text())["table"] == table]))
            t = pd.read_parquet(FEATURES_DIR / table, filters=[("split", "==", "test")])
            t = t.sort_values("split_pos").reset_index(drop=True)
            if len(t) != EXPECTED_SPLITS["test"] or not (t["split_pos"].to_numpy() == np.arange(len(t))).all():
                raise RuntimeError(f"{table}: test rows {len(t)} / order mismatch")
            tables[table] = t
        t = tables[table]
        if cfg["model"] == "lgbm":
            model = lgb.Booster(model_file=str(d / "model.txt"))
            if model.num_feature() != len(feats) or model.feature_name() != feats:
                raise RuntimeError(f"{d.name}: model features differ from config")
            k = model.predict(t[feats])
        else:
            with open(d / "model.pkl", "rb") as f:
                model = pickle.load(f)
            k = model.predict(t[feats])
        pred = t[PRED_COLS].copy()
        pred["k_pred"] = k
        pred["ghi_pred"] = k * pred["ghi_cs_t10"]
        pred.to_parquet(out, index=False)
        log[d.name] = {"n": int(len(pred)), "k_pred_nan": int(np.isnan(k).sum()), "table": table}
        print(d.name, log[d.name], flush=True)
    write_json(RUNS_DIR / "predict_test_log.json", log)


if __name__ == "__main__":
    main()
