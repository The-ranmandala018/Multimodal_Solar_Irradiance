"""Validation-only check of LGBM-B vs LGBM-A by k-based sky class and solar-zenith band.

Saved verbatim from the inline command that produced summary/val_B_vs_A_by_sky_and_sza.csv
(run 2026-10-17 20:56), so that file can be regenerated. Seed-averaged predictions on the
purged validation set. Writes to the path given as argument (default: the summary file;
refuses to overwrite).
"""

import sys

import numpy as np
import pandas as pd

from common import EXP_DIR, refuse_existing


def avg(prefix):
    fr = [pd.read_parquet(EXP_DIR / f"runs/{prefix}_seed{k}/predictions_val.parquet") for k in range(5)]
    d = fr[0][["datetime", "ghi_true", "ghi_cs_t10", "sky_class", "in_purged_val"]].copy()
    d["k"] = np.mean([f.k_pred for f in fr], 0)
    d["ghi_pred"] = d.k * d.ghi_cs_t10
    return d[d.in_purged_val]


def main():
    out = refuse_existing(sys.argv[1] if len(sys.argv) > 1 else EXP_DIR / "summary/val_B_vs_A_by_sky_and_sza.csv")
    A, B = avg("final_lgbm_A"), avg("final_lgbm_B")
    sza = pd.read_parquet(EXP_DIR / "features/samples_B.parquet", columns=["datetime", "split", "sza_t", "cf2_all"],
                          filters=[("split", "==", "val")]).set_index("datetime").loc[A.datetime]
    A["sza"] = B["sza"] = sza.sza_t.values
    B["cf2"] = sza.cf2_all.values
    A["band"] = B["band"] = pd.cut(A.sza, [0, 50, 60, 70, 85])
    rows = []
    for (cls, band), ga in A.groupby(["sky_class", "band"], observed=True):
        gb = B.loc[ga.index]
        rows.append(dict(sky=cls, sza=str(band), n=len(ga), MAE_A=(ga.ghi_pred - ga.ghi_true).abs().mean(),
                         MAE_B=(gb.ghi_pred - gb.ghi_true).abs().mean(),
                         MBE_B_minus_A=((gb.ghi_pred - gb.ghi_true).mean() - (ga.ghi_pred - ga.ghi_true).mean()),
                         cf2_median=gb.cf2.median()))
    r = pd.DataFrame(rows)
    r["dMAE"] = r.MAE_B - r.MAE_A
    r.to_csv(out, index=False)
    print(r.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
