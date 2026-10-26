"""Recompute image-vs-t offsets using file modified time (true clock), read-only."""
import json, sys
import numpy as np, pandas as pd
OUT = sys.argv[1]
tab = pd.read_pickle(f"{OUT}/sample_alignment.pkl")
img = pd.read_pickle(f"{OUT}/image_index.pkl")
img = img.sort_values("ts_name")
# CNN's chosen image: filename time = t + cnn_img_offset_s
tab["cnn_img_name"] = tab["datetime"] + pd.to_timedelta(tab["cnn_img_offset_s"], unit="s")
m = tab.merge(img[["ts_name", "ts_mtime"]].drop_duplicates("ts_name"),
              left_on="cnn_img_name", right_on="ts_name", how="left")
m["cnn_true_offset_s"] = (m["ts_mtime"] - m["datetime"]).dt.total_seconds()
# Per-day median correction (robust to per-file mtime jitter)
img["day"] = img["day_dir"]
img["day_med_off"] = img.groupby("day")["name_minus_mtime_s"].transform("median")
img["ts_corr"] = img["ts_name"] - pd.to_timedelta(img["day_med_off"], unit="s")
img["resid_s"] = (img["ts_corr"] - img["ts_mtime"]).dt.total_seconds()
# Latest image with mtime <= t and >= t-120 s
s = img[["ts_mtime"]].sort_values("ts_mtime")["ts_mtime"].values
t = m["datetime"].values
pos = np.searchsorted(s, t, side="right") - 1
ok = pos >= 0
age = np.full(len(t), np.nan)
age[ok] = (t[ok] - s[pos[ok]]) / np.timedelta64(1, "s")
age[age > 120] = np.nan
m["le_true_age_s"] = age
m["cnn_true_after_t"] = m["cnn_true_offset_s"] > 0
m["common_true"] = (~m["cnn_true_after_t"]) & m["le_true_age_s"].notna() & \
    np.isclose(m["le_true_age_s"], -m["cnn_true_offset_s"], atol=0.5)
rep = {"unmatched_cnn_images": int(m["ts_mtime"].isna().sum()),
       "per_image_resid_vs_daymedian_s": {q: float(img["resid_s"].abs().quantile(q)) for q in (.5, .9, .99, .999)}}
for sp, g in m.groupby("split"):
    off = g["cnn_true_offset_s"]
    rep[sp] = {
        "n": int(len(g)),
        "cnn_true_offset_quantiles_s": {q: float(off.quantile(q)) for q in (0, .01, .05, .25, .5, .75, .95, .99, 1)},
        "cnn_true_after_t_n": int((off > 0).sum()), "cnn_true_after_t_frac": float((off > 0).mean()),
        "cnn_true_after_t_gt30s_n": int((off > 30).sum()),
        "cnn_true_age_gt5min_frac": float((off < -300).mean()),
        "le_true_available_frac": float(g["le_true_age_s"].notna().mean()),
        "le_true_age_median_s": float(g["le_true_age_s"].median()),
        "common_true_n": int(g["common_true"].sum()),
    }
mon = m[m.split == "test"].groupby(m["datetime"].dt.to_period("M").astype(str))["cnn_true_offset_s"].median()
rep["test_cnn_true_offset_median_by_month_s"] = mon.round(0).to_dict()
mon = m[m.split != "test"].groupby(m["datetime"].dt.to_period("M").astype(str))["cnn_true_offset_s"].median()
rep["trainval_cnn_true_offset_median_by_month_s"] = mon.round(0).to_dict()
m.drop(columns=["ts_name"]).to_pickle(f"{OUT}/sample_alignment_true.pkl")
json.dump(rep, open(f"{OUT}/p1_true_offsets.json", "w"), indent=1)
print(json.dumps(rep, indent=1))
