"""Phase 1 (read-only): clock / time-zone checks.

(1) CSV clock: on clear days, fit GHI(t) ~ a * GHI_cs(t + L) and report the best L.
    L ~ 0 means the CSV is in UTC and its clock agrees with pvlib.
(2) Image label check: correlate first differences of whole-sky image brightness
    with first differences of measured k at lags -20..+20 min, using the image
    filename time and the file modified time as labels. The label whose peak sits
    at 0 is the one aligned with the irradiance data.
    Sign: peak at lag L means k changes L minutes after the labelled image change,
    so the label is ahead of the irradiance clock by -L minutes.
Writes to the scratchpad only.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

OUT = Path(sys.argv[1])
CSV = "/storage2/CV_Irradiance/datasets/1_Folsom/csv_files/Folsom_irradiance_weather.csv"
FEAT = "/storage2/CV_Irradiance/datasets/1_Folsom/csv_files/Folsom_sky_image_features.csv"
site = pvlib.location.Location(38.642, -121.148, altitude=60.0)

df = pd.read_csv(CSV, parse_dates=["Datetime"]).set_index("Datetime").sort_index()
df = df[~df.index.duplicated()]
full = pd.date_range(df.index.min().floor("D"), df.index.max().ceil("D"), freq="1min")
df = df.reindex(full)
sp = site.get_solarposition(full)
cs = site.get_clearsky(full, model="ineichen")["ghi"]
df["SZA"], df["cs"] = sp["zenith"].values, cs.values
df["k"] = (df["GHI"] / (df["cs"] + 1e-6)).clip(0, 1.2)
df.loc[df["SZA"] > 85, "k"] = np.nan

rep = {}

# ---- (1) CSV clock on clear days ------------------------------------------
lags = np.arange(-60, 60.5, 0.5)
cs_series = pd.Series(df["cs"].values, index=full)
day_rows = []
for d, g in df.groupby(df.index.date):
    g = g[g["SZA"] < 80]
    if len(g) < 300 or g["GHI"].isna().mean() > 0.05:
        continue
    k = g["k"]
    if not (k.mean() > 0.85 and k.diff().abs().quantile(0.99) < 0.03 and k.std() < 0.06):
        continue
    t_sec = (g.index - full[0]).total_seconds().values / 60.0
    best = None
    for L in lags:
        c = np.interp(t_sec + L, np.arange(len(full)), cs_series.values)
        y = g["GHI"].values
        a = np.dot(y, c) / np.dot(c, c)
        sse = np.sum((y - a * c) ** 2)
        if best is None or sse < best[1]:
            best = (L, sse, a)
    day_rows.append({"date": str(d), "best_lag_min": best[0], "scale": best[2]})
cd = pd.DataFrame(day_rows)
cd["month"] = cd["date"].str[:7]
cd.to_csv(OUT / "csv_clock_clear_days.csv", index=False)
rep["csv_clock"] = {
    "n_clear_days": int(len(cd)),
    "best_lag_median_min": float(cd["best_lag_min"].median()),
    "best_lag_iqr_min": [float(cd["best_lag_min"].quantile(.25)), float(cd["best_lag_min"].quantile(.75))],
    "by_year_median": {y: float(g["best_lag_min"].median()) for y, g in cd.groupby(cd["date"].str[:4])},
    "by_month_median": {m: float(g["best_lag_min"].median()) for m, g in cd.groupby("month")},
}

# ---- (2) image brightness vs k --------------------------------------------
idx = pd.read_pickle(OUT / "image_index.pkl")[["ts_name", "ts_mtime"]]
feat = pd.read_csv(FEAT, parse_dates=["timestamp"])
feat["bright"] = feat[["AVG(R)", "AVG(G)", "AVG(B)"]].mean(axis=1)
feat = feat.merge(idx, left_on="timestamp", right_on="ts_name", how="inner")
rep["image_features_rows"] = int(len(feat))

k_min = df["k"]
dk = k_min.diff()

LAGS = np.arange(-20, 21)


def lag_curve(label_col, mask_months=None):
    s = feat[[label_col, "bright"]].dropna().copy()
    s["minute"] = s[label_col].dt.round("1min")
    b = s.groupby("minute")["bright"].mean().reindex(full)
    db = b.diff()
    out = {}
    for L in LAGS:
        x = db
        y = dk.shift(-L)  # k change L minutes after the labelled image change
        m = x.notna() & y.notna()
        if mask_months is not None:
            m &= mask_months
        out[int(L)] = float(np.corrcoef(x[m], y[m])[0, 1]) if m.sum() > 1000 else np.nan
    return out


month_of = pd.Series(full.to_period("M").astype(str), index=full)
res = {}
for label in ("ts_name", "ts_mtime"):
    res[label] = {"all": lag_curve(label)}
    for m in sorted(month_of.unique()):
        mm = (month_of == m).values
        if mm.sum() == 0:
            continue
        res[label][m] = lag_curve(label, pd.Series(mm, index=full))
peaks = []
for label, curves in res.items():
    for m, c in curves.items():
        if all(np.isnan(v) for v in c.values()):
            continue
        L = max(c, key=lambda k: (-np.inf if np.isnan(c[k]) else c[k]))
        peaks.append({"label": label, "month": m, "peak_lag_min": L, "peak_corr": c[L], "corr_at_0": c[0]})
pk = pd.DataFrame(peaks)
pk.to_csv(OUT / "image_vs_k_lag_peaks.csv", index=False)
(OUT / "image_vs_k_lag_curves.json").write_text(json.dumps(res))
rep["lag_peaks_all"] = pk[pk["month"] == "all"].to_dict("records")
rep["lag_peak_by_label_month"] = pk[pk["month"] != "all"].pivot(index="month", columns="label", values="peak_lag_min").to_dict()

(OUT / "p1_lags.json").write_text(json.dumps(rep, indent=1, default=str))
print(json.dumps(rep, indent=1, default=str))
