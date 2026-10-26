"""Clear-sky reference ("library") and cloud thresholds, TRAIN images only
-> features/clear_sky_library.npz and features/cloud_thresholds.json

Folsom JPEGs are washed out: clear-sky pixels have a red-blue ratio of ~0.95
(vs ~0.99 for overcast), so fixed RBR thresholds do not separate cloud from sky.
Each pixel is instead compared with the clear-sky value expected at its geometry:

    library[v][a, z, s] = median of v over strict-clear pixels in bin
        a = angle pixel-sun (10 deg bins, 0-180), z = pixel zenith (7.5 deg, 0-75),
        s = solar zenith (10 deg), v in {sat, gray, rbr}
    sat  = (max(RGB) - min(RGB)) / (max(RGB) + 1)    gray = mean(RGB)
    lookups interpolate linearly between bin centres (cloud_features.csl_lookup)

Strict-clear train images: sky_class == clear and std(k) over the trailing 30 min < 0.02.
Library and held-out images come from different days.

Cloud rule: sat_d < ts or |gray_d| > tg. For each target false-cloud rate (20/10/5 %
of held-out clear pixels), ts and tg are the q and (1 - q) quantiles of sat_d and |gray_d|
on held-out clear pixels, with q the largest value whose combined rate stays <= target.
RBR threshold (cloud if rbr_d > t, one extra cover feature): 90 % quantile of rbr_d.
"""

from __future__ import annotations

import json
import multiprocessing as mp

import numpy as np
import pandas as pd
import pvlib

from cloud_features import (CSL_ANG_STEP, CSL_N_ANG, CSL_N_SZ, CSL_N_Z, CSL_SZ_STEP, CSL_Z_STEP, SAT_LEVEL,
                            MonthContext, colour_maps, csl_bin_index, csl_lookup, gray_map)
from common import ALTITUDE, CHECKS_DIR, FEATURES_DIR, LATITUDE, LONGITUDE, refuse_existing, write_json
from sky_geometry import read_small_rgb

OUT_LIB = FEATURES_DIR / "clear_sky_library.npz"
OUT_THR = FEATURES_DIR / "cloud_thresholds.json"
N_LIBRARY, N_HELDOUT, N_OVERCAST = 3000, 600, 600
PIX_PER_IMAGE = 2000
STRICT_K_STD = 0.02
TARGET_FPR = (0.20, 0.10, 0.05)
VALUES = ("sat", "gray", "rbr")
_CTX = {}


def _pixels(args):
    path, year, month, zen, az, seed = args
    if (year, month) not in _CTX:
        _CTX[(year, month)] = MonthContext(year, month)
    ctx = _CTX[(year, month)]
    rgb = read_small_rgb(path)
    if rgb is None:
        return None
    sat, rbr = colour_maps(rgb)
    gray = gray_map(rgb)
    region = ctx.usable & (rgb.min(axis=2) < SAT_LEVEL)
    ang = ctx.g.angle_to(zen, az)
    ii = np.flatnonzero(region.ravel())
    if len(ii) == 0:
        return None
    ii = np.random.default_rng(seed).choice(ii, size=min(PIX_PER_IMAGE, len(ii)), replace=False)
    n = len(ii)
    return np.column_stack([ang.ravel()[ii], ctx.g.zenith.ravel()[ii], np.full(n, zen),
                            sat.ravel()[ii], gray.ravel()[ii], rbr.ravel()[ii], np.full(n, seed)])


def collect(frame, pool, t_img_of, site, seed0):
    ti = frame["img_le_t"].map(t_img_of)
    sp = site.get_solarposition(pd.DatetimeIndex(ti))
    rows = [(p, t.year, t.month, z, a, seed0 + i) for i, (p, t, z, a) in
            enumerate(zip(frame["img_le_t"], ti, sp["apparent_zenith"], sp["azimuth"]))]
    parts = [r for r in pool.map(_pixels, rows, chunksize=4) if r is not None]
    return pd.DataFrame(np.vstack(parts), columns=["ang", "pz", "sz", "sat", "gray", "rbr", "img"])


def build_library(px: pd.DataFrame):
    shape = (CSL_N_ANG, CSL_N_Z, CSL_N_SZ)
    a, z, s = csl_bin_index(px["ang"].to_numpy(), px["pz"].to_numpy(), px["sz"].to_numpy())
    key = (a * CSL_N_Z + z) * CSL_N_SZ + s
    count = np.zeros(shape, np.int64)
    n = pd.Series(key).value_counts()
    count.ravel()[n.index.to_numpy()] = n.to_numpy()
    lib = {}
    for v in VALUES:
        arr = np.full(shape, np.nan)
        med = px.groupby(key)[v].median()
        arr.ravel()[med.index.to_numpy()] = med.to_numpy()
        arr[count < 30] = np.nan
        # fill sparse bins: same (angle, zenith) over solar zeniths, then same angle, then global
        for axis in (2, (1, 2)):
            with np.errstate(all="ignore"):
                fill = np.nanmedian(arr, axis=axis, keepdims=True)
            arr = np.where(np.isnan(arr), np.broadcast_to(fill, shape), arr)
        arr = np.where(np.isnan(arr), np.nanmedian(arr), arr)
        lib[v] = arr.astype(np.float32)
    return lib, count


def residuals(px, lib):
    a, z, s = px["ang"].to_numpy(), px["pz"].to_numpy(), px["sz"].to_numpy()
    return {v: px[v].to_numpy() - csl_lookup(lib[v], a, z, s) for v in VALUES}


def thresholds_for(res_clear, target):
    best = None
    for q in np.linspace(0.002, target, 200):
        ts = float(np.quantile(res_clear["sat"], q))
        tg = float(np.quantile(np.abs(res_clear["gray"]), 1 - q))
        fpr = float(np.mean((res_clear["sat"] < ts) | (np.abs(res_clear["gray"]) > tg)))
        if fpr <= target:
            best = (ts, tg, fpr, float(q))
    return best


def main():
    refuse_existing(OUT_LIB)
    refuse_existing(OUT_THR)
    s = pd.read_parquet(FEATURES_DIR / "samples_setA.parquet",
                        columns=["datetime", "split", "sky_class", "sza_t", "img_le_t", "k_std_30min"])
    s = s[(s["split"] == "train") & (s["sza_t"] < 85) & s["img_le_t"].notna()]
    strict = s[(s["sky_class"] == "clear") & (s["k_std_30min"] < STRICT_K_STD)]
    days = strict["datetime"].dt.date
    uniq = np.array(sorted(days.unique()))
    held_days = set(np.random.default_rng(11).choice(uniq, size=len(uniq) // 5, replace=False))
    is_held = days.isin(held_days)
    lib_imgs = strict[~is_held].sample(N_LIBRARY, random_state=12)
    held_imgs = strict[is_held].sample(N_HELDOUT, random_state=13)
    ovc_imgs = s[s["sky_class"] == "overcast"].sample(N_OVERCAST, random_state=14)

    t_img_of = pd.read_parquet(FEATURES_DIR / "image_index.parquet", columns=["path", "t_img"]).set_index("path")["t_img"]
    site = pvlib.location.Location(LATITUDE, LONGITUDE, altitude=ALTITUDE)
    with mp.get_context("fork").Pool(48) as pool:
        px_lib = collect(lib_imgs, pool, t_img_of, site, 0)
        px_held = collect(held_imgs, pool, t_img_of, site, 100000)
        px_ovc = collect(ovc_imgs, pool, t_img_of, site, 200000)
    lib, count = build_library(px_lib)
    np.savez(OUT_LIB, **lib, count=count, ang_step=CSL_ANG_STEP, z_step=CSL_Z_STEP, sz_step=CSL_SZ_STEP)

    r_held, r_ovc = residuals(px_held, lib), residuals(px_ovc, lib)
    found = [thresholds_for(r_held, t) for t in TARGET_FPR]
    cloud_thr = [[ts, tg] for ts, tg, _, _ in found]
    rbr_thr = float(np.quantile(r_held["rbr"], 0.90))

    def per_image(px, res, ts, tg):
        hit = (res["sat"] < ts) | (np.abs(res["gray"]) > tg)
        return pd.Series(hit).groupby(px["img"].to_numpy()).mean()

    report = {}
    for name, px, res in (("heldout_clear", px_held, r_held), ("overcast", px_ovc, r_ovc)):
        rep = {}
        for k, (ts, tg) in enumerate(cloud_thr, start=1):
            rep[f"cf{k}_per_image_median"] = float(per_image(px, res, ts, tg).median())
        near = px["ang"].to_numpy() < 35
        ts2, tg2 = cloud_thr[1]
        rep["pixels_lt35deg_from_sun_detected_cf2"] = float(np.mean(((res["sat"] < ts2) | (np.abs(res["gray"]) > tg2))[near]))
        report[name] = rep
    out = {"cloud_thresholds": cloud_thr, "target_fpr": list(TARGET_FPR),
           "achieved_fpr": [f[2] for f in found], "quantile_q": [f[3] for f in found],
           "rbr_threshold": rbr_thr, "strict_k_std": STRICT_K_STD,
           "n_images": {"library": len(lib_imgs), "heldout_clear": len(held_imgs), "overcast": len(ovc_imgs)},
           "library_bins_with_30plus_px": int((count >= 30).sum()), "library_bins_total": int(count.size),
           "separation_report": report}
    write_json(OUT_THR, out)
    write_json(CHECKS_DIR / "cloud_thresholds.json", out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
