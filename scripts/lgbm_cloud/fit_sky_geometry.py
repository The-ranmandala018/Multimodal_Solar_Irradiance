"""Fit the fisheye geometry used by the cloud features -> features/sky_geometry.json

1. Centre: median of the saved monthly calibration centres (scaled to 384 px).
2. Radial curve zenith = c1 r + c2 r^2 + c3 r^3: least squares on the visible sun
   (largest saturated blob) in TRAIN clear-sky samples (sky_class == clear,
   SZA < 75), sun position from pvlib at the corrected image time t_img.
3. Rotation alpha per (year, month), all months: circular median of
   (image_phi(blob) - az_sign * sun_azimuth) over up to 300 daytime images per
   month (SZA 20-75 at t_img), selected from the image index only, with the
   blob searched within 20 deg of the predicted sun. No irradiance is used, so
   this is the same kind of image-only calibration as calibration_results/,
   but on the corrected clock.

Usage:
    PYTHONPATH=. python scripts/lgbm_cloud/fit_sky_geometry.py
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pandas as pd
import pvlib

from common import (ALTITUDE, CHECKS_DIR, FEATURES_DIR, LATITUDE, LONGITUDE, git_head,
                    refuse_existing, write_json)
from sky_geometry import (GEOMETRY_FILE, READ_SCALE, SkyGeometry, detect_sun_blob, read_small_rgb,
                          saved_calibrations)

RADIAL_PER_MONTH = 40
ALPHA_PER_MONTH = 300
ALPHA_SEARCH_DEG = 20.0
P_START = 0.75  # better than the saved p for a first guess (checks/sun_geometry_check.json)
HORIZON_WEIGHT = 200  # pseudo-points pinning zenith 90 at R
MAX_BLOB_AREA = 5000  # saturated sun + glare is ~1900 px (p99 ~3600) at 384 px

site = pvlib.location.Location(LATITUDE, LONGITUDE, altitude=ALTITUDE)
_G = {}


def sun_pos(times):
    sp = site.get_solarposition(pd.DatetimeIndex(times))
    return sp["apparent_zenith"].to_numpy(), sp["azimuth"].to_numpy()


def circ_median_deg(x):
    x = np.asarray(x) % 360.0
    ref = np.degrees(np.arctan2(np.sin(np.radians(x)).mean(), np.cos(np.radians(x)).mean()))
    d = ((x - ref + 180) % 360) - 180
    return float((ref + np.median(d)) % 360), float(np.median(np.abs(d - np.median(d))))


def _detect(args):
    path, cx, cy, radial, az_sign, alpha, zen, az, search = args
    key = (cx, cy, radial, az_sign, alpha)
    if key not in _G:
        _G.clear()
        _G[key] = SkyGeometry(cx, cy, radial, az_sign, alpha, max_zenith=90.0)
    g = _G[key]
    rgb = read_small_rgb(path)
    if rgb is None or rgb.shape[:2] != (384, 384):
        return None
    blob = detect_sun_blob(rgb, (g.angle_to(zen, az) <= search) & g.inside)
    if blob is None or blob[0] > MAX_BLOB_AREA:
        return None
    _, bx, by = blob
    return bx, by


def detect_many(rows, pool):
    return pool.map(_detect, rows, chunksize=4)


def main():
    refuse_existing(GEOMETRY_FILE)
    cal = saved_calibrations()
    cx = float(np.median([c["cx"] for c in cal.values()]) * READ_SCALE)
    cy = float(np.median([c["cy"] for c in cal.values()]) * READ_SCALE)
    R = float(np.median([c["R"] for c in cal.values()]) * READ_SCALE)
    index = pd.read_parquet(FEATURES_DIR / "image_index.parquet",
                            columns=["path", "t_img", "sza_t_img", "corrupted"])
    t_img_of = index.set_index("path")["t_img"]
    pool = mp.get_context("fork").Pool(48)

    # ---- 2. radial curve from TRAIN clear-sky samples ----------------------
    s = pd.read_parquet(FEATURES_DIR / "samples_setA.parquet",
                        columns=["datetime", "split", "sky_class", "sza_t", "img_le_t"])
    s = s[(s["split"] == "train") & (s["sky_class"] == "clear") & (s["sza_t"] < 75) & s["img_le_t"].notna()]
    s["mk"] = s["datetime"].dt.strftime("%Y-%m")
    s = pd.concat([g.sample(min(RADIAL_PER_MONTH, len(g)), random_state=1) for _, g in s.groupby("mk")])
    s["t_img"] = s["img_le_t"].map(t_img_of)
    s["zen"], s["az"] = sun_pos(s["t_img"])
    rows = [(r.img_le_t, cx, cy, ("power", (R, P_START)), int(cal[(r.datetime.year, r.datetime.month)]["az_sign"]),
             cal[(r.datetime.year, r.datetime.month)]["az_alpha_deg"], r.zen, r.az, 40.0) for r in s.itertuples()]
    det = detect_many(rows, pool)
    ok = np.array([d is not None for d in det])
    bx = np.array([d[0] for d in det if d is not None])
    by = np.array([d[1] for d in det if d is not None])
    r = np.hypot(bx - cx, by - cy)
    zen = s["zen"].to_numpy()[ok]
    X = np.column_stack([r, r ** 2, r ** 3])
    # Training suns never go below SZA 75, so pin the curve at the horizon:
    # zenith 90 at the image-circle edge R (median saved R), as heavily weighted pseudo-points.
    Xc = np.tile([[R, R ** 2, R ** 3]], (HORIZON_WEIGHT, 1))
    yc = np.full(HORIZON_WEIGHT, 90.0)

    def fit(mask):
        c, *_ = np.linalg.lstsq(np.vstack([X[mask], Xc]), np.concatenate([zen[mask], yc]), rcond=None)
        return c

    coef = fit(np.ones(len(zen), bool))
    res = zen - X @ coef
    keep = np.abs(res) < 4 * np.median(np.abs(res)) + 1.0  # one robust refit
    coef = fit(keep)
    res = zen - X @ coef
    zen_at_R = float(np.array([R, R ** 2, R ** 3]) @ coef)
    radial = ("poly", tuple(float(c) for c in coef))
    radial_by_month = pd.Series(res, index=s["mk"].to_numpy()[ok]).groupby(level=0).median().round(2).to_dict()

    # ---- 3. rotation per month, all months, image-only selection -----------
    idx = index[~index["corrupted"] & index["sza_t_img"].between(20, 75)].copy()
    idx["mk"] = idx["t_img"].dt.strftime("%Y-%m")
    alpha_by_month, alpha_info = {}, {}
    for mk, g in idx.groupby("mk"):
        y, m = int(mk[:4]), int(mk[5:])
        g = g.sample(min(ALPHA_PER_MONTH, len(g)), random_state=2)
        zz, aa = sun_pos(g["t_img"])
        alpha = cal[(y, m)]["az_alpha_deg"]
        az_sign = int(cal[(y, m)]["az_sign"])
        for _ in range(2):  # refine once around the first estimate
            rows = [(p, cx, cy, radial, az_sign, alpha, z, a, ALPHA_SEARCH_DEG) for p, z, a in zip(g["path"], zz, aa)]
            det = detect_many(rows, pool)
            phis, azs = [], []
            for d, a in zip(det, aa):
                if d is None:
                    continue
                phis.append(np.degrees(np.arctan2(-(d[1] - cy), d[0] - cx)) % 360.0)
                azs.append(a)
            if len(phis) < 20:
                break
            alpha, mad = circ_median_deg(np.array(phis) - az_sign * np.array(azs))
        alpha_by_month[f"{y}-{m}"] = alpha
        alpha_info[mk] = {"n_blobs": len(phis), "alpha": round(alpha, 2), "mad_deg": round(mad, 2) if len(phis) >= 20 else None,
                          "saved_alpha": cal[(y, m)]["az_alpha_deg"]}
    pool.close()

    az_signs = {int(c["az_sign"]) for c in cal.values()}
    if len(az_signs) != 1:
        raise RuntimeError(f"saved az_sign differs between months: {az_signs}")
    geometry = {
        "cx": cx, "cy": cy, "R": R, "poly": list(radial[1]), "az_sign": az_signs.pop(), "alpha_by_month": alpha_by_month,
        "read_scale": READ_SCALE, "size": 384, "git_head": git_head(),
        "radial_fit": {"n_train_detections": int(ok.sum()), "n_used": int(keep.sum()), "R_edge_px": R, "zenith_at_R": zen_at_R,
                       "abs_resid_median_deg": float(np.median(np.abs(res[keep]))),
                       "abs_resid_p90_deg": float(np.quantile(np.abs(res[keep]), 0.9)),
                       "resid_median_by_month_deg": radial_by_month},
        "alpha_fit": alpha_info,
    }
    GEOMETRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    write_json(GEOMETRY_FILE, geometry)
    write_json(CHECKS_DIR / "sky_geometry_fit.json", geometry)
    import json
    print(json.dumps({k: geometry[k] for k in ("cx", "cy", "poly", "radial_fit")}, indent=1))
    print(pd.DataFrame(alpha_info).T.to_string())


if __name__ == "__main__":
    main()
