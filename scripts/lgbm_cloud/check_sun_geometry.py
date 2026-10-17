"""Gate-2 helper: how well do the saved monthly calibrations place the sun?

Uses TRAIN clear-sky samples only (sky_class == clear, SZA < 75, image at or before t).
For each image, the visible sun is the largest saturated blob within 40 deg of
the predicted position; we compare its pixel position with the prediction under
(a) the saved monthly p and (b) fixed p values, and report the angular error
(measured with the same geometry) and the pixel error.

Output: experiments/lightgbm_cloud_features/checks/sun_geometry_check.csv and .json
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pandas as pd
import pvlib

from common import ALTITUDE, CHECKS_DIR, FEATURES_DIR, LATITUDE, LONGITUDE, refuse_existing, write_json
from sky_geometry import SkyGeometry, detect_sun_blob, read_small_rgb

P_GRID = [None, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
PER_MONTH = 30
OUT_CSV = CHECKS_DIR / "sun_geometry_check.csv"
OUT_JSON = CHECKS_DIR / "sun_geometry_check.json"

_geoms = {}


def geom(year, month, p):
    key = (year, month, p)
    if key not in _geoms:
        _geoms[key] = SkyGeometry.from_saved(year, month, p_override=p)
    return _geoms[key]


def one(row):
    rgb = read_small_rgb(row["path"])
    if rgb is None or rgb.shape[:2] != (384, 384):
        return []
    out = []
    g0 = geom(row["year"], row["month"], None)
    near = g0.angle_to(row["sun_zen"], row["sun_az"]) <= 40.0
    blob = detect_sun_blob(rgb, near & g0.inside)
    if blob is None:
        return [{"path": row["path"], "month_key": row["month_key"], "p": "none_detected"}]
    area, bx, by = blob
    for p in P_GRID:
        g = geom(row["year"], row["month"], p)
        px, py = g.pixel_of(row["sun_zen"], row["sun_az"])
        ix, iy = int(round(bx)), int(round(by))
        ang = float(g.angle_to(row["sun_zen"], row["sun_az"])[iy, ix])
        out.append({"path": row["path"], "month_key": row["month_key"], "p": "saved" if p is None else p,
                    "p_value": g.radial[1][1], "blob_area": area, "pix_err": float(np.hypot(px - bx, py - by)),
                    "ang_err_deg": ang, "sun_zen": row["sun_zen"],
                    "blob_zen": float(g.zenith[iy, ix]), "blob_az": float(g.azimuth[iy, ix]),
                    "sun_az": row["sun_az"]})
    return out


def main():
    refuse_existing(OUT_CSV)
    CHECKS_DIR.mkdir(parents=True, exist_ok=True)
    s = pd.read_parquet(FEATURES_DIR / "samples_setA.parquet",
                        columns=["datetime", "split", "sky_class", "sza_t", "img_le_t"])
    s = s[(s["split"] == "train") & (s["sky_class"] == "clear") & (s["sza_t"] < 75) & s["img_le_t"].notna()]
    s["month_key"] = s["datetime"].dt.strftime("%Y-%m")
    s = s.groupby("month_key", group_keys=False).apply(lambda g: g.sample(min(PER_MONTH, len(g)), random_state=0))
    idx = pd.read_parquet(FEATURES_DIR / "image_index.parquet", columns=["path", "t_img"]).set_index("path")
    s["t_img"] = s["img_le_t"].map(idx["t_img"])
    site = pvlib.location.Location(LATITUDE, LONGITUDE, altitude=ALTITUDE)
    sp = site.get_solarposition(pd.DatetimeIndex(s["t_img"]))
    s["sun_zen"], s["sun_az"] = sp["apparent_zenith"].to_numpy(), sp["azimuth"].to_numpy()
    s["year"], s["month"] = s["datetime"].dt.year, s["datetime"].dt.month
    rows = s.rename(columns={"img_le_t": "path"})[["path", "month_key", "year", "month", "sun_zen", "sun_az"]].to_dict("records")

    with mp.get_context("fork").Pool(32) as pool:
        res = [r for chunk in pool.map(one, rows, chunksize=8) for r in chunk]
    df = pd.DataFrame(res)
    df.to_csv(OUT_CSV, index=False)

    det = df[df["p"] != "none_detected"]
    n_img = df["path"].nunique()
    summary = {"n_images": int(n_img), "n_sun_detected": int(det["path"].nunique()),
               "by_p_median_ang_err_deg": det.groupby("p")["ang_err_deg"].median().round(2).to_dict(),
               "by_p_median_pix_err": det.groupby("p")["pix_err"].median().round(1).to_dict(),
               "saved_by_month_median_ang_err_deg":
                   det[det["p"] == "saved"].groupby("month_key")["ang_err_deg"].median().round(1).to_dict()}
    # Azimuth-only and zenith-only error under saved calibration, to separate rotation from radial errors.
    sv = det[det["p"] == "saved"].copy()
    sv["az_err"] = ((sv["blob_az"] - sv["sun_az"] + 180) % 360) - 180
    sv["zen_err"] = sv["blob_zen"] - sv["sun_zen"]
    summary["saved_az_err_median_by_month"] = sv.groupby("month_key")["az_err"].median().round(1).to_dict()
    summary["saved_zen_err_median_by_month"] = sv.groupby("month_key")["zen_err"].median().round(1).to_dict()
    write_json(OUT_JSON, summary)
    import json
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
