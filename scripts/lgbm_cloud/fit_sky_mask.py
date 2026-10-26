"""Static obstruction mask (buildings, trees, poles) per year -> features/sky_mask_<year>.npy

Image content only (no irradiance): for each year, up to N_PER_YEAR daytime images
(SZA < 70 at t_img) spread over all days, per-pixel median brightness. A pixel is
an obstruction if it is below zenith MIN_ZENITH_DEG and its median brightness is
below GRAY_MAX (sky is bright in daytime, buildings and trees are not). The mask
is dilated by DILATE_PX. The usable sky for features is then:
    zenith <= MAX_ZENITH (80 deg) and not obstruction.

Writes a preview PNG of the median image and mask to checks/.
"""

from __future__ import annotations

import multiprocessing as mp

import cv2
import numpy as np
import pandas as pd

from common import CHECKS_DIR, FEATURES_DIR, refuse_existing, write_json
from sky_geometry import MAX_ZENITH, SIZE, SkyGeometry, read_small_rgb

N_PER_YEAR = 1500
GRAY_MAX = 80
MIN_ZENITH_DEG = 40.0
DILATE_PX = 3


def _gray(path):
    rgb = read_small_rgb(path)
    if rgb is None or rgb.shape[:2] != (SIZE, SIZE):
        return None
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def main():
    idx = pd.read_parquet(FEATURES_DIR / "image_index.parquet", columns=["path", "t_img", "sza_t_img", "corrupted"])
    idx = idx[~idx["corrupted"] & (idx["sza_t_img"] < 70)]
    g_geom = SkyGeometry.from_fitted(2015, 6)  # same centre and radial curve for every month
    summary, previews = {}, []
    with mp.get_context("fork").Pool(48) as pool:
        for year in (2014, 2015, 2016):
            out = refuse_existing(FEATURES_DIR / f"sky_mask_{year}.npy")
            sel = idx[idx["t_img"].dt.year == year].sample(N_PER_YEAR, random_state=year)
            grays = [g for g in pool.map(_gray, sel["path"], chunksize=8) if g is not None]
            med = np.median(np.stack(grays), axis=0)
            obstruction = (med < GRAY_MAX) & (g_geom.zenith > MIN_ZENITH_DEG) & g_geom.inside
            obstruction = cv2.dilate(obstruction.astype(np.uint8), np.ones((2 * DILATE_PX + 1,) * 2, np.uint8)) > 0
            np.save(out, obstruction)
            usable = g_geom.sky & ~obstruction
            summary[year] = {"n_images": len(grays), "obstruction_px_in_sky80": int((obstruction & g_geom.sky).sum()),
                             "usable_px": int(usable.sum()), "sky80_px": int(g_geom.sky.sum()),
                             "usable_frac_of_sky80": float(usable.sum() / g_geom.sky.sum())}
            prev = cv2.cvtColor(med.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            prev[obstruction] = (0.5 * prev[obstruction] + [0, 0, 127]).astype(np.uint8)
            c, _ = cv2.findContours(g_geom.sky.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(prev, c, -1, (0, 255, 255), 1)
            cv2.putText(prev, str(year), (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            previews.append(prev)
    cv2.imwrite(str(CHECKS_DIR / "sky_mask_preview.png"), np.hstack(previews))
    write_json(CHECKS_DIR / "sky_mask_summary.json", {"params": {"N_PER_YEAR": N_PER_YEAR, "GRAY_MAX": GRAY_MAX,
               "MIN_ZENITH_DEG": MIN_ZENITH_DEG, "DILATE_PX": DILATE_PX, "MAX_ZENITH": MAX_ZENITH}, "years": summary})
    print(summary)


if __name__ == "__main__":
    main()
