"""Gate 2: debug overlays for 50 random TRAIN samples -> checks/gate2/page_*.png + gate2_samples.csv

Each tile: original image | overlay with
    red tint     cloud mask at the middle threshold (cf2)
    yellow       saturated pixels (sun/glare)
    cyan line    usable-sky boundary (zenith 75 + obstruction mask)
    green circle predicted sun; magenta lines: 15/30/45/70 deg rings around it
    white arrow  cloud motion (when valid), drawn from the sun, length = 10 min of motion
Text: t_img, sky class (k-based), k(t), cf1/cf2/cf3, sun detected, motion.
Samples: 15 clear + 15 mixed + 15 overcast + 5 from Sep-Oct 2015, train split only.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import pvlib

from cloud_features import MonthContext, compute
from common import ALTITUDE, CHECKS_DIR, FEATURES_DIR, LATITUDE, LONGITUDE, refuse_existing
from sky_geometry import read_small_rgb

OUT_DIR = CHECKS_DIR / "gate2"
SITE = pvlib.location.Location(LATITUDE, LONGITUDE, altitude=ALTITUDE)


def pick_samples():
    s = pd.read_parquet(FEATURES_DIR / "samples_setA.parquet",
                        columns=["datetime", "split", "sky_class", "k_lag0", "img_le_t", "sza_t"])
    s = s[(s["split"] == "train") & s["img_le_t"].notna()]
    parts = [s[s["sky_class"] == c].sample(15, random_state=31 + i) for i, c in enumerate(("clear", "mixed", "overcast"))]
    late = s[s["datetime"] >= "2015-09-01"]
    parts.append(late.sample(5, random_state=40))
    return pd.concat(parts).reset_index(drop=True)


def frames_before(index, path, n=2):
    i = index.index.get_loc(path)
    day = index.iloc[i]["day_dir"]
    prev = index.iloc[max(0, i - n):i]
    return prev[prev["day_dir"] == day]


def contour(img, mask, colour):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cs, -1, colour, 1)


def tile(row, index, ctx_cache):
    path = row["img_le_t"]
    seq = pd.concat([frames_before(index, path), index.loc[[path]]])
    sp = SITE.get_solarposition(pd.DatetimeIndex(seq["t_img"]))
    prev, out = None, None
    for (p, r), zen, az in zip(seq.iterrows(), sp["apparent_zenith"], sp["azimuth"]):
        key = (r["t_img"].year, r["t_img"].month)
        if key not in ctx_cache:
            ctx_cache[key] = MonthContext(*key)
        rgb = read_small_rgb(p)
        out = compute(rgb, ctx_cache[key], float(zen), float(az), r["t_img"], prev, return_masks=True)
        prev = out[1]
    feats, _, masks = out
    ctx = ctx_cache[key]
    orig = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ov = orig.copy()
    cm = masks["clouds"][1]
    ov[cm] = (0.45 * ov[cm] + np.array([0, 0, 140])).astype(np.uint8)
    ov[masks["sat"] & ctx.g.inside] = (0, 230, 255)
    contour(ov, ctx.usable, (255, 255, 0))
    x, y = ctx.g.pixel_of(float(zen), float(az))
    cv2.circle(ov, (int(round(x)), int(round(y))), 5, (0, 255, 0), 2)
    for rr in (15, 30, 45, 70):
        contour(ov, (masks["ang"] <= rr) & ctx.g.inside, (255, 0, 255))
    mo = masks["motion"]
    if feats["mo_valid"] == 1 and mo is not None:
        # plane (u east, v north) velocity -> image direction: project the sun point moved by 10 min
        su, sv = np.tan(np.radians(zen)) * np.sin(np.radians(az)), np.tan(np.radians(zen)) * np.cos(np.radians(az))
        u2, v2 = su + mo["vx"] * 10, sv + mo["vy"] * 10
        z2 = np.degrees(np.arctan(np.hypot(u2, v2)))
        a2 = np.degrees(np.arctan2(u2, v2)) % 360
        if z2 < 85:
            x2, y2 = ctx.g.pixel_of(z2, a2)
            cv2.arrowedLine(ov, (int(x), int(y)), (int(x2), int(y2)), (255, 255, 255), 2, tipLength=0.2)
    txt = [f"{str(row['datetime'])[:16]}  img {str(seq['t_img'].iloc[-1])[11:19]}",
           f"{row['sky_class']}  k(t)={row['k_lag0']:.2f}  SZA={row['sza_t']:.0f}",
           f"cf1/2/3={feats['cf1_all']:.2f}/{feats['cf2_all']:.2f}/{feats['cf3_all']:.2f}",
           f"sun_det={int(feats['sun_detected'])} blob={int(feats['sun_blob_area'])}",
           (f"motion {feats['mo_speed']:.3f}/min resp={feats['mo_response']:.2f}" if feats["mo_valid"] == 1
            else "motion: none")]
    for i, t in enumerate(txt):
        cv2.putText(ov, t, (4, 14 + 14 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
        cv2.putText(ov, t, (4, 14 + 14 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return np.hstack([orig, ov]), feats


def main():
    refuse_existing(OUT_DIR)
    OUT_DIR.mkdir(parents=True)
    index = pd.read_parquet(FEATURES_DIR / "image_index.parquet",
                            columns=["path", "day_dir", "t_img", "corrupted"])
    index = index[~index["corrupted"]].sort_values("t_img").set_index("path")
    samples = pick_samples()
    ctx_cache, tiles, rows = {}, [], []
    for _, r in samples.iterrows():
        t, feats = tile(r, index, ctx_cache)
        tiles.append(t)
        rows.append({"datetime": r["datetime"], "sky_class": r["sky_class"], "img": r["img_le_t"], **feats})
    for p in range(0, len(tiles), 10):
        page = tiles[p:p + 10]
        while len(page) < 10:
            page.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(page[i:i + 2]) for i in range(0, 10, 2)])
        cv2.imwrite(str(OUT_DIR / f"page_{p // 10 + 1}.png"), grid)
    pd.DataFrame(rows).to_csv(OUT_DIR / "gate2_samples.csv", index=False)
    print(f"wrote {len(tiles)} tiles to {OUT_DIR}")


if __name__ == "__main__":
    main()
