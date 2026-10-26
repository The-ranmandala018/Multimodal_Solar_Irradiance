"""Cloud-score channel for the CNN input (optional, off by default).

Idea: the frozen cloud-feature pipeline (scripts/lgbm_cloud/cloud_features.py,
FROZEN_V1) decides, for every pixel, how different the sky is from the expected
clear sky at that sun-relative position. This module turns that per-pixel
difference into one image channel and puts it where the redundant "intensity"
channel was, so the tensor stays (10, 180, 720) and the CNN is unchanged.

Per pixel (same formulas as the cloud masks and the motion score in
cloud_features.compute, middle threshold k = 2):
    sat_d  = colour saturation - clear-sky library saturation
    gray_d = brightness        - clear-sky library brightness
    score  = max(sat_d / ts2, |gray_d| / tg2)       (> 1: cloud at threshold 2)
    value  = clip(score, 0, 2) / 2                   -> [0, 1]
Pixels that are not measured get 0: outside the usable sky (zenith > 75 deg or
obstruction mask) and saturated pixels (sun and glare, min(R,G,B) >= 250).
0 therefore means "clear or not measured".

The score map is computed on the 384 x 384 image (IMREAD_REDUCED_COLOR_4), in
the original fisheye pixel grid, and remapped to (180, 720) with the SAME
calibration maps as the RGB image (scaled by 1/4). It is therefore pixel-aligned
with the RGB channels whatever the calibration. The calibration only affects
the clear-sky lookup, which uses the fitted cubic sky geometry of the cloud
pipeline.

Sun position: pvlib apparent zenith/azimuth at the sample time given by the
caller (the dataset passes the sample time t; the extractor used the corrected
image time, which differs from t by at most the 120 s matching tolerance, about
0.5 deg of sun motion).

Needs scripts/lgbm_cloud and its fitted files in
experiments/lightgbm_cloud_features/features/ (clear_sky_library.npz,
cloud_thresholds.json, sky_geometry.json, sky_mask_<year>.npy).
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

LGBM_CLOUD_DIR = Path(__file__).resolve().parents[1] / "scripts" / "lgbm_cloud"
SCORE_CLIP = 2.0  # score above 2 (twice the threshold) is saturated to 1.0
MAP_SCALE = 0.25  # IMREAD_REDUCED_COLOR_4


@lru_cache(maxsize=1)
def _modules():
    """Import the frozen cloud pipeline lazily (it uses flat imports)."""
    if str(LGBM_CLOUD_DIR) not in sys.path:
        sys.path.insert(0, str(LGBM_CLOUD_DIR))
    import cloud_features as cf
    import pvlib
    import sky_geometry as sg

    site = pvlib.location.Location(38.642, -121.148, altitude=60.0)
    return cf, sg, site


@lru_cache(maxsize=36)
def _month_context(year: int, month: int):
    cf, _, _ = _modules()
    return cf.MonthContext(year, month)


def sun_position(timestamp):
    """pvlib apparent zenith and azimuth (deg) for a naive UTC timestamp."""
    import pandas as pd

    _, _, site = _modules()
    sp = site.get_solarposition(pd.DatetimeIndex([pd.Timestamp(timestamp)]))
    return float(sp["apparent_zenith"].iloc[0]), float(sp["azimuth"].iloc[0])


def cloud_score_small(rgb: np.ndarray, year: int, month: int, sun_zen: float, sun_az: float) -> np.ndarray:
    """Per-pixel cloud score in [0, 1] on the 384 x 384 image."""
    cf, _, _ = _modules()
    ctx = _month_context(int(year), int(month))
    g = ctx.g
    lib = cf.library()

    sat_px = rgb.min(axis=2) >= cf.SAT_LEVEL
    colour_sat, _ = cf.colour_maps(rgb)
    gray_f = cf.gray_map(rgb)
    usable = ctx.usable
    clf_region = usable & ~sat_px

    ang = g.angle_to(sun_zen, sun_az)
    sat_d = np.zeros_like(colour_sat)
    gray_d = np.zeros_like(colour_sat)
    sat_d[usable] = colour_sat[usable] - cf.csl_lookup(lib["sat"], ang[usable], g.zenith[usable], sun_zen)
    gray_d[usable] = gray_f[usable] - cf.csl_lookup(lib["gray"], ang[usable], g.zenith[usable], sun_zen)

    ts2, tg2 = lib["cloud_thr"][1]
    score = np.maximum(sat_d / ts2, np.abs(gray_d) / tg2)
    out = np.clip(score, 0.0, SCORE_CLIP) / SCORE_CLIP
    out[~clf_region] = 0.0
    return out.astype(np.float32)


def scale_maps(map_x: np.ndarray, map_y: np.ndarray):
    """Convert full-resolution remap coordinates to the 384 x 384 grid.

    Reduced pixel i covers original pixels 4i .. 4i+3, so original coordinate X
    corresponds to (X + 0.5) * 0.25 - 0.5 in the reduced image.
    """
    sx = ((map_x + 0.5) * MAP_SCALE - 0.5).astype(np.float32)
    sy = ((map_y + 0.5) * MAP_SCALE - 0.5).astype(np.float32)
    return sx, sy


def cloud_channel_map(
    image_path: str,
    year: int,
    month: int,
    timestamp,
    scaled_map_x: np.ndarray,
    scaled_map_y: np.ndarray,
) -> np.ndarray:
    """Cloud score remapped to the (180, 720) azimuth-zenith grid, float32 in [0, 1]."""
    import cv2

    _, sg, _ = _modules()
    rgb = sg.read_small_rgb(image_path)
    if rgb is None or rgb.shape[:2] != (sg.SIZE, sg.SIZE):
        raise ValueError(f"Could not read a {sg.SIZE}x{sg.SIZE} reduced image: {image_path}")
    zen, az = sun_position(timestamp)
    small = cloud_score_small(rgb, year, month, zen, az)
    return cv2.remap(
        small,
        scaled_map_x,
        scaled_map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    ).astype(np.float32)
