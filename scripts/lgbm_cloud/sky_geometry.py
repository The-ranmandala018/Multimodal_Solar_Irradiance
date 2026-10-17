"""Fisheye geometry for the cloud features.

Model (fitted by fit_sky_geometry.py, saved to features/sky_geometry.json):
    r          = distance (pixels, at READ_SCALE) from a single global centre (cx, cy)
    zenith_deg = c1 * r + c2 * r**2 + c3 * r**3          (global, fitted on TRAIN clear-sky sun detections)
    image_phi  = atan2(-(y - cy), x - cx)                 (degrees, mod 360; same convention as the source)
    azimuth    = az_sign * (image_phi - alpha[year, month]) (rotation per month, fitted from sun blobs
                                                           at the corrected image time; image content only)

The saved monthly calibrations (calibration_results/) are only used as a
starting point: their radial exponent p is poorly constrained in winter
(errors up to ~15 deg, see checks/sun_geometry_check.json).

Images are read at reduced size (cv2.IMREAD_REDUCED_COLOR_4: 384 x 384).
"""

from __future__ import annotations

import json
from functools import lru_cache

import cv2
import numpy as np

from common import FEATURES_DIR, PROJECT_DIR

READ_FLAG = cv2.IMREAD_REDUCED_COLOR_4
READ_SCALE = 0.25
SIZE = 384
CAL_DIR = PROJECT_DIR / "calibration_results"
GEOMETRY_FILE = FEATURES_DIR / "sky_geometry.json"
MAX_ZENITH = 75.0  # horizon cut-off for all features (80 deg still includes a shed and pole at the bottom)


@lru_cache(maxsize=None)
def saved_calibrations() -> dict:
    out = {}
    for year in (2014, 2015, 2016):
        months = json.loads((CAL_DIR / f"Folsom_{year}_monthly_calibrations.json").read_text())["months"]
        for m, c in months.items():
            out[(year, int(m))] = {k: float(c[k]) for k in ("cx", "cy", "R", "p", "az_sign", "az_alpha_deg")}
    return out


@lru_cache(maxsize=None)
def fitted_geometry() -> dict:
    g = json.loads(GEOMETRY_FILE.read_text())
    g["alpha"] = {tuple(int(x) for x in k.split("-")): v for k, v in g["alpha_by_month"].items()}
    return g


def read_small_rgb(path: str):
    bgr = cv2.imread(path, READ_FLAG)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def unit_vector(zenith_deg, azimuth_deg):
    z, a = np.radians(zenith_deg), np.radians(azimuth_deg)
    return np.stack([np.sin(z) * np.sin(a), np.sin(z) * np.cos(a), np.cos(z)], axis=-1)


class SkyGeometry:
    """Per-pixel zenith/azimuth/unit vectors at SIZE x SIZE.

    radial: ("poly", (c1, c2, c3)) or ("power", (R_px, p)); centre (cx, cy) in READ_SCALE pixels.
    """

    def __init__(self, cx, cy, radial, az_sign, alpha, max_zenith=MAX_ZENITH):
        self.cx, self.cy, self.radial = float(cx), float(cy), radial
        self.az_sign, self.alpha = int(az_sign), float(alpha)
        yy, xx = np.indices((SIZE, SIZE), dtype=np.float32)
        r = np.hypot(xx - self.cx, yy - self.cy)
        self.r = r
        self.zenith = self.zenith_of_r(r)
        phi = np.degrees(np.arctan2(-(yy - self.cy), xx - self.cx)) % 360.0
        self.phi = phi
        self.azimuth = (self.az_sign * (phi - self.alpha)) % 360.0
        self.inside = self.zenith <= 90.0
        self.sky = self.zenith <= max_zenith
        self.vec = unit_vector(self.zenith, self.azimuth).astype(np.float32)
        self._r_grid = np.linspace(0, SIZE, 4000)
        self._z_grid = self.zenith_of_r(self._r_grid)

    def zenith_of_r(self, r):
        kind, c = self.radial
        if kind == "poly":
            z = c[0] * r + c[1] * r ** 2 + c[2] * r ** 3
            # beyond the turning point the cubic is not monotone: treat as outside the sky
            return np.where(r <= self._r_max_monotone(c), z, 999.0)
        R, p = c
        return 90.0 * np.clip(r / R, 0, None) ** p

    @staticmethod
    def _r_max_monotone(c):
        roots = np.roots([3 * c[2], 2 * c[1], c[0]])
        roots = roots[np.isreal(roots)].real
        roots = roots[roots > 0]
        return float(roots.min()) if len(roots) else np.inf

    @classmethod
    def from_fitted(cls, year, month, max_zenith=MAX_ZENITH):
        g = fitted_geometry()
        return cls(g["cx"], g["cy"], ("poly", tuple(g["poly"])), g["az_sign"], g["alpha"][(year, month)], max_zenith)

    @classmethod
    def from_saved(cls, year, month, p_override=None, max_zenith=MAX_ZENITH):
        c = saved_calibrations()[(year, month)]
        p = c["p"] if p_override is None else p_override
        return cls(c["cx"] * READ_SCALE, c["cy"] * READ_SCALE, ("power", (c["R"] * READ_SCALE, p)),
                   c["az_sign"], c["az_alpha_deg"], max_zenith)

    def angle_to(self, zenith_deg: float, azimuth_deg: float) -> np.ndarray:
        s = unit_vector(zenith_deg, azimuth_deg).astype(np.float32)
        return np.degrees(np.arccos(np.clip(self.vec @ s, -1.0, 1.0)))

    def pixel_of(self, zenith_deg: float, azimuth_deg: float):
        ok = self._z_grid < 900
        r = float(np.interp(zenith_deg, self._z_grid[ok], self._r_grid[ok]))
        phi = np.radians((self.az_sign * azimuth_deg + self.alpha) % 360.0)
        return self.cx + r * np.cos(phi), self.cy - r * np.sin(phi)


def detect_sun_blob(rgb: np.ndarray, near: np.ndarray, sat_level: int = 250, min_area: int = 4):
    """Largest saturated blob inside `near` (bool mask). Returns (area_px, x, y) or None."""
    sat = (rgb.min(axis=2) >= sat_level) & near
    if sat.sum() < min_area:
        return None
    n, labels, stats, cents = cv2.connectedComponentsWithStats(sat.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[i, cv2.CC_STAT_AREA])
    if area < min_area:
        return None
    return area, float(cents[i, 0]), float(cents[i, 1])
