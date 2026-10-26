"""Per-image cloud features from one Folsom sky image (and the previous frame for motion).

All features use only the image at t_img (and the frame before it), never later data.
Sky region: zenith <= MAX_ZENITH (75 deg) and outside the yearly obstruction mask.
Pixels with min(R, G, B) >= SAT_LEVEL are "saturated" (sun and glare); they are
excluded from cloud classification.

Cloud detection: Folsom JPEGs are washed out, so a pixel is compared with the clear-sky
value expected at its geometry (clear-sky library from fit_clear_sky_library.py, train
images only, linearly interpolated in angle-to-sun, pixel zenith and solar zenith):
    sat_d  = colour saturation - library_sat      (clouds are grey: sat_d < 0)
    gray_d = brightness - library_gray            (clouds are brighter far from the sun,
                                                   darker than clear circumsolar sky)
    cloud_k: sat_d < ts_k  or  |gray_d| > tg_k
with (ts_k, tg_k) set so the combined false-cloud rate on held-out clear train images is
20 / 10 / 5 % for k = 1 / 2 / 3.

Groups (prefix):
    cf_   cloud fraction at the 3 thresholds, whole sky and 3 zenith rings; plus an RBR-residual cover
    sun_  saturation and the sun blob near the expected sun position
    pos_  cloud fraction in rings around the sun, and the angle to the nearest cloud
    tx_   colour/brightness statistics, entropy, edges (image measures, not optical depth)
    mo_   motion between the previous frame and this one, on a sky plane (NaN when undefined)
"""

from __future__ import annotations

import json
from functools import lru_cache

import cv2
import numpy as np
from scipy.ndimage import map_coordinates

from common import FEATURES_DIR
from sky_geometry import MAX_ZENITH, SkyGeometry, detect_sun_blob

SAT_LEVEL = 250
SUN_SEARCH_DEG = 25.0
SUN_DETECTED_MIN_AREA = 200        # px at 384 x 384; clear-sky sun + glare is ~1900 px
ZEN_RINGS = ((0, 30), (30, 60), (60, MAX_ZENITH))
SUN_RINGS = ((0, 15), (15, 30), (30, 45), (45, 70))
MIN_RING_PX = 50
# clear-sky library bins
CSL_ANG_STEP, CSL_N_ANG = 10.0, 18
CSL_Z_STEP, CSL_N_Z = 7.5, 10
CSL_SZ_STEP, CSL_N_SZ = 10.0, 9
# motion
PLANE_N = 128
PLANE_MAX_ZENITH = 70.0
PLANE_EXTENT = float(np.tan(np.radians(PLANE_MAX_ZENITH)))
MOTION_MIN_GAP_S, MOTION_MAX_GAP_S = 40.0, 150.0
MOTION_MIN_RESPONSE = 0.2   # below this, directions within a day are random (trial: circular sd 76 vs 4.6 deg)
MOTION_CF_RANGE = (0.03, 0.97)
UPWIND_MINUTES = 10.0
UPWIND_HALF_WIDTH = 0.15

FEATURE_GROUPS = {
    "cover": [f"cf{k}_{reg}" for k in (1, 2, 3) for reg in ("all", "z0_30", "z30_60", "z60_75")] + ["cf_rbr_all"],
    "sun": ["sun_sat_frac_all", "sun_sat_frac_5", "sun_sat_frac_15", "sun_blob_area", "sun_detected",
            "sun_blob_offset_deg"],
    "position": [f"pos_cf_s{a}_{b}" for a, b in SUN_RINGS] + ["pos_cf1_s0_15", "pos_min_angle_cloud"],
    "texture": ["tx_sat_mean", "tx_satd_mean", "tx_satd_std", "tx_grayd_mean", "tx_grayd_absmean", "tx_rbr_mean",
                "tx_rbr_std", "tx_gray_mean", "tx_gray_std", "tx_entropy", "tx_edge", "tx_satd_mean_s30",
                "tx_grayd_mean_s30", "tx_gray_std_s30", "tx_edge_s30"],
    "motion": ["mo_valid", "mo_dt_s", "mo_vx", "mo_vy", "mo_speed", "mo_response", "mo_resid_ratio",
               "mo_upwind_cf", "mo_time_to_sun_min"],
}
ALL_FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]


def colour_maps(rgb: np.ndarray):
    f = rgb.astype(np.float32)
    mx = f.max(axis=2)
    sat = (mx - f.min(axis=2)) / (mx + 1.0)
    rbr = f[..., 0] / (f[..., 2] + 1.0)
    return sat, rbr


def gray_map(rgb: np.ndarray):
    return rgb.astype(np.float32).mean(axis=2)


def csl_bin_index(ang, pixel_zenith, sun_zenith):
    """Integer bins used to BUILD the library."""
    a = np.clip((np.asarray(ang) // CSL_ANG_STEP).astype(int), 0, CSL_N_ANG - 1)
    z = np.clip((np.asarray(pixel_zenith) // CSL_Z_STEP).astype(int), 0, CSL_N_Z - 1)
    s = np.clip((np.asarray(sun_zenith) // CSL_SZ_STEP).astype(int), 1, CSL_N_SZ - 1)
    return a, z, s


def csl_lookup(arr, ang, pixel_zenith, sun_zenith):
    """Library value, linearly interpolated between bin centres."""
    n = np.size(ang)
    coords = np.vstack([
        np.clip(np.asarray(ang, np.float32) / CSL_ANG_STEP - 0.5, 0, CSL_N_ANG - 1),
        np.clip(np.asarray(pixel_zenith, np.float32) / CSL_Z_STEP - 0.5, 0, CSL_N_Z - 1),
        np.clip(np.broadcast_to(np.asarray(sun_zenith, np.float32), (n,)) / CSL_SZ_STEP - 0.5, 0.5, CSL_N_SZ - 1),
    ])
    return map_coordinates(arr, coords, order=1, mode="nearest")


@lru_cache(maxsize=1)
def library():
    d = np.load(FEATURES_DIR / "clear_sky_library.npz")
    thr = json.loads((FEATURES_DIR / "cloud_thresholds.json").read_text())
    return {"sat": d["sat"], "gray": d["gray"], "rbr": d["rbr"], "cloud_thr": thr["cloud_thresholds"],
            "rbr_thr": thr["rbr_threshold"]}


@lru_cache(maxsize=4)
def obstruction_mask(year: int) -> np.ndarray:
    return np.load(FEATURES_DIR / f"sky_mask_{year}.npy")


class MonthContext:
    """Geometry, masks and sky-plane remap for one (year, month)."""

    def __init__(self, year: int, month: int):
        self.g = g = SkyGeometry.from_fitted(year, month)
        self.usable = g.sky & ~obstruction_mask(year)
        self.zen_ring = {(a, b): self.usable & (g.zenith >= a) & ((g.zenith < b) if b < MAX_ZENITH else (g.zenith <= b))
                         for a, b in ZEN_RINGS}
        self.usable_eroded = cv2.erode(self.usable.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        # sky plane: u = tan(z) sin(az) (east), v = tan(z) cos(az) (north); row 0 = north
        lin = np.linspace(-PLANE_EXTENT, PLANE_EXTENT, PLANE_N, dtype=np.float32)
        uu, vv = np.meshgrid(lin, lin[::-1])
        zen = np.degrees(np.arctan(np.hypot(uu, vv)))
        az = np.degrees(np.arctan2(uu, vv)) % 360.0
        ok_z = g._z_grid < 900
        r = np.interp(zen, g._z_grid[ok_z], g._r_grid[ok_z])
        phi = np.radians((g.az_sign * az + g.alpha) % 360.0)
        self.map_x = (g.cx + r * np.cos(phi)).astype(np.float32)
        self.map_y = (g.cy - r * np.sin(phi)).astype(np.float32)
        valid_src = cv2.remap(self.usable.astype(np.uint8), self.map_x, self.map_y, cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
        self.plane_valid = (zen <= PLANE_MAX_ZENITH) & valid_src
        self.plane_uu, self.plane_vv = uu, vv
        self.hann = cv2.createHanningWindow((PLANE_N, PLANE_N), cv2.CV_32F)


def _frac(mask, region):
    n = int(region.sum())
    return float((mask & region).sum() / n) if n >= MIN_RING_PX else np.nan


def _entropy(gray, region, bins=64):
    h, _ = np.histogram(gray[region], bins=bins, range=(0, 256))
    if h.sum() == 0:
        return np.nan
    p = h[h > 0] / h.sum()
    return float(-(p * np.log2(p)).sum())


class FrameData:
    """What the next frame needs for motion: this frame's plane and its difference from the frame before."""

    def __init__(self, t_img, plane, diff=None):
        self.t_img = t_img
        self.plane = plane
        self.diff = diff  # plane(t) - plane(t - 1 frame), NaN where either is invalid


def compute(rgb: np.ndarray, ctx: MonthContext, sun_zen: float, sun_az: float, t_img, prev: FrameData | None,
            return_masks: bool = False):
    """Return (features dict, FrameData for the next frame[, masks for debug overlays])."""
    lib = library()
    g = ctx.g
    f = {}
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sat_px = rgb.min(axis=2) >= SAT_LEVEL
    colour_sat, rbr = colour_maps(rgb)
    gray_f = gray_map(rgb)
    usable = ctx.usable
    clf_region = usable & ~sat_px
    ang = g.angle_to(sun_zen, sun_az)
    sat_d = np.zeros_like(colour_sat)
    gray_d = np.zeros_like(colour_sat)
    rbr_d = np.zeros_like(colour_sat)
    u = usable
    sat_d[u] = colour_sat[u] - csl_lookup(lib["sat"], ang[u], g.zenith[u], sun_zen)
    gray_d[u] = gray_f[u] - csl_lookup(lib["gray"], ang[u], g.zenith[u], sun_zen)
    rbr_d[u] = rbr[u] - csl_lookup(lib["rbr"], ang[u], g.zenith[u], sun_zen)

    # cover
    clouds = [((sat_d < ts) | (np.abs(gray_d) > tg)) & clf_region for ts, tg in lib["cloud_thr"]]
    regions = {"all": clf_region, **{f"z{a}_{int(b)}": ctx.zen_ring[(a, b)] & ~sat_px for a, b in ZEN_RINGS}}
    for k, cm in enumerate(clouds, start=1):
        for name, reg in regions.items():
            f[f"cf{k}_{name}"] = _frac(cm, reg)
    f["cf_rbr_all"] = _frac((rbr_d > lib["rbr_thr"]) & clf_region, clf_region)

    # sun area
    f["sun_sat_frac_all"] = _frac(sat_px, usable)
    f["sun_sat_frac_5"] = _frac(sat_px, g.inside & (ang <= 5.0))
    f["sun_sat_frac_15"] = _frac(sat_px, g.inside & (ang <= 15.0))
    blob = detect_sun_blob(rgb, g.inside & (ang <= SUN_SEARCH_DEG))
    if blob is None:
        f["sun_blob_area"], f["sun_detected"], f["sun_blob_offset_deg"] = 0.0, 0.0, np.nan
    else:
        area, bx, by = blob
        f["sun_blob_area"] = float(area)
        f["sun_detected"] = float(area >= SUN_DETECTED_MIN_AREA)
        f["sun_blob_offset_deg"] = float(ang[int(round(by)), int(round(bx))])

    # position relative to the sun (middle threshold; the sensitive one for the inner ring)
    cmid = clouds[1]
    for a, b in SUN_RINGS:
        f[f"pos_cf_s{a}_{b}"] = _frac(cmid, clf_region & (ang >= a) & (ang < b))
    f["pos_cf1_s0_15"] = _frac(clouds[0], clf_region & (ang < 15))
    f["pos_min_angle_cloud"] = float(ang[cmid].min()) if cmid.any() else np.nan

    # texture
    ue = ctx.usable_eroded
    n_clf = int(clf_region.sum())
    f["tx_sat_mean"] = float(colour_sat[clf_region].mean()) if n_clf else np.nan
    f["tx_satd_mean"] = float(sat_d[clf_region].mean()) if n_clf else np.nan
    f["tx_satd_std"] = float(sat_d[clf_region].std()) if n_clf else np.nan
    f["tx_grayd_mean"] = float(gray_d[clf_region].mean()) if n_clf else np.nan
    f["tx_grayd_absmean"] = float(np.abs(gray_d[clf_region]).mean()) if n_clf else np.nan
    f["tx_rbr_mean"] = float(rbr[clf_region].mean()) if n_clf else np.nan
    f["tx_rbr_std"] = float(rbr[clf_region].std()) if n_clf else np.nan
    f["tx_gray_mean"], f["tx_gray_std"] = float(gray[usable].mean()), float(gray[usable].std())
    f["tx_entropy"] = _entropy(gray, usable)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.hypot(gx, gy)
    f["tx_edge"] = float(edge[ue].mean())
    s30 = clf_region & (ang < 30)
    if s30.sum() >= MIN_RING_PX:
        f["tx_satd_mean_s30"] = float(sat_d[s30].mean())
        f["tx_grayd_mean_s30"] = float(gray_d[s30].mean())
        f["tx_gray_std_s30"] = float(gray[s30].std())
        f["tx_edge_s30"] = float(edge[s30 & ue].mean()) if (s30 & ue).any() else np.nan
    else:
        f["tx_satd_mean_s30"] = f["tx_grayd_mean_s30"] = f["tx_gray_std_s30"] = f["tx_edge_s30"] = np.nan

    # motion on the sky plane, using a cloudiness score (clouds high): max of the two cues,
    # each scaled by its middle threshold, so score > 1 means cloud at threshold 2
    ts2, tg2 = lib["cloud_thr"][1]
    score = np.maximum(sat_d / ts2, np.abs(gray_d) / tg2)
    field = np.where(clf_region, np.clip(score, 0, 5), -10.0).astype(np.float32)
    plane = cv2.remap(field, ctx.map_x, ctx.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-10.0)
    plane[~ctx.plane_valid | (plane < -1.0)] = np.nan
    for k in FEATURE_GROUPS["motion"]:
        f[k] = np.nan
    f["mo_valid"] = 0.0
    # Static patterns (dome reflections, glare, residual clear-sky structure) dominate a
    # plain frame-to-frame correlation and lock it at zero shift, so motion is estimated by
    # correlating successive frame DIFFERENCES: (t-2 -> t-1) against (t-1 -> t).
    motion_dbg = None
    diff = None
    if prev is not None:
        dt = (t_img - prev.t_img).total_seconds()
        f["mo_dt_s"] = dt
        if MOTION_MIN_GAP_S <= dt <= MOTION_MAX_GAP_S:
            diff = plane - prev.plane
            cf_mid = f["cf2_all"]
            ok_cf = cf_mid == cf_mid and MOTION_CF_RANGE[0] <= cf_mid <= MOTION_CF_RANGE[1]
            if ok_cf and prev.diff is not None:
                motion_dbg = _motion(f, prev.diff, diff, prev.plane, plane, dt, ctx, sun_zen, sun_az, 1.0)
    out = (f, FrameData(t_img, plane, diff))
    if return_masks:
        return out + ({"clouds": clouds, "sat": sat_px, "usable": usable, "ang": ang, "sat_d": sat_d, "gray_d": gray_d,
                       "motion": motion_dbg},)
    return out


def _fill(a):
    m = np.isnan(a)
    out = a.copy()
    out[m] = np.nanmean(a) if (~m).any() else 0.0
    return out, ~m


def _motion(f, prev_diff, cur_diff, prev_plane, cur_plane, dt, ctx, sun_zen, sun_az, cloud_level):
    d0 = np.nan_to_num(prev_diff, nan=0.0).astype(np.float32)
    d1 = np.nan_to_num(cur_diff, nan=0.0).astype(np.float32)
    if np.isfinite(cur_diff).sum() < 500 or not d0.any() or not d1.any():
        return None
    (dx, dy), resp = cv2.phaseCorrelate(d0, d1, ctx.hann)
    p, pv = _fill(prev_plane)
    c, cv = _fill(cur_plane)
    both = pv & cv
    f["mo_response"] = float(resp)
    if resp < MOTION_MIN_RESPONSE:
        return None
    cell = 2 * PLANE_EXTENT / (PLANE_N - 1)
    per_min = 60.0 / dt
    vx, vy = dx * cell * per_min, -dy * cell * per_min  # plane rows run north -> south
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(p, M, (PLANE_N, PLANE_N), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    sv = cv2.warpAffine(pv.astype(np.uint8), M, (PLANE_N, PLANE_N), flags=cv2.INTER_NEAREST) > 0
    m = sv & cv
    base = float(np.abs(c[both] - p[both]).mean())
    resid = float(np.abs(c[m] - shifted[m]).mean()) if m.sum() >= 500 else np.nan
    f["mo_valid"] = 1.0
    f["mo_vx"], f["mo_vy"] = float(vx), float(vy)
    f["mo_speed"] = float(np.hypot(vx, vy))
    f["mo_resid_ratio"] = resid / base if base > 0 else np.nan
    dbg = {"dx": dx, "dy": dy, "vx": vx, "vy": vy}

    # upwind of the sun: corridor from the sun's plane position, against the motion
    if sun_zen > PLANE_MAX_ZENITH or f["mo_speed"] <= 0:
        return dbg
    rho = np.tan(np.radians(sun_zen))
    su, sv_ = rho * np.sin(np.radians(sun_az)), rho * np.cos(np.radians(sun_az))
    ux, uy = -vx / f["mo_speed"], -vy / f["mo_speed"]
    reach = min(f["mo_speed"] * UPWIND_MINUTES, 2 * PLANE_EXTENT)
    du, dv = ctx.plane_uu - su, ctx.plane_vv - sv_
    along = du * ux + dv * uy
    across = np.abs(-du * uy + dv * ux)
    corridor = (along >= 0) & (along <= reach) & (across <= UPWIND_HALF_WIDTH) & cv
    if corridor.sum() < 20:
        return dbg
    cloud = c > cloud_level
    f["mo_upwind_cf"] = float(cloud[corridor].mean())
    hit = corridor & cloud
    f["mo_time_to_sun_min"] = float(along[hit].min() / f["mo_speed"]) if hit.any() else np.nan
    dbg.update({"sun_uv": (su, sv_), "reach": reach})
    return dbg
