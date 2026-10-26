"""Benchmark: GPU (torch, GPU 0) vs CPU (frozen v1 numpy/scipy) for the hot path of
cloud_features.compute: colour maps, 3 clear-sky lookups, cloud masks, cover fractions.
Checks exact equality of the 13 cover features against the frozen CPU output."""
import sys, time
sys.path.insert(0, "/storage2/CV_Irradiance/Multimodal_Solar_Irradiance/scripts/lgbm_cloud")
import numpy as np, pandas as pd, pvlib, torch, cv2
cv2.setNumThreads(1)
from cloud_features import (MonthContext, library, colour_maps, gray_map, csl_lookup, SAT_LEVEL, ZEN_RINGS,
                            CSL_ANG_STEP, CSL_Z_STEP, CSL_SZ_STEP, CSL_N_ANG, CSL_N_Z, CSL_N_SZ, _frac)
from sky_geometry import read_small_rgb, unit_vector
from common import FEATURES_DIR

dev = torch.device("cuda:0")
ANG_ON_CPU = len(sys.argv) > 1 and sys.argv[1] == "ang_cpu"
lib = library()
DAYS = ["2014/10/31", "2015/02/14", "2015/09/29", "2016/03/10"]
idx = pd.read_parquet(FEATURES_DIR / "image_index.parquet", columns=["path", "day_dir", "t_img", "sza_t_img", "corrupted"])
sel = idx[idx.day_dir.isin(DAYS) & (idx.sza_t_img <= 87) & ~idx.corrupted].sort_values("t_img")
sel = sel.groupby("day_dir").head(250)
site = pvlib.location.Location(38.642, -121.148, altitude=60.0)
sp = site.get_solarposition(pd.DatetimeIndex(sel.t_img))
sel["zen"], sel["az"] = sp.apparent_zenith.to_numpy(), sp.azimuth.to_numpy()
imgs = [read_small_rgb(p) for p in sel.path]
ref = pd.read_parquet(FEATURES_DIR / "cloud_features.parquet", filters=[("path", "in", list(sel.path))]).set_index("path")
COVER = [f"cf{k}_{r}" for k in (1, 2, 3) for r in ("all", "z0_30", "z30_60", "z60_75")] + ["cf_rbr_all"]
ctxs = {}
def ctx_of(t):
    k = (t.year, t.month)
    if k not in ctxs: ctxs[k] = MonthContext(*k)
    return ctxs[k]

# ---------------- CPU hot path (exact copy of the frozen code's steps) ----------------
def cpu_cover(rgb, ctx, zen, az):
    g = ctx.g
    sat_px = rgb.min(axis=2) >= SAT_LEVEL
    colour_sat, rbr = colour_maps(rgb); gray_f = gray_map(rgb)
    usable = ctx.usable; clf = usable & ~sat_px
    ang = g.angle_to(zen, az)
    sat_d = np.zeros_like(colour_sat); gray_d = np.zeros_like(colour_sat); rbr_d = np.zeros_like(colour_sat)
    u = usable
    sat_d[u] = colour_sat[u] - csl_lookup(lib["sat"], ang[u], g.zenith[u], zen)
    gray_d[u] = gray_f[u] - csl_lookup(lib["gray"], ang[u], g.zenith[u], zen)
    rbr_d[u] = rbr[u] - csl_lookup(lib["rbr"], ang[u], g.zenith[u], zen)
    clouds = [((sat_d < ts) | (np.abs(gray_d) > tg)) & clf for ts, tg in lib["cloud_thr"]]
    regions = {"all": clf, **{f"z{a}_{int(b)}": ctx.zen_ring[(a, b)] & ~sat_px for a, b in ZEN_RINGS}}
    f = {}
    for k, cm in enumerate(clouds, 1):
        for n, reg in regions.items(): f[f"cf{k}_{n}"] = _frac(cm, reg)
    f["cf_rbr_all"] = _frac((rbr_d > lib["rbr_thr"]) & clf, clf)
    return f

# ---------------- GPU hot path (torch; float32 elementwise as CPU, float64 interpolation as scipy) -------------
LIB_T = {v: torch.tensor(lib[v], dtype=torch.float64, device=dev) for v in ("sat", "gray", "rbr")}
def gpu_lookup(arr, ang, pz, sz):
    # coords exactly as csl_lookup: float32 arithmetic, then scipy's float64 linear interpolation, cast to float32
    a = torch.clamp(ang / np.float32(CSL_ANG_STEP) - np.float32(0.5), 0, CSL_N_ANG - 1)
    z = torch.clamp(pz / np.float32(CSL_Z_STEP) - np.float32(0.5), 0, CSL_N_Z - 1)
    s = torch.clamp(torch.full_like(ang, np.float32(sz)) / np.float32(CSL_SZ_STEP) - np.float32(0.5), 0.5, CSL_N_SZ - 1)
    out = torch.zeros(ang.shape, dtype=torch.float64, device=dev)
    c = [a.double(), z.double(), s.double()]
    f0 = [torch.floor(x) for x in c]
    w1 = [x - f for x, f in zip(c, f0)]
    n = (CSL_N_ANG, CSL_N_Z, CSL_N_SZ)
    for da in (0, 1):
        for dz in (0, 1):
            for ds in (0, 1):
                i = [torch.clamp(f0[k] + d, 0, n[k] - 1).long() for k, d in enumerate((da, dz, ds))]
                w = ((w1[0] if da else 1 - w1[0]) * (w1[1] if dz else 1 - w1[1]) * (w1[2] if ds else 1 - w1[2]))
                out += w * arr[i[0], i[1], i[2]]
    return out.float()

gctx = {}
def gpu_ctx(ctx):
    k = id(ctx)
    if k not in gctx:
        g = ctx.g
        gctx[k] = {"vec": torch.tensor(g.vec, device=dev), "zen": torch.tensor(g.zenith.astype(np.float32), device=dev),
                   "usable": torch.tensor(ctx.usable, device=dev),
                   "rings": {r: torch.tensor(m, device=dev) for r, m in ctx.zen_ring.items()}}
    return gctx[k]

def gpu_cover(rgb_t, ctx, zen, az):
    G = gpu_ctx(ctx)
    f32 = rgb_t.float()
    mx, mn = f32.max(dim=2).values, f32.min(dim=2).values
    sat_px = rgb_t.min(dim=2).values >= SAT_LEVEL
    colour_sat = (mx - mn) / (mx + np.float32(1.0))
    rbr = f32[..., 0] / (f32[..., 2] + np.float32(1.0))
    gray_f = f32.mean(dim=2)
    ang = torch.from_numpy(ctx.g.angle_to(zen, az)).to(dev) if ANG_ON_CPU else \
        torch.rad2deg(torch.arccos(torch.clamp(G["vec"] @ torch.tensor(unit_vector(zen, az).astype(np.float32), device=dev), -1.0, 1.0)))
    u = G["usable"]; clf = u & ~sat_px
    au, zu = ang[u], G["zen"][u]
    sat_d = torch.zeros_like(colour_sat); gray_d = torch.zeros_like(colour_sat); rbr_d = torch.zeros_like(colour_sat)
    sat_d[u] = colour_sat[u] - gpu_lookup(LIB_T["sat"], au, zu, zen)
    gray_d[u] = gray_f[u] - gpu_lookup(LIB_T["gray"], au, zu, zen)
    rbr_d[u] = rbr[u] - gpu_lookup(LIB_T["rbr"], au, zu, zen)
    clouds = [((sat_d < np.float32(ts)) | (gray_d.abs() > np.float32(tg))) & clf for ts, tg in lib["cloud_thr"]]
    regions = {"all": clf, **{f"z{a}_{int(b)}": G["rings"][(a, b)] & ~sat_px for a, b in ZEN_RINGS}}
    counts = []
    for cm in clouds:
        for reg in regions.values(): counts += [(cm & reg).sum(), reg.sum()]
    rb = (rbr_d > np.float32(lib["rbr_thr"])) & clf
    counts += [rb.sum(), clf.sum()]
    c = torch.stack(counts).cpu().numpy()  # one sync per image
    f, i = {}, 0
    for k in (1, 2, 3):
        for n in regions:
            f[f"cf{k}_{n}"] = float(c[i] / c[i + 1]) if c[i + 1] >= 50 else np.nan; i += 2
    f["cf_rbr_all"] = float(c[i] / c[i + 1]) if c[i + 1] >= 50 else np.nan
    return f

rows = list(zip(imgs, sel.t_img, sel.zen, sel.az, sel.path))
# CPU timing (single thread, hot path only)
t0 = time.perf_counter(); cpu = {p: cpu_cover(im, ctx_of(t), float(z), float(a)) for im, t, z, a, p in rows}; t_cpu = time.perf_counter() - t0
# GPU timing incl. host->device transfer, warm-up first
for im, t, z, a, p in rows[:20]: gpu_cover(torch.from_numpy(im).to(dev), ctx_of(t), float(z), float(a))
torch.cuda.synchronize(); t0 = time.perf_counter()
gpu = {p: gpu_cover(torch.from_numpy(im).to(dev), ctx_of(t), float(z), float(a)) for im, t, z, a, p in rows}
torch.cuda.synchronize(); t_gpu = time.perf_counter() - t0

n = len(rows)
cpu_df, gpu_df = pd.DataFrame(cpu).T[COVER].astype(float), pd.DataFrame(gpu).T[COVER].astype(float)
refd = ref.loc[cpu_df.index, COVER].astype(float)
print(f"images {n}")
print(f"CPU hot path : {1000 * t_cpu / n:.2f} ms/image (1 thread)")
print(f"GPU hot path : {1000 * t_gpu / n:.2f} ms/image (1 process, GPU 0, incl. transfer)")
print("CPU re-run == frozen output (bit-identical):", bool((cpu_df.fillna(-9) == refd.fillna(-9)).all().all()))
eq = (gpu_df.fillna(-9) == refd.fillna(-9))
print("GPU == frozen output: images with any differing cover feature:", int((~eq.all(axis=1)).sum()), "of", n)
d = (gpu_df - refd).abs()
print("max |GPU - frozen| per feature:", d.max().round(6).to_dict())
