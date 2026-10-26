"""Phase 1 (read-only): rebuild the sample list exactly as FolsomDataset does,
then measure where each sample's image sits relative to t.

Outputs go to the scratchpad only. Nothing in the repo is written.
"""
import bisect
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO = "/storage2/CV_Irradiance/Multimodal_Solar_Irradiance"
sys.path.insert(0, REPO)
from datasets.folsom_dataset import FolsomDataset  # noqa: E402
from datasets.chrono import chronological_year_split  # noqa: E402

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "data": {
        "csv_path": "/storage2/CV_Irradiance/datasets/1_Folsom/csv_files/Folsom_irradiance_weather.csv",
        "image_root": "/storage2/CV_Irradiance/datasets/1_Folsom",
        "sequence_length": 40,
        "sampling_rate_sec": 60,
        "image_tolerance_sec": 120,
        "load_images": False,
    },
    "model": {"horizons": [10]},
    "training": {"val_split": 0.1},
}

report = {}

# ---- raw CSV facts ---------------------------------------------------------
raw = pd.read_csv(CONFIG["data"]["csv_path"])
raw["Datetime"] = pd.to_datetime(raw["Datetime"])
report["csv_rows_raw"] = int(len(raw))
report["csv_first"] = str(raw["Datetime"].min())
report["csv_last"] = str(raw["Datetime"].max())
report["csv_duplicate_timestamps"] = int(raw["Datetime"].duplicated().sum())
report["csv_nonzero_seconds"] = int((raw["Datetime"].dt.second != 0).sum())
report["csv_rows_missing_required"] = int(
    raw[["GHI", "temperature", "pressure"]].isna().any(axis=1).sum()
)

ds = FolsomDataset(CONFIG)
report["df_rows_after_sza85"] = int(len(ds.df))
report["df_duplicate_index"] = int(ds.df.index.duplicated().sum())

tr, va, te = chronological_year_split(ds.samples, 0.1, (2014, 2015), 2016)
report["n_samples"] = len(ds.samples)
report["n_train"], report["n_val"], report["n_test"] = len(tr), len(va), len(te)
for name, idx in (("train", tr), ("val", va), ("test", te)):
    report[f"{name}_first"] = str(ds.samples[idx[0]][1])
    report[f"{name}_last"] = str(ds.samples[idx[-1]][1])
times = [s[1] for s in ds.samples]
report["samples_sorted"] = bool(all(a < b for a, b in zip(times, times[1:])))


# ---- image offsets ---------------------------------------------------------
def parse(name):
    stem = os.path.splitext(os.path.basename(name))[0]
    for fmt in ("%Y%m%d_%H%M%S", "%Y-%m-%d_%H-%M-%S", "%Y-%m-%d_%H-%M"):
        try:
            return datetime.strptime(stem, fmt)
        except Exception:
            pass
    return None


cache = {}


def day_index(dt):
    key = dt.date()
    if key not in cache:
        cache[key] = ds._build_day_image_index(
            os.path.join(ds.image_root, dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%d"))
        )
    return cache[key]


def latest_at_or_before(dt, max_age_s=120):
    """Latest image with time <= dt and >= dt - max_age_s (same-day folder,
    plus the previous day's folder for samples just after midnight)."""
    best = None
    for d in (dt, dt - timedelta(days=1)):
        idx = day_index(d)
        if not idx:
            continue
        pos = bisect.bisect_right(idx["timestamps"], dt)
        if pos > 0:
            ts = idx["timestamps"][pos - 1]
            if best is None or ts > best:
                best = ts
    if best is not None and (dt - best).total_seconds() <= max_age_s:
        return best
    return None


n = len(ds.samples)
cnn_off = np.empty(n)          # CNN image time - t (s); >0 means after t
le_age = np.full(n, np.nan)    # t - latest image <= t (s), within 120 s
prev_gap = np.full(n, np.nan)  # gap between that image and the one before it (s)
for i, (path, dt) in enumerate(ds.samples):
    cnn_off[i] = (parse(path) - dt).total_seconds()
    ts = latest_at_or_before(dt)
    if ts is not None:
        le_age[i] = (dt - ts).total_seconds()
        idx = day_index(ts)
        p = bisect.bisect_left(idx["timestamps"], ts)
        if p > 0:
            prev_gap[i] = (ts - idx["timestamps"][p - 1]).total_seconds()

split = np.empty(n, dtype=object)
split[tr], split[va], split[te] = "train", "val", "test"
pos = np.empty(n, dtype=np.int64)
pos[tr], pos[va], pos[te] = np.arange(len(tr)), np.arange(len(va)), np.arange(len(te))

tab = pd.DataFrame({
    "datetime": times, "split": split, "split_pos": pos,
    "cnn_img_offset_s": cnn_off, "le_img_age_s": le_age, "le_prev_gap_s": prev_gap,
})
tab["cnn_img_after_t"] = tab["cnn_img_offset_s"] > 0
tab["has_le_image"] = tab["le_img_age_s"].notna()
# Same image under both rules <=> CNN's nearest image is at or before t.
tab["common_subset"] = ~tab["cnn_img_after_t"]
tab.to_pickle(OUT / "sample_alignment.pkl")

summ = {}
for name in ("train", "val", "test"):
    s = tab[tab["split"] == name]
    off = s["cnn_img_offset_s"]
    summ[name] = {
        "n": int(len(s)),
        "cnn_after_t_frac": float((off > 0).mean()),
        "cnn_after_t_n": int((off > 0).sum()),
        "cnn_offset_quantiles_s": {q: float(off.quantile(q)) for q in (0, 0.05, 0.25, 0.5, 0.75, 0.95, 1)},
        "cnn_after_t_gt30s_n": int((off > 30).sum()),
        "cnn_after_t_gt60s_n": int((off > 60).sum()),
        "le_image_available_frac": float(s["has_le_image"].mean()),
        "le_age_quantiles_s": {q: float(s["le_img_age_s"].quantile(q)) for q in (0.05, 0.5, 0.95, 1)},
        "le_age_le30s_frac": float((s["le_img_age_s"] <= 30).mean()),
        "le_prev_gap_le90s_frac": float((s["le_prev_gap_s"] <= 90).mean()),
        "common_subset_n": int(s["common_subset"].sum()),
    }
report["image_alignment"] = summ
offs = tab["cnn_img_offset_s"].round().astype(int)
report["cnn_offset_histogram_top"] = {int(k): int(v) for k, v in offs.value_counts().head(15).items()}

# ---- saved predictions line up with this sample order? ---------------------
test_times = tab.loc[tab["split"] == "test", "datetime"]
k_ref = np.array([float(ds.target_df.loc[t + timedelta(minutes=10), "k_index"]) for t in test_times])
checks = {}
for run in sorted(Path(REPO, "experiments").glob("*/evaluation/test_predictions.npz")):
    d = np.load(run)
    kt = d["k_index_true"]
    checks[run.parent.parent.name] = {
        "n": int(len(kt)),
        "max_abs_diff_vs_rebuilt_target": float(np.max(np.abs(kt - k_ref))) if len(kt) == len(k_ref) else None,
    }
report["saved_prediction_order_check"] = checks

(OUT / "p1_samples.json").write_text(json.dumps(report, indent=1, default=str))
print(json.dumps(report, indent=1, default=str))
