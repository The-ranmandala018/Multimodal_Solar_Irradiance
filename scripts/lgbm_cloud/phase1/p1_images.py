"""Phase 1 (read-only): per-image filename time vs file modified time, and
image intervals, for every image in 2014-2016. Writes to the scratchpad only."""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

ROOT = Path("/storage2/CV_Irradiance/datasets/1_Folsom")
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

rows = []
for year in ("2014", "2015", "2016"):
    for day_dir in sorted(ROOT.glob(f"{year}/[0-1][0-9]/[0-3][0-9]")):
        with os.scandir(day_dir) as it:
            for e in it:
                if not e.name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                try:
                    ts = datetime.strptime(os.path.splitext(e.name)[0], "%Y%m%d_%H%M%S")
                except ValueError:
                    continue
                st = e.stat()
                rows.append((ts, st.st_mtime, st.st_size, str(day_dir.relative_to(ROOT)), e.name))

img = pd.DataFrame(rows, columns=["ts_name", "mtime_epoch", "size", "day_dir", "file"])
img["ts_mtime"] = pd.to_datetime(img["mtime_epoch"], unit="s")  # UTC, naive
img = img.sort_values("ts_name").reset_index(drop=True)
img["name_minus_mtime_s"] = (img["ts_name"] - img["ts_mtime"]).dt.total_seconds()

site = pvlib.location.Location(38.642, -121.148, altitude=60.0)
img["sza_name"] = site.get_solarposition(pd.DatetimeIndex(img["ts_name"]))["zenith"].values
img.to_pickle(OUT / "image_index.pkl")

day = pd.to_datetime(img["day_dir"].str.replace("/", "-"))
img["month"] = day.dt.to_period("M").astype(str)
daylight = img["sza_name"] <= 85

rep = {"n_images": int(len(img)), "n_daylight": int(daylight.sum()),
       "server_tz_note": "ts_mtime from st_mtime epoch, i.e. UTC regardless of server TZ"}

# per-month offset (filename - mtime) and intervals
months = []
for m, g in img[daylight].groupby("month"):
    off = g["name_minus_mtime_s"]
    gaps_name = g["ts_name"].diff().dt.total_seconds()
    gaps_mt = g["ts_mtime"].diff().dt.total_seconds()
    same_day = g["day_dir"].eq(g["day_dir"].shift())
    gaps_name, gaps_mt = gaps_name[same_day], gaps_mt[same_day]
    months.append({
        "month": m, "n_img": int(len(g)), "n_days": int(g["day_dir"].nunique()),
        "off_med_s": float(off.median()), "off_p05_s": float(off.quantile(.05)),
        "off_p95_s": float(off.quantile(.95)), "off_min_s": float(off.min()), "off_max_s": float(off.max()),
        "gap_name_med_s": float(gaps_name.median()), "gap_name_p95_s": float(gaps_name.quantile(.95)),
        "gap_mtime_med_s": float(gaps_mt.median()),
        "gaps_gt2min": int((gaps_name > 120).sum()), "gaps_gt5min": int((gaps_name > 300).sum()),
        "mtime_nonmonotone": int((gaps_mt <= 0).sum()),
    })
mdf = pd.DataFrame(months)
mdf.to_csv(OUT / "image_offsets_by_month.csv", index=False)

# per-day offset for plotting / step detection
ddf = img[daylight].groupby("day_dir")["name_minus_mtime_s"].agg(["median", "min", "max", "count"])
ddf["within_day_range_s"] = ddf["max"] - ddf["min"]
ddf.to_csv(OUT / "image_offsets_by_day.csv")

rep["within_day_range_quantiles_s"] = {q: float(ddf["within_day_range_s"].quantile(q)) for q in (.5, .9, .99, 1)}
rep["name_seconds_top"] = {int(k): int(v) for k, v in img["ts_name"].dt.second.value_counts().head(8).items()}
rep["mtime_seconds_top"] = {int(k): int(v) for k, v in img["ts_mtime"].dt.second.value_counts().head(8).items()}
rep["by_year_offset_median_s"] = {str(y): float(g["name_minus_mtime_s"].median())
                                  for y, g in img[daylight].groupby(img["ts_name"].dt.year)}
(OUT / "p1_images.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))
print(mdf.to_string(index=False))
