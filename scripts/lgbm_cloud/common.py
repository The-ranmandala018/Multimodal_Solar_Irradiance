"""Shared paths, dataset config and helpers for the LightGBM cloud-feature experiment.

Nothing here changes the existing CNN/LSTM pipeline: the dataset config below is
the same one used by scripts/train_gpu.py, scripts/evaluate_test.py and
scripts/analysis_bias_and_sky_conditions.py, with images switched off.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

DATA_ROOT = Path("/storage2/CV_Irradiance/datasets/1_Folsom")
CSV_PATH = DATA_ROOT / "csv_files" / "Folsom_irradiance_weather.csv"
CORRUPTED_LIST = DATA_ROOT / "corrupted_images.txt"

EXP_DIR = PROJECT_DIR / "experiments" / "lightgbm_cloud_features"
CHECKS_DIR = EXP_DIR / "checks"
FEATURES_DIR = EXP_DIR / "features"
RUNS_DIR = EXP_DIR / "runs"
SUMMARY_DIR = EXP_DIR / "summary"

LATITUDE, LONGITUDE, ALTITUDE = 38.642, -121.148, 60.0
HORIZON_MIN = 10
SEQ_LEN = 40
IMAGE_MAX_AGE_S = 120

# Expected split sizes (verified in Phase 1).
EXPECTED_SPLITS = {"train": 389442, "val": 43271, "test": 224022}

# Weather-history column order inside the 40 x 7 window (FolsomDataset.feature_cols).
WINDOW_COLS = ["k_index", "temperature", "pressure", "SZA", "Azimuth", "sin_hour", "cos_hour"]


def dataset_config(load_images: bool = False) -> dict:
    return {
        "data": {
            "csv_path": str(CSV_PATH),
            "image_root": str(DATA_ROOT),
            "sequence_length": SEQ_LEN,
            "sampling_rate_sec": 60,
            "image_tolerance_sec": 120,
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "altitude": ALTITUDE,
            "load_images": load_images,
        },
        "model": {"horizons": [HORIZON_MIN]},
        "training": {"val_split": 0.1},
    }


def refuse_existing(path: Path) -> Path:
    """Refuse to overwrite: raise if `path` (file or folder) already exists."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists; refusing to overwrite. Remove it yourself to rerun.")
    return path


def versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ("numpy", "pandas", "pvlib", "sklearn", "lightgbm", "shap", "pyarrow", "cv2", "torch"):
        try:
            out[name] = __import__(name).__version__
        except Exception:
            out[name] = None
    return out


def write_json(path: Path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=1, default=str))


def git_head() -> str:
    try:
        return os.popen(f"git -C {PROJECT_DIR} rev-parse --short HEAD").read().strip()
    except Exception:
        return "unknown"


def regression_metrics(y_true, y_pred) -> dict:
    """Same definitions as scripts/evaluate_test.py and analysis_bias_and_sky_conditions.metrics."""
    y_true = np.asarray(y_true, np.float64)
    err = np.asarray(y_pred, np.float64) - y_true
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    mean = float(y_true.mean())
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return {
        "n": int(len(y_true)),
        "MAE_Wm2": mae,
        "RMSE_Wm2": rmse,
        "MBE_Wm2": float(np.mean(err)),
        "R2": 1.0 - float(np.sum(err ** 2)) / ss_tot if ss_tot > 0 else float("nan"),
        "nMAE": mae / mean,
        "nRMSE": rmse / mean,
    }
