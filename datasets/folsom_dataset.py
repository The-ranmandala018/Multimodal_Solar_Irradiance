"""Folsom multimodal Dataset using the existing Oshadha/Mamba CSV contract.

The CSV interpretation follows the benchmark loader:
Datetime/Date, GHI, DNI, DHI, temperature, pressure.
Derived features: k_index, temperature, pressure, SZA, Azimuth,
sin_hour, cos_hour. Weather history is 40 x 7.

Visual data are NOT resized to 224x224. Instead, each matched Folsom image
is processed by preprocessing.FolsomPreprocessor and returned as (10,180,720).
No generated 3D .npy files are required.
"""

from __future__ import annotations

import bisect
import os
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd
import pvlib
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from preprocessing.folsom_adapter import FolsomPreprocessor, DEFAULT_IMAGE_ROOT
from .chrono import chronological_year_split


class FolsomDataset(Dataset):
    def __init__(self, config):
        self.config = config
        data_cfg = config["data"]

        self.csv_path = data_cfg["csv_path"]
        self.image_root = data_cfg.get("image_root", DEFAULT_IMAGE_ROOT)
        self.sequence_length = int(data_cfg.get("sequence_length", 40))
        self.sampling_rate = int(data_cfg.get("sampling_rate_sec", 60))
        self.image_tolerance = timedelta(
            seconds=int(data_cfg.get("image_tolerance_sec", 120))
        )
        self.horizons = [int(h) for h in config.get("model", {}).get("horizons", [10])]
        self.lat = float(data_cfg.get("latitude", 38.642))
        self.lon = float(data_cfg.get("longitude", -121.148))
        self.alt = float(data_cfg.get("altitude", 60.0))

        self.df = self._load_csv()
        self.df = self._add_solar_features(self.df)
        self.df = self.df[self.df["SZA"] <= 85].copy()

        self.feature_cols = [
            "k_index",
            "temperature",
            "pressure",
            "SZA",
            "Azimuth",
            "sin_hour",
            "cos_hour",
        ]

        self.mean = self.df[self.feature_cols].mean()
        self.std = self.df[self.feature_cols].std()
        self.df[self.feature_cols] = (
            self.df[self.feature_cols] - self.mean
        ) / (self.std + 1e-6)

        self.samples = self._match_samples()
        self._preprocessors = {}

    def _load_csv(self):
        df = pd.read_csv(self.csv_path)
        col = "Datetime" if "Datetime" in df.columns else "Date"
        if col == "Datetime":
            df["Datetime"] = pd.to_datetime(df[col])
        else:
            df["Datetime"] = pd.to_datetime(
                df[col], format="%Y%m%d%H%M%S"
            )

        for col_name in ["GHI", "DNI", "DHI", "temperature", "pressure"]:
            if col_name in df.columns:
                df[col_name] = pd.to_numeric(df[col_name], errors="coerce")

        required = ["GHI", "temperature", "pressure"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required CSV columns: {missing}")

        return df.sort_values("Datetime").set_index("Datetime").dropna(
            subset=required
        )

    def _add_solar_features(self, df):
        site = pvlib.location.Location(
            self.lat, self.lon, altitude=self.alt
        )
        solar_position = site.get_solarposition(df.index)
        df = df.copy()
        df["SZA"] = solar_position["zenith"]
        df["Azimuth"] = solar_position["azimuth"]
        df["GHI_cs"] = site.get_clearsky(
            df.index, model="ineichen"
        )["ghi"]
        df["k_index"] = (
            df["GHI"] / (df["GHI_cs"] + 1e-6)
        ).clip(0.0, 1.2)
        hour = df.index.hour + df.index.minute / 60.0
        df["sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
        df["cos_hour"] = np.cos(2 * np.pi * hour / 24.0)
        return df

    def _build_day_image_index(self, day_dir):
        if not os.path.isdir(day_dir):
            return None

        timestamps, files = [], []
        for name in os.listdir(day_dir):
            if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            stem = os.path.splitext(name)[0]
            parsed = None
            for fmt in ("%Y%m%d_%H%M%S", "%Y-%m-%d_%H-%M-%S", "%Y-%m-%d_%H-%M"):
                try:
                    parsed = pd.Timestamp.strptime(stem, fmt).to_pydatetime()
                    break
                except Exception:
                    continue
            if parsed is None:
                continue
            timestamps.append(parsed)
            files.append(name)

        if not timestamps:
            return None

        order = np.argsort(timestamps)
        return {
            "dir": day_dir,
            "timestamps": [timestamps[i] for i in order],
            "files": [files[i] for i in order],
        }

    def _closest_image(self, dt, cache):
        day_key = dt.date()
        if day_key not in cache:
            cache[day_key] = self._build_day_image_index(
                os.path.join(
                    self.image_root,
                    dt.strftime("%Y"),
                    dt.strftime("%m"),
                    dt.strftime("%d"),
                )
            )

        index = cache[day_key]
        if not index:
            return None

        pos = bisect.bisect_left(index["timestamps"], dt)
        candidates = []
        if pos < len(index["timestamps"]):
            candidates.append((index["timestamps"][pos], index["files"][pos]))
        if pos > 0:
            candidates.append((index["timestamps"][pos - 1], index["files"][pos - 1]))
        if not candidates:
            return None

        best_ts, best_file = min(candidates, key=lambda x: abs(x[0] - dt))
        if abs(best_ts - dt) <= self.image_tolerance:
            return os.path.join(index["dir"], best_file)
        return None

    def _match_samples(self):
        samples = []
        cache = {}
        for dt in self.df.index:
            history_start = dt - timedelta(
                seconds=self.sampling_rate * (self.sequence_length - 1)
            )
            if history_start not in self.df.index:
                continue

            if not all(
                dt + timedelta(minutes=h) in self.df.index
                for h in self.horizons
            ):
                continue

            image_path = self._closest_image(dt, cache)
            if image_path is None:
                continue

            samples.append((image_path, dt))
        return samples

    def _get_preprocessor(self, year):
        if year not in self._preprocessors:
            self._preprocessors[year] = FolsomPreprocessor(
                year=year,
                image_root=self.image_root,
            )
        return self._preprocessors[year]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, dt = self.samples[idx]

        processor = self._get_preprocessor(dt.year)
        image = processor.process_image_to_tensor(
            image_path,
            month=dt.month,
        ).float()

        sequence_times = [
            dt - timedelta(seconds=i * self.sampling_rate)
            for i in range(self.sequence_length)
        ][::-1]

        weather = (
            self.df[self.feature_cols]
            .reindex(
                sequence_times,
                method="nearest",
                tolerance=pd.Timedelta("10min"),
            )
            .ffill()
            .bfill()
            .fillna(0.0)
        )
        weather_seq = torch.tensor(
            weather.values,
            dtype=torch.float32,
        )

        targets = []
        ghi_cs = []
        for horizon in self.horizons:
            target_dt = dt + timedelta(minutes=horizon)
            row = self.df.loc[target_dt]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            targets.append(float(row["k_index"]))
            ghi_cs.append(float(row["GHI_cs"]))

        return (
            image,
            weather_seq,
            torch.tensor(targets, dtype=torch.float32),
            torch.tensor(ghi_cs, dtype=torch.float32),
        )


def get_data_loaders(config):
    """Build train/validation/test loaders using the 2014/2015 -> 2016 split."""
    dataset = FolsomDataset(config)
    val_split = float(config.get("training", {}).get("val_split", 0.1))

    train_idx, val_idx, test_idx = chronological_year_split(
        dataset.samples,
        val_split=val_split,
        train_years=(2014, 2015),
        test_year=2016,
    )

    batch_size = int(config["data"].get("batch_size", 16))
    num_workers = int(config["data"].get("num_workers", 4))

    kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(config["data"].get("prefetch_factor", 2))

    return (
        DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, **kwargs),
        DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, **kwargs),
        DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False, **kwargs),
    )
