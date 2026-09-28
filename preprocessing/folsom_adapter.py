"""In-memory adapter around the original Folsom preprocessing source.

Important:
- The original source file remains unchanged and is the source of truth.
- Folsom images are read directly from the server dataset.
- No generated 3D .npy files are written by this adapter.
- Verified 2014 January-September calibrations are reused from the
  repository calibration file instead of being recalculated.
- Months without a saved calibration are calibrated in memory using
  the original source functions.
- Years 2014, 2015 and 2016 are supported through the year argument.
"""

from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SOURCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "Folsom_2014_DAYLIGHT_INDEPENDENT_P_RESUME(1).py"
)

DEFAULT_IMAGE_ROOT = "/storage2/CV_Irradiance/datasets/1_Folsom"
SUPPORTED_YEARS = (2014, 2015, 2016)

CALIBRATION_RESULTS_DIR = (
    Path(__file__).resolve().parents[1] / "calibration_results"
)

CALIBRATION_2014_FILE = (
    CALIBRATION_RESULTS_DIR / "Folsom_2014_monthly_calibrations.json"
)


def _load_original_module():
    spec = importlib.util.spec_from_file_location(
        "folsom_original_source",
        SOURCE_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load original Folsom source: {SOURCE_PATH}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class Calibration:
    year: int
    month: int
    cx: float
    cy: float
    radius: float
    p: float
    az_sign: int
    az_alpha_deg: float
    num_calibration_points: int | None
    radial_rmse_deg: float | None
    azimuth_rmse_deg: float | None
    azimuth_mae_deg: float | None
    map_x: np.ndarray
    map_y: np.ndarray


class FolsomPreprocessor:
    """Run the original Folsom calibration/projection pipeline in memory."""

    def __init__(
        self,
        year: int,
        image_root: str = DEFAULT_IMAGE_ROOT,
    ):
        if year not in SUPPORTED_YEARS:
            raise ValueError(
                f"Supported years are {SUPPORTED_YEARS}; got {year}"
            )

        if not os.path.isdir(image_root):
            raise FileNotFoundError(
                f"Folsom dataset root not found: {image_root}"
            )

        self.year = int(year)
        self.image_root = image_root
        self.source = _load_original_module()

        # The source functions read YEAR and IMAGE_DIR as module globals.
        self.source.YEAR = self.year
        self.source.IMAGE_DIR = self.image_root

        self._previous_sign = int(
            self.source.JANUARY_AZ_SIGN
        )
        self._previous_alpha = float(
            self.source.JANUARY_AZ_ALPHA
        )

        self._calibration_cache: dict[int, Calibration] = {}
        self._saved_calibrations = self._load_saved_calibrations()

    def _load_saved_calibrations(self) -> dict[int, dict]:
        """Load verified reusable calibrations for the current year.

        At present, only the verified 2014 January-September calibration
        file is included. Other years intentionally remain independent
        and will use their own calibration files when added.
        """
        if self.year != 2014:
            return {}

        if not CALIBRATION_2014_FILE.exists():
            return {}

        with CALIBRATION_2014_FILE.open(
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        months = data.get("months", {})

        return {
            int(month): calibration
            for month, calibration in months.items()
        }

    def _calibration_from_saved(
        self,
        month: int,
    ) -> Calibration | None:
        """Build a Calibration object from a saved monthly result."""
        data = self._saved_calibrations.get(month)

        if data is None:
            return None

        required = (
            "cx",
            "cy",
            "R",
            "p",
            "az_sign",
            "az_alpha_deg",
        )

        missing = [
            key for key in required
            if key not in data
        ]

        if missing:
            raise ValueError(
                f"Saved calibration for {self.year}/{month:02d} "
                f"is missing fields: {missing}"
            )

        map_x, map_y = self.source.create_zenith_azimuth_maps(
            float(data["cx"]),
            float(data["cy"]),
            float(data["R"]),
            float(data["p"]),
            int(data["az_sign"]),
            float(data["az_alpha_deg"]),
            self.source.AZ_WIDTH,
            self.source.ZEN_HEIGHT,
        )

        return Calibration(
            year=self.year,
            month=month,
            cx=float(data["cx"]),
            cy=float(data["cy"]),
            radius=float(data["R"]),
            p=float(data["p"]),
            az_sign=int(data["az_sign"]),
            az_alpha_deg=float(data["az_alpha_deg"]),
            num_calibration_points=(
                int(data["num_calibration_points"])
                if data.get("num_calibration_points") is not None
                else None
            ),
            radial_rmse_deg=(
                float(data["radial_rmse_deg"])
                if data.get("radial_rmse_deg") is not None
                else None
            ),
            azimuth_rmse_deg=(
                float(data["azimuth_rmse_deg"])
                if data.get("azimuth_rmse_deg") is not None
                else None
            ),
            azimuth_mae_deg=(
                float(data["azimuth_mae_deg"])
                if data.get("azimuth_mae_deg") is not None
                else None
            ),
            map_x=map_x,
            map_y=map_y,
        )

    def _saved_previous_azimuth(
        self,
        month: int,
    ) -> tuple[int, float]:
        """Return the saved previous-month azimuth when available."""
        previous = self._saved_calibrations.get(month - 1)

        if previous is not None:
            return (
                int(previous["az_sign"]),
                float(previous["az_alpha_deg"]),
            )

        return (
            self._previous_sign,
            self._previous_alpha,
        )

    def image_path(
        self,
        month: int,
        day: int | None = None,
    ) -> str:
        path = os.path.join(
            self.image_root,
            str(self.year),
            f"{month:02d}",
        )

        if day is not None:
            path = os.path.join(
                path,
                f"{day:02d}",
            )

        return path

    def calibrate_month(self, month: int) -> Calibration:
        """Return a monthly calibration, reusing verified results when available."""
        if not 1 <= month <= 12:
            raise ValueError(
                "month must be between 1 and 12"
            )

        if month in self._calibration_cache:
            return self._calibration_cache[month]

        # --------------------------------------------------------
        # Reuse verified saved calibration.
        # --------------------------------------------------------
        saved_calibration = self._calibration_from_saved(month)

        if saved_calibration is not None:
            self._calibration_cache[month] = saved_calibration
            self._previous_sign = saved_calibration.az_sign
            self._previous_alpha = saved_calibration.az_alpha_deg

            return saved_calibration

        # --------------------------------------------------------
        # No saved calibration: use the original calibration pipeline.
        # --------------------------------------------------------
        src = self.source
        month_input = self.image_path(month)

        image_index_df = src.build_month_index(
            month,
            month_input,
        )

        if image_index_df.empty:
            raise RuntimeError(
                f"No valid daylight images for "
                f"{self.year}/{month:02d}"
            )

        reference_image = src.read_rgb(
            image_index_df.iloc[0]["path"]
        )

        cx, cy, radius, _ = src.detect_sky_circle(
            reference_image
        )

        records = src.collect_month_calibration_records(
            image_index_df,
            cx,
            cy,
            radius,
            checkpoint_path=None,
        )

        if len(records) < src.MIN_SUN_POINTS:
            raise RuntimeError(
                f"Too few calibration images for "
                f"{self.year}/{month:02d}: "
                f"{len(records)} < {src.MIN_SUN_POINTS}"
            )

        # Preserve the original monthly strategy.
        if month == 1:
            start_sign = int(
                src.JANUARY_AZ_SIGN
            )
            start_alpha = float(
                src.JANUARY_AZ_ALPHA
            )
        else:
            start_sign, start_alpha = (
                self._saved_previous_azimuth(month)
            )

        selected_start = src.select_calibration_points(
            records,
            start_sign,
            start_alpha,
        )

        if len(selected_start) < src.MIN_SUN_POINTS:
            raise RuntimeError(
                f"Too few starting calibration points for "
                f"{self.year}/{month:02d}: "
                f"{len(selected_start)} < {src.MIN_SUN_POINTS}"
            )

        import pandas as pd

        sun_df_start = pd.DataFrame(
            selected_start
        )

        p_fit, _ = src.optimize_p_independently(
            sun_df_start
        )

        if month == 1:
            az_sign = int(
                src.JANUARY_AZ_SIGN
            )
            az_alpha = float(
                src.JANUARY_AZ_ALPHA
            )
        else:
            az_sign, az_alpha, _ = (
                src.optimize_azimuth_from_previous(
                    sun_df_start,
                    start_sign,
                    start_alpha,
                )
            )

        selected_records = src.select_calibration_points(
            records,
            az_sign,
            az_alpha,
        )

        if len(selected_records) < src.MIN_SUN_POINTS:
            raise RuntimeError(
                f"Too few final calibration points for "
                f"{self.year}/{month:02d}: "
                f"{len(selected_records)} < {src.MIN_SUN_POINTS}"
            )

        sun_df = pd.DataFrame(
            selected_records
        )

        quality = src.calculate_calibration_quality(
            sun_df,
            p_fit,
            az_sign,
            az_alpha,
        )

        map_x, map_y = src.create_zenith_azimuth_maps(
            cx,
            cy,
            radius,
            p_fit,
            az_sign,
            az_alpha,
            src.AZ_WIDTH,
            src.ZEN_HEIGHT,
        )

        calibration = Calibration(
            year=self.year,
            month=month,
            cx=float(cx),
            cy=float(cy),
            radius=float(radius),
            p=float(p_fit),
            az_sign=int(az_sign),
            az_alpha_deg=float(az_alpha),
            num_calibration_points=int(len(sun_df)),
            radial_rmse_deg=float(quality["radial_rmse_deg"]),
            azimuth_rmse_deg=float(quality["azimuth_rmse_deg"]),
            azimuth_mae_deg=float(quality["azimuth_mae_deg"]),
            map_x=map_x,
            map_y=map_y,
        )

        self._calibration_cache[month] = calibration
        self._previous_sign = calibration.az_sign
        self._previous_alpha = calibration.az_alpha_deg

        return calibration

    def process_image(
        self,
        image_path: str,
        month: int | None = None,
    ) -> np.ndarray:
        """Return the original 3D representation as (129600, 10), without saving it."""
        src = self.source

        if month is None:
            timestamp = (
                src.parse_timestamp_from_filename(
                    image_path,
                    src.TIMESTAMP_TIMEZONE,
                )
            )

            if timestamp is None:
                raise ValueError(
                    f"Could not parse timestamp from {image_path}"
                )

            month = int(timestamp.month)

        calibration = self.calibrate_month(
            int(month)
        )

        image = src.read_rgb(
            image_path
        )

        azzen = src.remap_to_azimuth_zenith(
            image,
            calibration.map_x,
            calibration.map_y,
        )

        vectors = src.azzen_to_helical_3d_vectors(
            azzen
        )

        expected = tuple(
            src.EXPECTED_SHAPE
        )

        if vectors.shape != expected:
            raise ValueError(
                f"Unexpected vector shape: "
                f"{vectors.shape}; expected {expected}"
            )

        if not np.all(
            np.isfinite(vectors)
        ):
            raise ValueError(
                "3D feature representation contains NaN or Inf"
            )

        return vectors

    def process_image_to_tensor(
        self,
        image_path: str,
        month: int | None = None,
    ):
        from .projection import (
            vectors_to_projection_tensor,
        )

        return vectors_to_projection_tensor(
            self.process_image(
                image_path,
                month=month,
            )
        )


def process_folsom_image(
    image_path: str,
    year: int,
    image_root: str = DEFAULT_IMAGE_ROOT,
    month: int | None = None,
):
    """Convenience function returning one image as (129600, 10) in memory."""
    return FolsomPreprocessor(
        year=year,
        image_root=image_root,
    ).process_image(
        image_path,
        month=month,
    )
