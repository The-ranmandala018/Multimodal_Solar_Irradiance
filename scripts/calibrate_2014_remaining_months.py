"""Calibrate 2014 October-December and update the existing master JSON.

Existing January-September calibrations are preserved unchanged.
October, November and December are calibrated with the original
Folsom source through FolsomPreprocessor, then appended to the
same 2014 calibration JSON.

No per-image 3D .npy files are generated.
"""

from __future__ import annotations

import json
from pathlib import Path

from preprocessing.folsom_adapter import FolsomPreprocessor


ROOT = Path(__file__).resolve().parents[1]

CALIBRATION_FILE = (
    ROOT
    / "calibration_results"
    / "Folsom_2014_monthly_calibrations.json"
)

IMAGE_ROOT = "/storage2/CV_Irradiance/datasets/1_Folsom"

MONTHS_TO_PROCESS = (10, 11, 12)


def main() -> None:
    if not CALIBRATION_FILE.exists():
        raise FileNotFoundError(
            f"Master calibration file not found: {CALIBRATION_FILE}"
        )

    with CALIBRATION_FILE.open("r", encoding="utf-8") as f:
        master = json.load(f)

    if master.get("year") != 2014:
        raise ValueError(
            f"Expected 2014 master calibration file, "
            f"got year={master.get('year')}"
        )

    months = master.setdefault("months", {})

    preprocessor = FolsomPreprocessor(
        year=2014,
        image_root=IMAGE_ROOT,
    )

    for month in MONTHS_TO_PROCESS:
        key = f"{month:02d}"

        print()
        print("=" * 70)
        print(f"FOLSOM 2014 CALIBRATION - MONTH {month:02d}")
        print("=" * 70)

        # Never overwrite an existing month.
        if key in months:
            print(
                f"Month {month:02d} already exists in the master JSON. "
                "Skipping."
            )
            continue

        calibration = preprocessor.calibrate_month(month)

        months[key] = {
            "month_number": month,
            "month_name": calibration_month_name(month),
            "source": "newly_calibrated_using_original_folsom_pipeline",
            "dataset": "Folsom",
            "year": 2014,
            "month": month,
            "latitude": 38.642,
            "longitude": -121.148,
            "timestamp_timezone": "UTC",
            "cx": calibration.cx,
            "cy": calibration.cy,
            "R": calibration.radius,
            "projection_model": "theta_deg = 90 * (r_norm ** p)",
            "p": calibration.p,
            "azimuth_model": (
                "predicted_azimuth = "
                "az_sign * (image_phi - az_alpha)"
            ),
            "az_sign": calibration.az_sign,
            "az_alpha_deg": calibration.az_alpha_deg,
            "num_calibration_points": calibration.num_calibration_points,
            "radial_rmse_deg": calibration.radial_rmse_deg,
            "azimuth_rmse_deg": calibration.azimuth_rmse_deg,
            "azimuth_mae_deg": calibration.azimuth_mae_deg,
            "map_x_shape": list(calibration.map_x.shape),
            "map_y_shape": list(calibration.map_y.shape),
        }

        # Save immediately after each successful month.
        # Therefore, if a later month fails, earlier completed
        # months remain saved.
        with CALIBRATION_FILE.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                master,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(
            f"Month {month:02d} added to: {CALIBRATION_FILE}"
        )

    print()
    print("=" * 70)
    print("2014 CALIBRATION UPDATE COMPLETED")
    print("=" * 70)

    for key in sorted(months):
        print(
            f"{key}: "
            f"{months[key].get('month_name', 'Unknown')}"
        )

    print(f"\nMaster JSON: {CALIBRATION_FILE}")


def calibration_month_name(month: int) -> str:
    names = (
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    )
    return names[month - 1]


if __name__ == "__main__":
    main()
