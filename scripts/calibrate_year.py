"""Calibrate all 12 months for one Folsom year."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from preprocessing.folsom_adapter import FolsomPreprocessor

IMAGE_ROOT = "/storage2/CV_Irradiance/datasets/1_Folsom"
NAMES = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, choices=(2014, 2015, 2016), required=True)
    parser.add_argument("--image-root", default=IMAGE_ROOT)
    args = parser.parse_args()

    out = Path("calibration_results") / f"Folsom_{args.year}_monthly_calibrations.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists():
        with out.open("r", encoding="utf-8") as f:
            master = json.load(f)
    else:
        master = {
            "dataset": "Folsom",
            "year": args.year,
            "calibration_type": "monthly",
            "source": "calibrated_using_original_folsom_pipeline",
            "months": {},
        }

    months = master.setdefault("months", {})
    processor = FolsomPreprocessor(args.year, args.image_root)

    for month in range(1, 13):
        key = f"{month:02d}"
        if key in months:
            print(f"{key} {NAMES[month-1]} already saved; skipping.")
            continue

        print("=" * 70)
        print(f"CALIBRATING {args.year} {NAMES[month-1]}")
        print("=" * 70)

        c = processor.calibrate_month(month)
        months[key] = {
            "month_number": c.month,
            "month_name": NAMES[month - 1],
            "dataset": "Folsom",
            "year": c.year,
            "month": c.month,
            "cx": c.cx,
            "cy": c.cy,
            "R": c.radius,
            "projection_model": "theta_deg = 90 * (r_norm ** p)",
            "p": c.p,
            "azimuth_model": "predicted_azimuth = az_sign * (image_phi - az_alpha)",
            "az_sign": c.az_sign,
            "az_alpha_deg": c.az_alpha_deg,
            "num_calibration_points": c.num_calibration_points,
            "radial_rmse_deg": c.radial_rmse_deg,
            "azimuth_rmse_deg": c.azimuth_rmse_deg,
            "azimuth_mae_deg": c.azimuth_mae_deg,
            "map_x_shape": list(c.map_x.shape),
            "map_y_shape": list(c.map_y.shape),
        }

        with out.open("w", encoding="utf-8") as f:
            json.dump(master, f, indent=2)

        print(f"Saved {key} to {out}")

    print(f"Completed {args.year}: {len(months)}/12 months")
    print(f"Master JSON: {out}")


if __name__ == "__main__":
    main()
