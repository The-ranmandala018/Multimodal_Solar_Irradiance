"""Plot fast monthly calibration summaries without processing images.

This script reads the saved Folsom yearly calibration JSON files only.
It does not load or process the sky images and does not recalibrate.

Outputs:
    - reference solar azimuth from pvlib for the 15th of each month at 12:00 UTC
    - monthly azimuth calibration offset (az_alpha_deg)
    - monthly azimuth RMSE and MAE
    - monthly projection parameter p
    - CSV summary
    - PNG figures

IMPORTANT:
az_alpha_deg is a calibration offset parameter, NOT the predicted
solar azimuth itself. A true per-image predicted azimuth requires
image_phi_deg from the individual calibration records.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pvlib


DEFAULT_CALIBRATION_DIR = "calibration_results"
DEFAULT_OUTPUT_DIR = "analysis/calibration_summary"

LATITUDE = 38.642
LONGITUDE = -121.148

YEARS = (2014, 2015, 2016)

MONTH_NAMES = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create fast plots from Folsom calibration JSON files."
    )
    parser.add_argument(
        "--calibration-dir",
        default=DEFAULT_CALIBRATION_DIR,
        help="Directory containing Folsom_*_monthly_calibrations.json",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV and PNG outputs.",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=list(YEARS),
        choices=YEARS,
        help="Years to include.",
    )
    return parser.parse_args()


def load_json(calibration_dir: Path, year: int) -> dict:
    path = (
        calibration_dir
        / f"Folsom_{year}_monthly_calibrations.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Calibration file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_dataframe(
    calibration_dir: Path,
    years: list[int],
) -> pd.DataFrame:
    rows = []

    for year in years:
        data = load_json(calibration_dir, year)

        months = data.get("months", {})

        for month in range(1, 13):
            key = f"{month:02d}"

            if key not in months:
                print(
                    f"{year}-{key}: calibration not found; skipping."
                )
                continue

            result = months[key]

            # A fixed monthly reference time is used only so that
            # pvlib can provide an easy year/month solar-position curve.
            timestamp = pd.Timestamp(
                year=year,
                month=month,
                day=15,
                hour=12,
                tz="UTC",
            )

            rows.append(
                {
                    "year": year,
                    "month": month,
                    "month_name": MONTH_NAMES[month - 1],
                    "timestamp": timestamp,
                    "az_alpha_deg": float(
                        result["az_alpha_deg"]
                    ),
                    "azimuth_rmse_deg": float(
                        result["azimuth_rmse_deg"]
                    ),
                    "azimuth_mae_deg": float(
                        result["azimuth_mae_deg"]
                    ),
                    "radial_rmse_deg": float(
                        result["radial_rmse_deg"]
                    ),
                    "p": float(result["p"]),
                    "calibration_points": int(
                        result["num_calibration_points"]
                    ),
                }
            )

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(
            "No calibration records were loaded."
        )

    return df.sort_values(
        ["year", "month"]
    ).reset_index(drop=True)


def add_pvlib_reference_azimuth(df: pd.DataFrame) -> pd.DataFrame:
    solar = pvlib.solarposition.get_solarposition(
        df["timestamp"],
        latitude=LATITUDE,
        longitude=LONGITUDE,
    )

    result = df.copy()
    result["pvlib_reference_actual_azimuth_deg"] = (
        solar["azimuth"].to_numpy()
    )

    return result


def save_plot(
    path: Path,
    title: str,
    ylabel: str,
    df: pd.DataFrame,
    column: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    plt.figure(figsize=(12, 6))

    for year in sorted(df["year"].unique()):
        subset = df[df["year"] == year]

        plt.plot(
            subset["month_name"],
            subset[column],
            marker="o",
            linewidth=2,
            label=str(year),
        )

    plt.xlabel("Month")
    plt.ylabel(ylabel)
    plt.title(title)

    if ylim is not None:
        plt.ylim(*ylim)

    plt.grid(True, alpha=0.3)
    plt.legend(title="Year")
    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    calibration_dir = Path(args.calibration_dir)
    if not calibration_dir.is_absolute():
        calibration_dir = repo_root / calibration_dir

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("FAST FOLSOM CALIBRATION SUMMARY")
    print("=" * 70)
    print(f"Calibration dir : {calibration_dir}")
    print(f"Output dir      : {output_dir}")
    print(f"Years           : {args.years}")
    print()
    print(
        "No sky images are processed. Only the saved JSON calibration "
        "results and pvlib monthly reference positions are used."
    )

    df = build_dataframe(
        calibration_dir,
        args.years,
    )

    df = add_pvlib_reference_azimuth(df)

    # Save data table.
    csv_path = (
        output_dir
        / "folsom_calibration_summary_2014_2016.csv"
    )
    df.to_csv(
        csv_path,
        index=False,
    )

    # Plot 1: reference actual solar azimuth from pvlib.
    save_plot(
        output_dir
        / "reference_actual_solar_azimuth_by_year.png",
        "Folsom Reference Solar Azimuth by Year",
        "Solar Azimuth (°)",
        df,
        "pvlib_reference_actual_azimuth_deg",
        (0, 360),
    )

    # Plot 2: saved azimuth calibration offset.
    save_plot(
        output_dir
        / "azimuth_calibration_alpha_by_year.png",
        "Folsom Azimuth Calibration Offset by Year",
        "Azimuth Calibration Offset, alpha (°)",
        df,
        "az_alpha_deg",
    )

    # Plot 3: azimuth RMSE.
    save_plot(
        output_dir
        / "azimuth_rmse_by_year.png",
        "Folsom Azimuth Calibration RMSE by Year",
        "Azimuth RMSE (°)",
        df,
        "azimuth_rmse_deg",
    )

    # Plot 4: azimuth MAE.
    save_plot(
        output_dir
        / "azimuth_mae_by_year.png",
        "Folsom Azimuth Calibration MAE by Year",
        "Azimuth MAE (°)",
        df,
        "azimuth_mae_deg",
    )

    # Plot 5: projection parameter p.
    save_plot(
        output_dir
        / "projection_parameter_p_by_year.png",
        "Folsom Projection Parameter p by Year",
        "Projection Parameter (p)",
        df,
        "p",
    )

    # Print yearly summary.
    summary = (
        df.groupby("year")
        .agg(
            mean_az_alpha_deg=("az_alpha_deg", "mean"),
            mean_azimuth_rmse_deg=("azimuth_rmse_deg", "mean"),
            mean_azimuth_mae_deg=("azimuth_mae_deg", "mean"),
            mean_radial_rmse_deg=("radial_rmse_deg", "mean"),
            mean_p=("p", "mean"),
            total_calibration_points=("calibration_points", "sum"),
        )
        .reset_index()
    )

    summary_path = (
        output_dir
        / "folsom_yearly_calibration_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print()
    print("=" * 70)
    print("YEARLY SUMMARY")
    print("=" * 70)
    print(summary.to_string(index=False))

    print()
    print(f"CSV summary : {csv_path}")
    print(f"Year summary: {summary_path}")
    print(f"Plots       : {output_dir}")
    print()
    print(
        "Note: pvlib reference azimuth is not the per-image predicted "
        "azimuth. A true Actual-vs-Predicted per-image plot requires "
        "the individual image calibration records containing phi_img_deg."
    )


if __name__ == "__main__":
    main()
