"""Plot actual vs predicted solar azimuth from saved Folsom calibrations.

This analysis script does NOT recalibrate the camera and does NOT modify
the original Folsom source file. It reuses the saved monthly calibration
parameters and the original source functions to collect the calibration
points, then computes:

    actual_azimuth   = solar_azimuth_deg
    predicted_azimuth = az_sign * (phi_img_deg - az_alpha_deg) mod 360

For each year it writes:
    - a CSV containing actual/predicted azimuth and circular error
    - a scatter plot of actual vs predicted azimuth

It also writes a combined yearly summary and comparison plot.

Example:
    PYTHONPATH=. python scripts/plot_azimuth_calibration.py

For a faster test:
    PYTHONPATH=. python scripts/plot_azimuth_calibration.py \
        --years 2016 --max-points-per-month 1000
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_IMAGE_ROOT = "/storage2/CV_Irradiance/datasets/1_Folsom"
DEFAULT_CALIBRATION_DIR = "calibration_results"
DEFAULT_OUTPUT_DIR = "analysis/azimuth_calibration"

MONTH_NAMES = (
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

SOURCE_FILENAME = "Folsom_2014_DAYLIGHT_INDEPENDENT_P_RESUME(1).py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot actual vs predicted Folsom solar azimuth by year."
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2014, 2015, 2016],
        choices=(2014, 2015, 2016),
        help="Years to process.",
    )
    parser.add_argument(
        "--image-root",
        default=DEFAULT_IMAGE_ROOT,
        help="Root directory of the Folsom image dataset.",
    )
    parser.add_argument(
        "--calibration-dir",
        default=DEFAULT_CALIBRATION_DIR,
        help="Directory containing Folsom_*_monthly_calibrations.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV files and plots.",
    )
    parser.add_argument(
        "--max-points-per-month",
        type=int,
        default=None,
        help=(
            "Optional maximum number of selected calibration points per month. "
            "Use a small value for a quick test; omit for all points."
        ),
    )
    return parser.parse_args()


def load_original_source(repo_root: Path):
    source_path = repo_root / "source" / SOURCE_FILENAME

    if not source_path.exists():
        raise FileNotFoundError(
            f"Original Folsom source not found: {source_path}"
        )

    spec = importlib.util.spec_from_file_location(
        "folsom_original_for_azimuth_analysis",
        source_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load original Folsom source: {source_path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def circular_error(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return ((actual - predicted + 180.0) % 360.0) - 180.0


def load_calibration(
    calibration_dir: Path,
    year: int,
) -> dict:
    path = (
        calibration_dir
        / f"Folsom_{year}_monthly_calibrations.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Calibration JSON not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def collect_year_points(
    src,
    year: int,
    image_root: Path,
    calibration_dir: Path,
    max_points_per_month: int | None,
) -> pd.DataFrame:
    calibration = load_calibration(
        calibration_dir,
        year,
    )

    months = calibration.get("months", {})
    all_frames: list[pd.DataFrame] = []

    src.YEAR = year
    src.IMAGE_DIR = str(image_root)

    for month in range(1, 13):
        key = f"{month:02d}"

        if key not in months:
            print(f"[{year}-{key}] Calibration not found; skipping.")
            continue

        params = months[key]
        az_sign = int(params["az_sign"])
        az_alpha = float(params["az_alpha_deg"])

        month_input = image_root / str(year) / key

        print()
        print("=" * 70)
        print(f"AZIMUTH ANALYSIS: {year} {MONTH_NAMES[month - 1]}")
        print("=" * 70)
        print(f"Image folder : {month_input}")
        print(f"az_sign      : {az_sign}")
        print(f"az_alpha     : {az_alpha:.6f}°")

        if not month_input.is_dir():
            print("Input folder does not exist; skipping.")
            continue

        image_index = src.build_month_index(
            month,
            str(month_input),
        )

        if image_index.empty:
            print("No valid timestamped images; skipping.")
            continue

        reference_image = src.read_rgb(
            image_index.iloc[0]["path"]
        )

        cx, cy, radius, _ = src.detect_sky_circle(
            reference_image
        )

        records = src.collect_month_calibration_records(
            image_index,
            cx,
            cy,
            radius,
        )

        if len(records) < src.MIN_SUN_POINTS:
            print(
                f"Too few candidate records: "
                f"{len(records)} < {src.MIN_SUN_POINTS}; skipping."
            )
            continue

        selected = src.select_calibration_points(
            records,
            az_sign,
            az_alpha,
        )

        if len(selected) < src.MIN_SUN_POINTS:
            print(
                f"Too few selected points: "
                f"{len(selected)} < {src.MIN_SUN_POINTS}; skipping."
            )
            continue

        df = pd.DataFrame(selected)

        if max_points_per_month is not None:
            if max_points_per_month <= 0:
                raise ValueError(
                    "--max-points-per-month must be greater than zero."
                )

            if len(df) > max_points_per_month:
                # Deterministic sampling for reproducibility.
                sample = np.linspace(
                    0,
                    len(df) - 1,
                    max_points_per_month,
                    dtype=int,
                )
                df = df.iloc[sample].copy()

        df["actual_azimuth_deg"] = (
            df["solar_azimuth_deg"].astype(float)
        )

        df["predicted_azimuth_deg"] = (
            az_sign
            * (
                df["phi_img_deg"].astype(float)
                - az_alpha
            )
        ) % 360.0

        df["azimuth_error_deg"] = circular_error(
            df["actual_azimuth_deg"].to_numpy(),
            df["predicted_azimuth_deg"].to_numpy(),
        )

        df["year"] = year
        df["month"] = month
        df["month_name"] = MONTH_NAMES[month - 1]
        df["az_sign_used"] = az_sign
        df["az_alpha_deg_used"] = az_alpha

        all_frames.append(df)

        print(
            f"Selected points : {len(df)}"
        )

    if not all_frames:
        return pd.DataFrame()

    return pd.concat(
        all_frames,
        ignore_index=True,
    )


def summarize_year(df: pd.DataFrame, year: int) -> dict:
    error = df["azimuth_error_deg"].to_numpy()

    return {
        "year": year,
        "points": int(len(df)),
        "azimuth_rmse_deg": float(
            np.sqrt(np.mean(error ** 2))
        ),
        "azimuth_mae_deg": float(
            np.mean(np.abs(error))
        ),
        "mean_absolute_error_deg": float(
            np.mean(np.abs(error))
        ),
        "max_absolute_error_deg": float(
            np.max(np.abs(error))
        ),
    }


def plot_year_scatter(
    df: pd.DataFrame,
    year: int,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 7))

    plt.scatter(
        df["actual_azimuth_deg"],
        df["predicted_azimuth_deg"],
        s=5,
        alpha=0.25,
    )

    plt.plot(
        [0, 360],
        [0, 360],
        linewidth=2,
    )

    plt.xlim(0, 360)
    plt.ylim(0, 360)

    plt.xlabel("Actual Solar Azimuth (°)")
    plt.ylabel("Predicted Solar Azimuth (°)")
    plt.title(
        f"Folsom {year}: Actual vs Predicted Solar Azimuth"
    )

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_year_comparison(
    year_data: dict[int, pd.DataFrame],
    output_path: Path,
) -> None:
    years = sorted(year_data.keys())

    if not years:
        return

    fig, axes = plt.subplots(
        1,
        len(years),
        figsize=(7 * len(years), 6),
        squeeze=False,
    )

    axes = axes[0]

    for ax, year in zip(axes, years):
        df = year_data[year]

        ax.scatter(
            df["actual_azimuth_deg"],
            df["predicted_azimuth_deg"],
            s=4,
            alpha=0.20,
        )

        ax.plot(
            [0, 360],
            [0, 360],
            linewidth=2,
        )

        ax.set_xlim(0, 360)
        ax.set_ylim(0, 360)

        ax.set_xlabel("Actual Azimuth (°)")
        ax.set_ylabel("Predicted Azimuth (°)")
        ax.set_title(f"Folsom {year}")

        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Folsom Actual vs Predicted Solar Azimuth by Year"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    image_root = Path(args.image_root)
    calibration_dir = (
        repo_root / args.calibration_dir
        if not Path(args.calibration_dir).is_absolute()
        else Path(args.calibration_dir)
    )
    output_dir = (
        repo_root / args.output_dir
        if not Path(args.output_dir).is_absolute()
        else Path(args.output_dir)
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not image_root.is_dir():
        raise FileNotFoundError(
            f"Folsom image root not found: {image_root}"
        )

    src = load_original_source(repo_root)

    year_data: dict[int, pd.DataFrame] = {}
    summary_rows: list[dict] = []

    print("=" * 70)
    print("FOLSOM ACTUAL vs PREDICTED AZIMUTH ANALYSIS")
    print("=" * 70)
    print(f"Image root      : {image_root}")
    print(f"Calibration dir : {calibration_dir}")
    print(f"Output dir      : {output_dir}")
    print(f"Years           : {args.years}")
    print(
        "Point limit     : "
        + (
            str(args.max_points_per_month)
            if args.max_points_per_month is not None
            else "all selected points"
        )
    )
    print()
    print(
        "This script reads the saved calibration parameters and reuses "
        "the original Folsom functions. It does not recalibrate or modify "
        "the original source."
    )

    for year in args.years:
        df = collect_year_points(
            src=src,
            year=year,
            image_root=image_root,
            calibration_dir=calibration_dir,
            max_points_per_month=args.max_points_per_month,
        )

        if df.empty:
            print(f"No azimuth points collected for {year}.")
            continue

        year_data[year] = df

        csv_path = (
            output_dir
            / f"Folsom_{year}_actual_vs_predicted_azimuth.csv"
        )

        df.to_csv(
            csv_path,
            index=False,
        )

        plot_path = (
            output_dir
            / f"Folsom_{year}_actual_vs_predicted_azimuth.png"
        )

        plot_year_scatter(
            df,
            year,
            plot_path,
        )

        summary = summarize_year(
            df,
            year,
        )
        summary_rows.append(summary)

        print()
        print(
            f"{year} RMSE : "
            f"{summary['azimuth_rmse_deg']:.4f}°"
        )
        print(
            f"{year} MAE  : "
            f"{summary['azimuth_mae_deg']:.4f}°"
        )
        print(f"CSV        : {csv_path}")
        print(f"Plot       : {plot_path}")

    if not summary_rows:
        raise RuntimeError(
            "No yearly azimuth data was generated."
        )

    summary_df = pd.DataFrame(
        summary_rows
    ).sort_values("year")

    summary_csv = (
        output_dir
        / "azimuth_yearly_summary.csv"
    )

    summary_df.to_csv(
        summary_csv,
        index=False,
    )

    comparison_plot = (
        output_dir
        / "actual_vs_predicted_azimuth_by_year.png"
    )

    plot_year_comparison(
        year_data,
        comparison_plot,
    )

    print()
    print("=" * 70)
    print("COMPLETED")
    print("=" * 70)
    print()
    print(summary_df.to_string(index=False))
    print()
    print(f"Yearly summary : {summary_csv}")
    print(f"Comparison plot : {comparison_plot}")


if __name__ == "__main__":
    main()
