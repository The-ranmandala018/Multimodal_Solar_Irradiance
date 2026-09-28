#!/usr/bin/env python3
"""
Folsom January 2014 - Optimized 3D Vector Batch Processing

ONLY 2014 JANUARY is processed.

Workflow:
    Folsom fisheye image
        -> sky-circle detection
        -> sun detection
        -> solar position
        -> optimized camera calibration
        -> actual vs predicted azimuth
        -> azimuth/zenith remapping
        -> 3D vector generation
        -> validation and summaries

The script directly searches the 2014/01 directory. It does NOT
scan the entire Folsom dataset first.
"""

import os
import re
import json
import glob
import time
import traceback
import shutil

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.optimize import least_squares
from pvlib import solarposition


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_DIR = "/storage2/CV_Irradiance/datasets/1_Folsom"

OUTPUT_DIR = (
    "/datasets/CV_Irradiance/cv_phase2/"
    "Folsom_January_optimized_3d_vectors"
)

# Secondary output location. If the primary /datasets filesystem
# gets close to full, new day outputs are written here instead.
SECONDARY_OUTPUT_DIR = (
    "/storage/CV_Irradiance/phase_2/"
    "Folsom_January_optimized_3d_vectors"
)

# Keep this much free space on the primary filesystem as a safety margin.
MIN_PRIMARY_FREE_GB = 50.0

DATASET_NAME = "Folsom"

LATITUDE = 38.642
LONGITUDE = -121.148

# Verify this against the Folsom dataset documentation.
TIMESTAMP_TIMEZONE = "UTC"

# ONLY THIS YEAR AND MONTH
YEAR = 2014
MONTH = 1

OVERWRITE = False

MAX_SOLAR_ZENITH_DEG = 85.0

MIN_SUN_POINTS = 15

AZ_WIDTH = 720
ZEN_HEIGHT = 180

DEFAULT_P = 1.0

# ============================================================
# OPTIMIZED CALIBRATION PARAMETERS
# ============================================================
# Obtained from the separate P optimization, azimuth optimization,
# and azimuth fine-tuning steps.
OPTIMIZED_P = 0.4
OPTIMIZED_AZ_SIGN = 1
OPTIMIZED_AZ_ALPHA_DEG = 96.7

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".JPG", ".JPEG", ".PNG"
}

# Sun candidate settings
CANDIDATE_PERCENTILES = (
    99.7,
    99.3,
    99.0,
    98.5,
)

MIN_COMPONENT_AREA = 5
MAX_COMPONENT_AREA = 10000
MAX_CANDIDATES_PER_IMAGE = 20

# Candidate must agree reasonably with the fitted azimuth.
MAX_CALIBRATION_AZ_ERROR_DEG = 35.0

# ============================================================
# OUTPUT
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)
os.makedirs(
    SECONDARY_OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# FILE / TIMESTAMP HELPERS
# ============================================================

def list_all_images(root_dir):
    files = []

    if not os.path.isdir(root_dir):
        return files

    for root, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if os.path.splitext(filename)[1] in IMAGE_EXTENSIONS:
                files.append(
                    os.path.join(root, filename)
                )

    return sorted(files)


def read_rgb(path):
    image_bgr = cv2.imread(
        path,
        cv2.IMREAD_COLOR
    )

    if image_bgr is None:
        raise ValueError(
            f"Could not read image: {path}"
        )

    return cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )


def parse_timestamp_from_filename(
    path,
    timezone="UTC"
):
    """
    Correctly parse Folsom filenames.

    Examples:
        20140101_000011.jpg
        2014-01-01_00-00-11.jpg
        2014-01-01_00-00.jpg
    """

    filename = os.path.basename(path)

    # --------------------------------------------------------
    # 20140101_000011.jpg
    # --------------------------------------------------------

    match = re.search(
        r"(?<!\d)"
        r"(20\d{2})(0[1-9]|1[0-2])"
        r"(0[1-9]|[12]\d|3[01])"
        r"[_T-]"
        r"([01]\d|2[0-3])"
        r"([0-5]\d)"
        r"([0-5]\d)"
        r"(?!\d)",
        filename
    )

    if match:
        timestamp_string = (
            f"{match.group(1)}-"
            f"{match.group(2)}-"
            f"{match.group(3)} "
            f"{match.group(4)}:"
            f"{match.group(5)}:"
            f"{match.group(6)}"
        )

        return pd.to_datetime(
            timestamp_string,
            format="%Y-%m-%d %H:%M:%S"
        ).tz_localize(timezone)

    # --------------------------------------------------------
    # 2014-01-01_00-00-11.jpg
    # --------------------------------------------------------

    match = re.search(
        r"(20\d{2})[-_]"
        r"(0[1-9]|1[0-2])[-_]"
        r"(0[1-9]|[12]\d|3[01])"
        r"[T _-]"
        r"([01]\d|2[0-3])"
        r"[-_:]"
        r"([0-5]\d)"
        r"[-_:]"
        r"([0-5]\d)",
        filename
    )

    if match:
        timestamp_string = (
            f"{match.group(1)}-"
            f"{match.group(2)}-"
            f"{match.group(3)} "
            f"{match.group(4)}:"
            f"{match.group(5)}:"
            f"{match.group(6)}"
        )

        return pd.to_datetime(
            timestamp_string,
            format="%Y-%m-%d %H:%M:%S"
        ).tz_localize(timezone)

    # --------------------------------------------------------
    # 2014-01-01_00-00.jpg
    # --------------------------------------------------------

    match = re.search(
        r"(20\d{2})[-_]"
        r"(0[1-9]|1[0-2])[-_]"
        r"(0[1-9]|[12]\d|3[01])"
        r"[T _-]"
        r"([01]\d|2[0-3])"
        r"[-_:]"
        r"([0-5]\d)",
        filename
    )

    if match:
        timestamp_string = (
            f"{match.group(1)}-"
            f"{match.group(2)}-"
            f"{match.group(3)} "
            f"{match.group(4)}:"
            f"{match.group(5)}"
        )

        return pd.to_datetime(
            timestamp_string,
            format="%Y-%m-%d %H:%M"
        ).tz_localize(timezone)

    return None


def circular_error_deg(actual, predicted):
    return (
        (
            np.asarray(actual)
            - np.asarray(predicted)
            + 180.0
        ) % 360.0
    ) - 180.0


def circular_rmse_deg(actual, predicted):
    error = circular_error_deg(
        actual,
        predicted
    )

    return float(
        np.sqrt(
            np.mean(error ** 2)
        )
    )


def circular_mae_deg(actual, predicted):
    error = circular_error_deg(
        actual,
        predicted
    )

    return float(
        np.mean(np.abs(error))
    )


# ============================================================
# SKY CIRCLE
# ============================================================

def detect_sky_circle(image_rgb):
    gray = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        (9, 9),
        0
    )

    _, mask = cv2.threshold(
        blurred,
        12,
        255,
        cv2.THRESH_BINARY
    )

    kernel = np.ones(
        (15, 15),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise ValueError(
            "No sky contour found."
        )

    contour = max(
        contours,
        key=cv2.contourArea
    )

    (cx, cy), radius = (
        cv2.minEnclosingCircle(contour)
    )

    if radius <= 0:
        raise ValueError(
            "Invalid sky-circle radius."
        )

    return (
        float(cx),
        float(cy),
        float(radius),
        mask
    )


# ============================================================
# SUN DETECTION
# ============================================================

def image_phi_deg(
    x,
    y,
    cx,
    cy
):
    return (
        np.degrees(
            np.arctan2(
                -(y - cy),
                x - cx
            )
        ) % 360.0
    )


def detect_sun_candidates(
    image_rgb,
    cx,
    cy,
    radius
):
    """
    Return several bright candidates rather than selecting
    one candidate blindly.
    """

    height, width, _ = image_rgb.shape

    yy, xx = np.indices(
        (height, width)
    )

    distance = np.sqrt(
        (xx - cx) ** 2
        + (yy - cy) ** 2
    )

    sky_mask = (
        distance <=
        0.97 * radius
    )

    image = image_rgb.astype(
        np.float32
    )

    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]

    gray = (
        0.299 * red
        + 0.587 * green
        + 0.114 * blue
    )

    sky_values = gray[sky_mask]

    if len(sky_values) < 1000:
        return []

    rgb_min = np.minimum.reduce(
        [red, green, blue]
    )

    brightness_floor = np.percentile(
        sky_values,
        90
    )

    candidates = {}

    for percentile in CANDIDATE_PERCENTILES:

        threshold = np.percentile(
            sky_values,
            percentile
        )

        bright = (
            (gray >= threshold)
            & sky_mask
            & (rgb_min >= brightness_floor)
        )

        bright_uint8 = (
            bright.astype(
                np.uint8
            )
            * 255
        )

        kernel = np.ones(
            (5, 5),
            np.uint8
        )

        bright_uint8 = cv2.morphologyEx(
            bright_uint8,
            cv2.MORPH_OPEN,
            kernel
        )

        bright_uint8 = cv2.morphologyEx(
            bright_uint8,
            cv2.MORPH_CLOSE,
            kernel
        )

        contours, _ = cv2.findContours(
            bright_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            area = cv2.contourArea(
                contour
            )

            if (
                area < MIN_COMPONENT_AREA
                or
                area > MAX_COMPONENT_AREA
            ):
                continue

            moments = cv2.moments(
                contour
            )

            if moments["m00"] == 0:
                continue

            x = (
                moments["m10"]
                /
                moments["m00"]
            )

            y = (
                moments["m01"]
                /
                moments["m00"]
            )

            radial_distance = np.sqrt(
                (x - cx) ** 2
                + (y - cy) ** 2
            )

            if (
                radial_distance
                >
                0.97 * radius
            ):
                continue

            x0 = max(
                0,
                int(x - 8)
            )

            x1 = min(
                width,
                int(x + 9)
            )

            y0 = max(
                0,
                int(y - 8)
            )

            y1 = min(
                height,
                int(y + 9)
            )

            local_gray = gray[
                y0:y1,
                x0:x1
            ]

            if local_gray.size == 0:
                continue

            local_brightness = float(
                np.mean(local_gray)
            )

            local_max = float(
                np.max(local_gray)
            )

            score = (
                local_brightness
                * np.sqrt(
                    max(area, 1.0)
                )
                + 0.25
                * local_max
            )

            key = (
                round(x, 1),
                round(y, 1)
            )

            candidates[key] = {
                "x": float(x),
                "y": float(y),
                "area": float(area),
                "brightness":
                    local_brightness,
                "score":
                    float(score),
                "r_norm":
                    float(
                        radial_distance
                        / radius
                    ),
                "phi_img_deg":
                    float(
                        image_phi_deg(
                            x,
                            y,
                            cx,
                            cy
                        )
                    ),
            }

    result = sorted(
        candidates.values(),
        key=lambda item:
            item["score"],
        reverse=True
    )

    return result[
        :MAX_CANDIDATES_PER_IMAGE
    ]


# ============================================================
# SOLAR POSITION
# ============================================================

def solar_position_for_time(
    timestamp
):
    result = solarposition.get_solarposition(
        time=pd.DatetimeIndex(
            [timestamp]
        ),
        latitude=LATITUDE,
        longitude=LONGITUDE
    )

    zenith = float(
        result[
            "apparent_zenith"
        ].iloc[0]
    )

    azimuth = float(
        result[
            "azimuth"
        ].iloc[0]
    )

    return (
        zenith,
        azimuth
    )


# ============================================================
# AZIMUTH CALIBRATION
# ============================================================

def fit_azimuth_orientation(
    calibration_records
):
    """
    Fit:

        predicted_azimuth =
            sign * (image_phi - alpha) mod 360

    The best candidate from each image is selected for each
    possible orientation. This reduces the effect of bright
    cloud false detections.
    """

    if len(calibration_records) < MIN_SUN_POINTS:
        return None

    best = None

    for sign in [1, -1]:

        for alpha in np.arange(
            0.0,
            360.0,
            1.0
        ):

            errors = []

            for record in calibration_records:

                candidates = record[
                    "candidates"
                ]

                if not candidates:
                    continue

                true_az = record[
                    "solar_azimuth_deg"
                ]

                candidate_phis = np.array(
                    [
                        c["phi_img_deg"]
                        for c in candidates
                    ],
                    dtype=float
                )

                predicted = (
                    sign
                    * (
                        candidate_phis
                        - alpha
                    )
                ) % 360.0

                error = np.abs(
                    circular_error_deg(
                        predicted,
                        true_az
                    )
                )

                errors.append(
                    float(
                        np.min(error)
                    )
                )

            if len(errors) < MIN_SUN_POINTS:
                continue

            errors = np.asarray(
                errors
            )

            inliers = (
                errors
                <= MAX_CALIBRATION_AZ_ERROR_DEG
            )

            num_inliers = int(
                np.sum(inliers)
            )

            if num_inliers < MIN_SUN_POINTS:
                continue

            median_error = float(
                np.median(
                    errors[inliers]
                )
            )

            rmse = float(
                np.sqrt(
                    np.mean(
                        errors[inliers] ** 2
                    )
                )
            )

            candidate_result = {
                "sign": sign,
                "alpha": float(alpha),
                "num_inliers":
                    num_inliers,
                "median_error":
                    median_error,
                "rmse":
                    rmse,
            }

            if (
                best is None
                or
                num_inliers
                >
                best["num_inliers"]
                or
                (
                    num_inliers
                    ==
                    best["num_inliers"]
                    and
                    median_error
                    <
                    best["median_error"]
                )
            ):
                best = candidate_result

    if best is None:
        return None

    # Refine alpha using circular mean of the selected points.
    sign = best["sign"]
    alpha0 = best["alpha"]

    selected_differences = []

    for record in calibration_records:

        candidates = record[
            "candidates"
        ]

        if not candidates:
            continue

        true_az = record[
            "solar_azimuth_deg"
        ]

        phis = np.array(
            [
                c["phi_img_deg"]
                for c in candidates
            ],
            dtype=float
        )

        predicted = (
            sign
            * (
                phis
                - alpha0
            )
        ) % 360.0

        errors = np.abs(
            circular_error_deg(
                predicted,
                true_az
            )
        )

        idx = int(
            np.argmin(errors)
        )

        if (
            errors[idx]
            <= MAX_CALIBRATION_AZ_ERROR_DEG
        ):
            selected_differences.append(
                float(
                    (
                        phis[idx]
                        - sign * true_az
                    ) % 360.0
                )
            )

    if len(selected_differences) < MIN_SUN_POINTS:
        return None

    alpha = float(
        np.degrees(
            np.arctan2(
                np.mean(
                    np.sin(
                        np.radians(
                            selected_differences
                        )
                    )
                ),
                np.mean(
                    np.cos(
                        np.radians(
                            selected_differences
                        )
                    )
                )
            )
        ) % 360.0
    )

    return (
        int(sign),
        alpha
    )


def select_calibration_points(
    calibration_records,
    az_sign,
    az_alpha
):
    selected = []

    for record in calibration_records:

        candidates = record[
            "candidates"
        ]

        if not candidates:
            continue

        true_az = record[
            "solar_azimuth_deg"
        ]

        errors = []

        for candidate in candidates:

            predicted = (
                az_sign
                * (
                    candidate[
                        "phi_img_deg"
                    ]
                    - az_alpha
                )
            ) % 360.0

            errors.append(
                abs(
                    circular_error_deg(
                        predicted,
                        true_az
                    )
                )
            )

        idx = int(
            np.argmin(errors)
        )

        if (
            errors[idx]
            > MAX_CALIBRATION_AZ_ERROR_DEG
        ):
            continue

        candidate = candidates[idx]

        row = {
            "year":
                record["year"],
            "month":
                record["month"],
            "day":
                record["day"],
            "filename":
                record["filename"],
            "path":
                record["path"],
            "timestamp":
                record["timestamp"],
            "sun_x":
                candidate["x"],
            "sun_y":
                candidate["y"],
            "r_pix":
                candidate["r_norm"],
            "r_norm":
                candidate["r_norm"],
            "phi_img_deg":
                candidate[
                    "phi_img_deg"
                ],
            "solar_zenith_deg":
                record[
                    "solar_zenith_deg"
                ],
            "solar_azimuth_deg":
                record[
                    "solar_azimuth_deg"
                ],
            "candidate_score":
                candidate["score"],
            "azimuth_error_deg":
                float(errors[idx]),
        }

        selected.append(row)

    return selected


# ============================================================
# ZENITH CALIBRATION
# ============================================================

def fit_power_projection(
    dataframe
):
    if len(dataframe) == 0:
        return DEFAULT_P

    r = dataframe[
        "r_norm"
    ].values.astype(float)

    theta = dataframe[
        "solar_zenith_deg"
    ].values.astype(float)

    valid = (
        (r > 0.02)
        & (r < 0.98)
        & (theta > 1)
        & (theta < 89)
    )

    r = r[valid]
    theta = theta[valid]

    if len(r) < MIN_SUN_POINTS:
        return DEFAULT_P

    def residual(params):

        p = params[0]

        predicted = (
            90.0
            * (
                r ** p
            )
        )

        return (
            predicted
            - theta
        )

    result = least_squares(
        residual,
        x0=np.array([1.0]),
        bounds=(
            [0.4],
            [2.5]
        )
    )

    return float(
        result.x[0]
    )


def calculate_calibration_quality(
    dataframe,
    p,
    az_sign,
    az_alpha
):
    result = {
        "num_points":
            int(len(dataframe)),
        "radial_rmse_deg":
            None,
        "azimuth_rmse_deg":
            None,
        "azimuth_mae_deg":
            None,
    }

    if len(dataframe) == 0:
        return result

    r = dataframe[
        "r_norm"
    ].values.astype(float)

    true_zenith = dataframe[
        "solar_zenith_deg"
    ].values.astype(float)

    valid = (
        (r > 0.02)
        & (r < 0.98)
        & (true_zenith > 1)
        & (true_zenith < 89)
    )

    if np.any(valid):

        predicted_zenith = (
            90.0
            * (
                r[valid] ** p
            )
        )

        radial_error = (
            predicted_zenith
            - true_zenith[valid]
        )

        result[
            "radial_rmse_deg"
        ] = float(
            np.sqrt(
                np.mean(
                    radial_error ** 2
                )
            )
        )

    image_phi = dataframe[
        "phi_img_deg"
    ].values.astype(float)

    true_azimuth = dataframe[
        "solar_azimuth_deg"
    ].values.astype(float)

    predicted_azimuth = (
        az_sign
        * (
            image_phi
            - az_alpha
        )
    ) % 360.0

    az_error = circular_error_deg(
        predicted_azimuth,
        true_azimuth
    )

    result[
        "azimuth_rmse_deg"
    ] = float(
        np.sqrt(
            np.mean(
                az_error ** 2
            )
        )
    )

    result[
        "azimuth_mae_deg"
    ] = float(
        np.mean(
            np.abs(az_error)
        )
    )

    return result


# ============================================================
# REMAPPING
# ============================================================

def create_zenith_azimuth_maps(
    cx,
    cy,
    radius,
    p,
    az_sign,
    az_alpha_deg,
    out_w=720,
    out_h=180
):
    azimuth = np.linspace(
        0,
        360,
        out_w,
        endpoint=False
    )

    zenith = np.linspace(
        0,
        90,
        out_h
    )

    AZ, ZEN = np.meshgrid(
        azimuth,
        zenith
    )

    normalized_radius = (
        ZEN / 90.0
    ) ** (
        1.0 / p
    )

    radius_pixels = (
        normalized_radius
        * radius
    )

    image_phi = (
        az_sign
        * AZ
        + az_alpha_deg
    ) % 360.0

    phi_rad = np.radians(
        image_phi
    )

    map_x = (
        cx
        + radius_pixels
        * np.cos(phi_rad)
    )

    map_y = (
        cy
        - radius_pixels
        * np.sin(phi_rad)
    )

    return (
        map_x.astype(np.float32),
        map_y.astype(np.float32)
    )


def remap_to_azimuth_zenith(
    image_rgb,
    map_x,
    map_y
):
    return cv2.remap(
        image_rgb,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )


# ============================================================
# 3D VECTOR GENERATION
# ============================================================

FEATURE_NAMES = [
    "x",
    "y",
    "z",
    "theta_rad",
    "phi_rad",
    "R",
    "G",
    "B",
    "intensity",
    "solid_angle_weight",
]

EXPECTED_SHAPE = (
    ZEN_HEIGHT * AZ_WIDTH,
    len(FEATURE_NAMES)
)


def azzen_to_helical_3d_vectors(
    azzen_image_rgb
):
    height, width, _ = (
        azzen_image_rgb.shape
    )

    row, col = np.indices(
        (height, width)
    )

    theta = (
        np.pi
        * row
        /
        (
            2.0
            * (
                height - 1
            )
        )
    )

    phi = (
        2.0
        * np.pi
        * col
        /
        width
    )

    x = (
        np.sin(theta)
        * np.cos(phi)
    )

    y = (
        np.sin(theta)
        * np.sin(phi)
    )

    z = np.cos(theta)

    coordinates = np.stack(
        [
            x,
            y,
            z
        ],
        axis=-1
    )

    rgb = (
        azzen_image_rgb.astype(
            np.float32
        )
        / 255.0
    )

    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]

    intensity = (
        0.299 * red
        + 0.587 * green
        + 0.114 * blue
    )

    solid_angle_weight = np.maximum(
        np.sin(theta),
        1e-8
    )

    feature_image = np.concatenate(
        [
            coordinates,
            theta[..., None],
            phi[..., None],
            rgb,
            intensity[..., None],
            solid_angle_weight[..., None],
        ],
        axis=-1
    )

    return feature_image.reshape(
        -1,
        feature_image.shape[-1]
    ).astype(
        np.float32
    )


# ============================================================
# OUTPUT STORAGE SELECTION
# ============================================================

def _free_gb(path):
    """Return free space (GB) for the filesystem containing path."""
    target = path
    while not os.path.exists(target):
        parent = os.path.dirname(target)
        if parent == target:
            break
        target = parent
    usage = shutil.disk_usage(target)
    return usage.free / (1024 ** 3)


def choose_day_output_dir(year, month, day):
    """
    Keep an interrupted day in the same storage location.

    Otherwise use /datasets while it has the configured safety margin;
    once it gets too full, continue new days under /storage/CV_Irradiance/phase_2.
    """
    relative_day = os.path.join(
        str(year),
        f"{month:02d}",
        f"{day:02d}"
    )

    primary_day = os.path.join(
        OUTPUT_DIR,
        relative_day
    )
    secondary_day = os.path.join(
        SECONDARY_OUTPUT_DIR,
        relative_day
    )

    # If this day was already partially processed, resume from the
    # location where its files already exist.
    primary_has_files = os.path.isdir(primary_day) and any(
        os.scandir(primary_day)
    )
    secondary_has_files = os.path.isdir(secondary_day) and any(
        os.scandir(secondary_day)
    )

    if primary_has_files and not secondary_has_files:
        return primary_day
    if secondary_has_files:
        return secondary_day

    primary_free = _free_gb(OUTPUT_DIR)
    if primary_free >= MIN_PRIMARY_FREE_GB:
        return primary_day

    print(
        f"Primary storage has only {primary_free:.1f} GB free; "
        f"using secondary storage: {secondary_day}"
    )
    return secondary_day


# ============================================================
# PROCESS ONE DAY
# ============================================================

def process_one_day(
    year,
    month,
    day,
    image_files,
    output_dir,
    map_x,
    map_y
):
    os.makedirs(
        output_dir,
        exist_ok=True
    )

    saved = 0
    skipped = 0
    failed = 0

    index_records = []
    failure_records = []

    start = time.time()

    for image_path in tqdm(
        sorted(image_files),
        desc=f"{year}-{month:02d}-{day:02d}"
    ):

        base = os.path.splitext(
            os.path.basename(image_path)
        )[0]

        vector_filename = (
            base
            + "_helical_3d_vectors.npy"
        )

        vector_path = os.path.join(
            output_dir,
            vector_filename
        )

        if (
            os.path.exists(vector_path)
            and not OVERWRITE
        ):
            skipped += 1

            index_records.append(
                {
                    "source_image":
                        os.path.basename(
                            image_path
                        ),
                    "vector_file":
                        vector_filename,
                    "status":
                        "existing_skipped",
                }
            )

            continue

        try:

            image = read_rgb(
                image_path
            )

            azzen = (
                remap_to_azimuth_zenith(
                    image,
                    map_x,
                    map_y
                )
            )

            vectors = (
                azzen_to_helical_3d_vectors(
                    azzen
                )
            )

            if vectors.shape != EXPECTED_SHAPE:
                raise ValueError(
                    f"Unexpected vector shape: "
                    f"{vectors.shape}"
                )

            if not np.all(
                np.isfinite(vectors)
            ):
                raise ValueError(
                    "Vector contains NaN or Inf."
                )

            # Keep the same 10-feature vector representation and shape,
            # but store it as float16 to reduce disk usage by about half.
            vectors_to_save = vectors.astype(
                np.float16,
                copy=False
            )

            np.save(
                vector_path,
                vectors_to_save
            )

            saved += 1

            index_records.append(
                {
                    "source_image":
                        os.path.basename(
                            image_path
                        ),
                    "vector_file":
                        vector_filename,
                    "shape":
                        str(vectors.shape),
                    "dtype":
                        str(vectors.dtype),
                    "status":
                        "saved",
                }
            )

        except Exception as error:

            failed += 1

            failure_records.append(
                {
                    "source_image":
                        os.path.basename(
                            image_path
                        ),
                    "path":
                        image_path,
                    "error":
                        repr(error),
                    "traceback":
                        traceback.format_exc(),
                }
            )

    pd.DataFrame(
        index_records
    ).to_csv(
        os.path.join(
            output_dir,
            "vector_index.csv"
        ),
        index=False
    )

    pd.DataFrame(
        failure_records
    ).to_csv(
        os.path.join(
            output_dir,
            "vector_failures.csv"
        ),
        index=False
    )

    total = len(image_files)

    successful = (
        saved + skipped
    )

    success_rate = (
        100.0
        * successful
        / total
        if total
        else 0.0
    )

    summary = {
        "year": year,
        "month": month,
        "day": day,
        "images_found": total,
        "saved": saved,
        "existing_skipped": skipped,
        "failed": failed,
        "success_rate_percent":
            success_rate,
        "processing_seconds":
            time.time() - start,
    }

    with open(
        os.path.join(
            output_dir,
            "day_summary.json"
        ),
        "w"
    ) as f:
        json.dump(
            summary,
            f,
            indent=4
        )

    return summary


# ============================================================
# MAIN
# ============================================================


# ============================================================
# FULL-YEAR MONTHLY CONTROL FLOW
# ============================================================

# These are the validated January values from the original January
# script. January uses them directly; they are NOT re-fitted.
JANUARY_P = OPTIMIZED_P
JANUARY_AZ_SIGN = OPTIMIZED_AZ_SIGN
JANUARY_AZ_ALPHA = OPTIMIZED_AZ_ALPHA_DEG

# Only the monthly control/optimization is added here.
# The image processing, candidate detection, solar-position,
# calibration-quality, remapping and vector-generation functions
# above are unchanged from the original January script.

MONTHS = range(1, 13)

# For February -> December, search around the previous month's result.
# Independent P search for EVERY month.
# The previous month's P is NOT used.
P_MIN = 0.01
P_MAX = 2.00
P_STEP = 0.01

AZ_HALF_RANGE = 10.0
AZ_STEP = 0.1

# Optional metadata only. The original solar_position_for_time()
# intentionally remains unchanged, so this value does NOT alter the
# January calculation/results.
ALTITUDE_M = None
RUN_VERSION = "daylight_independent_p_v1"


def optimize_p_independently(sun_df):
    """
    Independently search P over the same fixed range for every month.

    Model:
        theta = 90 * r_norm**p

    No previous-month P is used.
    """

    r = sun_df["r_norm"].values.astype(float)
    true_zenith = (
        sun_df["solar_zenith_deg"].values.astype(float)
    )

    valid = (
        (r > 0.02)
        & (r < 0.98)
        & (true_zenith > 1)
        & (true_zenith < 89)
    )

    r = r[valid]
    true_zenith = true_zenith[valid]

    if len(r) < MIN_SUN_POINTS:
        raise RuntimeError(
            "Too few valid points for p optimization."
        )

    p_values = np.arange(
        P_MIN,
        P_MAX + P_STEP / 2.0,
        P_STEP
    )

    results = []

    for p in p_values:

        predicted = (
            90.0
            * (
                r ** p
            )
        )

        error = predicted - true_zenith

        results.append(
            {
                "p": float(p),
                "radial_mae_deg":
                    float(np.mean(np.abs(error))),
                "radial_rmse_deg":
                    float(np.sqrt(np.mean(error ** 2))),
                "num_points": int(len(r)),
            }
        )

    result_df = pd.DataFrame(results)

    best = (
        result_df
        .sort_values("radial_rmse_deg")
        .iloc[0]
    )

    return float(best["p"]), result_df


def optimize_azimuth_from_previous(
    sun_df,
    previous_sign,
    previous_alpha
):
    """
    Same azimuth model as the original January code:
        predicted = sign * (phi_img - alpha)

    Search around the previous month's alpha:
        +/- 10 degrees, step 0.1 degree

    Both signs are tested, exactly as in the monthly source script.
    """

    phi = sun_df["phi_img_deg"].values.astype(float)
    actual = (
        sun_df["solar_azimuth_deg"].values.astype(float)
    )

    alpha_values = np.arange(
        previous_alpha - AZ_HALF_RANGE,
        previous_alpha + AZ_HALF_RANGE
        + AZ_STEP / 2.0,
        AZ_STEP
    )

    results = []

    for sign in [1, -1]:

        for alpha in alpha_values:

            predicted = (
                sign
                * (
                    phi - alpha
                )
            ) % 360.0

            error = circular_error_deg(
                predicted,
                actual
            )

            results.append(
                {
                    "previous_az_sign": int(previous_sign),
                    "previous_az_alpha_deg": float(previous_alpha),
                    "az_sign": int(sign),
                    "az_alpha_deg": float(alpha),
                    "azimuth_mae_deg":
                        float(np.mean(np.abs(error))),
                    "azimuth_rmse_deg":
                        float(np.sqrt(np.mean(error ** 2))),
                    "num_points": int(len(phi)),
                }
            )

    result_df = pd.DataFrame(results)

    best = (
        result_df
        .sort_values(
            ["azimuth_rmse_deg", "azimuth_mae_deg"]
        )
        .iloc[0]
    )

    return (
        int(best["az_sign"]),
        float(best["az_alpha_deg"]),
        result_df
    )


def build_month_index(month, month_input):
    """Same timestamp parsing logic as the January script."""

    image_files = list_all_images(month_input)

    records = []

    for path in tqdm(
        image_files,
        desc="Parsing timestamps"
    ):

        timestamp = parse_timestamp_from_filename(
            path,
            TIMESTAMP_TIMEZONE
        )

        if timestamp is None:
            continue

        if (
            timestamp.year != YEAR
            or timestamp.month != month
        ):
            continue

        # ----------------------------------------------------
        # DAYLIGHT FILTER
        # Use the same PVLIB solar-position calculation already
        # used by the original script. Images at/after sunset
        # and before the next sunrise are excluded.
        # ----------------------------------------------------
        try:
            solar_zenith, _ = solar_position_for_time(timestamp)
        except Exception:
            continue

        solar_elevation = 90.0 - solar_zenith

        if solar_elevation <= 0.0:
            continue

        records.append(
            {
                "path": path,
                "filename": os.path.basename(path),
                "year": int(timestamp.year),
                "month": int(timestamp.month),
                "day": int(timestamp.day),
                "timestamp": timestamp,
            }
        )

    image_index_df = pd.DataFrame(records)

    if image_index_df.empty:
        return image_index_df

    return (
        image_index_df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def collect_month_calibration_records(
    image_index_df,
    cx,
    cy,
    radius,
    checkpoint_path=None
):
    """Same calibration-candidate collection logic as January, with resume checkpointing."""

    calibration_records = []
    processed_paths = set()

    if checkpoint_path:
        checkpoint_data = _load_json(checkpoint_path)
        if (
            isinstance(checkpoint_data, dict)
            and checkpoint_data.get("run_version") == RUN_VERSION
        ):
            calibration_records = checkpoint_data.get(
                "calibration_records", []
            )
            processed_paths = set(
                checkpoint_data.get("processed_paths", [])
            )

    pending_items = [
        item
        for _, item in image_index_df.iterrows()
        if item["path"] not in processed_paths
    ]

    for count, item in enumerate(
        tqdm(
            pending_items,
            total=len(pending_items),
            desc="Sun calibration"
        ),
        start=1
    ):

        timestamp = item["timestamp"]
        processed_paths.add(item["path"])

        try:
            true_zenith, true_azimuth = (
                solar_position_for_time(timestamp)
            )
        except Exception:
            if checkpoint_path:
                _save_calibration_checkpoint(
                    checkpoint_path,
                    {
                        "run_version": RUN_VERSION,
                        "calibration_records": calibration_records,
                        "processed_paths": sorted(processed_paths),
                    }
                )
            continue

        if true_zenith > MAX_SOLAR_ZENITH_DEG:
            if checkpoint_path:
                _save_calibration_checkpoint(
                    checkpoint_path,
                    {
                        "run_version": RUN_VERSION,
                        "calibration_records": calibration_records,
                        "processed_paths": sorted(processed_paths),
                    }
                )
            continue

        try:
            image = read_rgb(item["path"])

            candidates = detect_sun_candidates(
                image,
                cx,
                cy,
                radius
            )
        except Exception:
            if checkpoint_path:
                _save_calibration_checkpoint(
                    checkpoint_path,
                    {
                        "run_version": RUN_VERSION,
                        "calibration_records": calibration_records,
                        "processed_paths": sorted(processed_paths),
                    }
                )
            continue

        if candidates:
            calibration_records.append(
                {
                    "year": int(item["year"]),
                    "month": int(item["month"]),
                    "day": int(item["day"]),
                    "filename": item["filename"],
                    "path": item["path"],
                    "timestamp": str(timestamp),
                    "solar_zenith_deg": true_zenith,
                    "solar_azimuth_deg": true_azimuth,
                    "candidates": candidates,
                }
            )

        # Save a checkpoint every image. This makes interruption/resume
        # reliable; the atomic write prevents a half-written checkpoint.
        if checkpoint_path:
            _save_calibration_checkpoint(
                checkpoint_path,
                {
                    "calibration_records": calibration_records,
                    "processed_paths": sorted(processed_paths),
                }
            )

    return calibration_records


def _atomic_json_save(path, data):
    """Save a JSON checkpoint atomically so an interruption cannot leave a partial file."""
    temp_path = path + ".tmp"
    with open(temp_path, "w") as f:
        json.dump(data, f, indent=4)
    os.replace(temp_path, path)


def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def _save_calibration_checkpoint(path, records):
    _atomic_json_save(path, records)


def _load_calibration_checkpoint(path):
    data = _load_json(path)
    if not isinstance(data, list):
        return None
    return data


def process_month(month, start_p, start_sign, start_alpha):

    month_name = pd.Timestamp(
        YEAR,
        month,
        1
    ).strftime("%B")

    month_input = os.path.join(
        IMAGE_DIR,
        str(YEAR),
        f"{month:02d}"
    )

    month_output = os.path.join(
        OUTPUT_DIR,
        f"{month:02d}_{month_name}"
    )

    os.makedirs(month_output, exist_ok=True)

    # --------------------------------------------------------
    # RESUME: if this exact new workflow already completed this
    # month successfully, load its result and continue to the next month.
    # --------------------------------------------------------
    completed_summary_path = os.path.join(
        month_output,
        "monthly_calibration_summary.json"
    )

    if os.path.exists(completed_summary_path):
        try:
            completed_result = _load_json(completed_summary_path)
            if (
                isinstance(completed_result, dict)
                and completed_result.get("run_version") == RUN_VERSION
                and int(completed_result.get("failed", 1)) == 0
                and float(completed_result.get("success_rate_percent", 0.0)) >= 100.0
            ):
                print(f"\n{month_name.upper()} already completed. Resuming from next month.")
                return completed_result
        except Exception:
            pass

    print("\n" + "=" * 70)
    print(f"{month_name.upper()} {YEAR}")
    print("=" * 70)
    print("Input :", month_input)
    print("Output:", month_output)
    print("\nStarting parameters:")
    print(f"  p        = {start_p:.6f}")
    print(f"  az_sign  = {start_sign}")
    print(f"  az_alpha = {start_alpha:.6f}°")

    if not os.path.isdir(month_input):
        print("Input folder does not exist:", month_input)
        return None

    image_index_df = build_month_index(
        month,
        month_input
    )

    if image_index_df.empty:
        print("No valid timestamped images.")
        return None

    image_index_df.to_csv(
        os.path.join(
            month_output,
            "image_index.csv"
        ),
        index=False
    )

    print(
        "Images selected:",
        len(image_index_df)
    )

    # --------------------------------------------------------
    # Reference image and sky circle: EXACT January functions.
    # --------------------------------------------------------

    reference_path = image_index_df.iloc[0]["path"]
    reference_image = read_rgb(reference_path)

    cx, cy, radius, _ = detect_sky_circle(
        reference_image
    )

    print(
        f"Sky circle: cx={cx:.3f}, "
        f"cy={cy:.3f}, R={radius:.3f}"
    )

    # --------------------------------------------------------
    # Solar candidates: EXACT January logic.
    # --------------------------------------------------------

    calibration_checkpoint_path = os.path.join(
        month_output,
        "calibration_records_checkpoint.json"
    )

    calibration_records = collect_month_calibration_records(
        image_index_df,
        cx,
        cy,
        radius,
        calibration_checkpoint_path
    )

    print(
        "Images with candidates:",
        len(calibration_records)
    )

    if len(calibration_records) < MIN_SUN_POINTS:
        raise RuntimeError(
            "Too few sun calibration images. "
            f"Found {len(calibration_records)}, "
            f"need at least {MIN_SUN_POINTS}."
        )

    # --------------------------------------------------------
    # Starting candidate selection.
    # January keeps its validated azimuth parameters. For later
    # months, the existing previous-month azimuth logic is kept.
    # P is independently optimized over 0.01 -> 2.00 for EVERY month.
    # --------------------------------------------------------

    selected_start = select_calibration_points(
        calibration_records,
        start_sign,
        start_alpha
    )

    if len(selected_start) < MIN_SUN_POINTS:
        raise RuntimeError(
            "Too few valid calibration points using "
            "starting azimuth parameters. "
            f"Found {len(selected_start)}."
        )

    sun_df_start = pd.DataFrame(
        selected_start
    )

    sun_df_start.to_csv(
        os.path.join(
            month_output,
            "sun_calibration_points_starting_selection.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Independent P optimization.
    # --------------------------------------------------------

    p_fit, p_results = optimize_p_independently(
        sun_df_start
    )

    p_results.to_csv(
        os.path.join(
            month_output,
            "p_optimization_results.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Existing azimuth optimization logic.
    # January uses its validated azimuth; other months keep
    # the original previous-month azimuth search.
    # --------------------------------------------------------

    if month == 1:
        az_sign = JANUARY_AZ_SIGN
        az_alpha = JANUARY_AZ_ALPHA

        print("\nJANUARY: using validated azimuth parameters")
        print("Azimuth sign:", az_sign)
        print("Azimuth offset:", f"{az_alpha:.6f} degrees")
    else:
        az_sign, az_alpha, az_results = (
            optimize_azimuth_from_previous(
                sun_df_start,
                start_sign,
                start_alpha
            )
        )

        az_results.to_csv(
            os.path.join(
                month_output,
                "azimuth_optimization_results.csv"
            ),
            index=False
        )

    print("\nMonthly optimized parameters:")
    print("Projection p:", f"{p_fit:.6f}")
    print("Azimuth sign:", az_sign)
    print("Azimuth offset:", f"{az_alpha:.6f} degrees")

    # --------------------------------------------------------
    # Final selection with the chosen calibration.
    # EXACT January function.
    # --------------------------------------------------------

    selected_records = select_calibration_points(
        calibration_records,
        az_sign,
        az_alpha
    )

    if len(selected_records) < MIN_SUN_POINTS:
        raise RuntimeError(
            "Too few valid calibration points after azimuth "
            "selection. "
            f"Found {len(selected_records)}."
        )

    sun_df = pd.DataFrame(
        selected_records
    )

    sun_df.to_csv(
        os.path.join(
            month_output,
            "sun_calibration_points.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # EXACT calibration-quality function.
    # --------------------------------------------------------

    quality = calculate_calibration_quality(
        sun_df,
        p_fit,
        az_sign,
        az_alpha
    )

    print("\nCalibration points:", len(sun_df))
    print(f"Projection p: {p_fit:.6f}")
    print("Radial RMSE:", quality["radial_rmse_deg"])
    print("Azimuth RMSE:", quality["azimuth_rmse_deg"])
    print("Azimuth MAE:", quality["azimuth_mae_deg"])

    # --------------------------------------------------------
    # Save monthly calibration.
    # --------------------------------------------------------

    calibration = {
        "dataset": DATASET_NAME,
        "year": YEAR,
        "month": month,
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "altitude_m": ALTITUDE_M,
        "timestamp_timezone": TIMESTAMP_TIMEZONE,
        "run_version": RUN_VERSION,
        "p_search_min": P_MIN,
        "p_search_max": P_MAX,
        "p_search_step": P_STEP,
        "cx": cx,
        "cy": cy,
        "R": radius,
        "projection_model":
            "theta_deg = 90 * (r_norm ** p)",
        "p": float(p_fit),
        "azimuth_model":
            "predicted_azimuth = "
            "az_sign * (image_phi - az_alpha)",
        "az_sign": int(az_sign),
        "az_alpha_deg": float(az_alpha),
        "num_calibration_points": int(len(sun_df)),
        "radial_rmse_deg": quality["radial_rmse_deg"],
        "azimuth_rmse_deg": quality["azimuth_rmse_deg"],
        "azimuth_mae_deg": quality["azimuth_mae_deg"],
        "starting_p": None,
        "starting_az_sign": int(start_sign),
        "starting_az_alpha_deg": float(start_alpha),
    }

    with open(
        os.path.join(
            month_output,
            "monthly_calibration.json"
        ),
        "w"
    ) as f:
        json.dump(
            calibration,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # EXACT January remapping function.
    # --------------------------------------------------------

    map_x, map_y = create_zenith_azimuth_maps(
        cx,
        cy,
        radius,
        p_fit,
        az_sign,
        az_alpha,
        AZ_WIDTH,
        ZEN_HEIGHT
    )

    # --------------------------------------------------------
    # Group by day and use EXACT process_one_day().
    # --------------------------------------------------------

    grouped = {}

    for _, row in image_index_df.iterrows():

        key = (
            int(row["year"]),
            int(row["month"]),
            int(row["day"])
        )

        grouped.setdefault(
            key,
            []
        ).append(
            row["path"]
        )

    all_summaries = []

    day_checkpoint_path = os.path.join(
        month_output,
        "day_processing_checkpoint.json"
    )

    completed_days = set()
    day_checkpoint = _load_json(day_checkpoint_path)
    if isinstance(day_checkpoint, dict):
        completed_days = set(
            day_checkpoint.get("completed_days", [])
        )

    for (
        year_value,
        month_value,
        day_value
    ) in sorted(grouped):

        day_key = f"{year_value:04d}-{month_value:02d}-{day_value:02d}"

        day_output = choose_day_output_dir(
            year_value,
            month_value,
            day_value
        )

        if day_key in completed_days:
            day_summary_path = os.path.join(
                day_output,
                "day_summary.json"
            )
            day_summary = _load_json(day_summary_path)
            if isinstance(day_summary, dict):
                all_summaries.append(day_summary)
                continue

        summary = process_one_day(
            year_value,
            month_value,
            day_value,
            grouped[
                (
                    year_value,
                    month_value,
                    day_value
                )
            ],
            day_output,
            map_x,
            map_y
        )

        all_summaries.append(summary)

        if (
            int(summary.get("failed", 0)) == 0
            and int(summary.get("saved", 0))
            + int(summary.get("existing_skipped", 0))
            == int(summary.get("images_found", 0))
        ):
            completed_days.add(day_key)
        _atomic_json_save(
            day_checkpoint_path,
            {"completed_days": sorted(completed_days)}
        )

    summary_df = pd.DataFrame(
        all_summaries
    )

    summary_df.to_csv(
        os.path.join(
            month_output,
            "processing_summary.csv"
        ),
        index=False
    )

    total_images = int(
        summary_df["images_found"].sum()
    ) if len(summary_df) else 0

    total_saved = int(
        summary_df["saved"].sum()
    ) if len(summary_df) else 0

    total_skipped = int(
        summary_df["existing_skipped"].sum()
    ) if len(summary_df) else 0

    total_failed = int(
        summary_df["failed"].sum()
    ) if len(summary_df) else 0

    successful = (
        total_saved
        + total_skipped
    )

    success_rate = (
        100.0 * successful / total_images
        if total_images
        else 0.0
    )

    result = {
        "dataset": DATASET_NAME,
        "year": YEAR,
        "month": month,
        "run_version": RUN_VERSION,
        "month_name": month_name,
        "images_found": total_images,
        "saved": total_saved,
        "skipped": total_skipped,
        "failed": total_failed,
        "success_rate_percent": success_rate,
        "starting_p": None,
        "starting_az_sign": int(start_sign),
        "starting_az_alpha_deg": float(start_alpha),
        "optimized_p": float(p_fit),
        "optimized_az_sign": int(az_sign),
        "optimized_az_alpha_deg": float(az_alpha),
        "calibration_points": int(len(sun_df)),
        "radial_rmse_deg": quality["radial_rmse_deg"],
        "azimuth_mae_deg": quality["azimuth_mae_deg"],
        "azimuth_rmse_deg": quality["azimuth_rmse_deg"],
        "altitude_m": ALTITUDE_M,
        "processing_complete": True,
    }

    with open(
        os.path.join(
            month_output,
            "monthly_calibration_summary.json"
        ),
        "w"
    ) as f:
        json.dump(
            result,
            f,
            indent=4
        )

    print("\n" + "=" * 70)
    print(f"{month_name.upper()} COMPLETED")
    print("=" * 70)
    print(f"Starting p        : {start_p:.6f}")
    print(f"Optimized p       : {p_fit:.6f}")
    print(f"Starting az alpha : {start_alpha:.6f}°")
    print(f"Optimized az alpha: {az_alpha:.6f}°")
    print(f"Azimuth MAE       : {quality['azimuth_mae_deg']:.6f}°")
    print(f"Azimuth RMSE      : {quality['azimuth_rmse_deg']:.6f}°")
    print(f"Vectors saved     : {total_saved}")
    print(f"Failed            : {total_failed}")
    print(f"Success rate      : {success_rate:.2f}%")

    return result


def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("=" * 70)
    print(
        "FOLSOM 2014 - FULL YEAR MONTHLY PROCESSING"
    )
    print("=" * 70)
    print("Input:", IMAGE_DIR)
    print("Primary output:", OUTPUT_DIR)
    print("Secondary output:", SECONDARY_OUTPUT_DIR)
    print(
        "Primary free-space safety threshold:",
        f"{MIN_PRIMARY_FREE_GB:.1f} GB"
    )
    print("Year:", YEAR)
    print("Lat/Lon:", LATITUDE, LONGITUDE)
    print("Timestamp TZ:", TIMESTAMP_TIMEZONE)
    print("Altitude metadata:", ALTITUDE_M)
    print("January p:", JANUARY_P)
    print("January az_sign:", JANUARY_AZ_SIGN)
    print(
        "January az_alpha:",
        f"{JANUARY_AZ_ALPHA:.1f} degrees"
    )
    print("=" * 70)

    if not os.path.isdir(IMAGE_DIR):
        raise RuntimeError(
            f"Input directory not found: {IMAGE_DIR}"
        )

    # P is independently optimized for every month.
    # Only azimuth parameters continue to use the existing
    # previous-month propagation logic.
    previous_p = None
    previous_sign = JANUARY_AZ_SIGN
    previous_alpha = JANUARY_AZ_ALPHA

    all_results = []

    for month in MONTHS:

        if month == 1:
            start_p = JANUARY_P
            start_sign = JANUARY_AZ_SIGN
            start_alpha = JANUARY_AZ_ALPHA
        else:
            start_p = DEFAULT_P
            start_sign = previous_sign
            start_alpha = previous_alpha

        try:

            result = process_month(
                month,
                start_p,
                start_sign,
                start_alpha
            )

            if result is None:
                continue

            all_results.append(result)

            # Carry the optimized values to the next month.
            # optimized_p is intentionally NOT carried to the next month.
            previous_sign = result[
                "optimized_az_sign"
            ]

            previous_alpha = result[
                "optimized_az_alpha_deg"
            ]

        except Exception as error:

            print(
                f"\nERROR in month {month}: {error}"
            )

            traceback.print_exc()

            # Keep the last successful parameters for the next month.
            continue

    if all_results:

        summary_df = pd.DataFrame(
            all_results
        )

        summary_path = os.path.join(
            OUTPUT_DIR,
            "Folsom_2014_monthly_summary.csv"
        )

        summary_df.to_csv(
            summary_path,
            index=False
        )

        print("\n" + "=" * 70)
        print(
            "FOLSOM 2014 MONTHLY PROCESSING COMPLETED"
        )
        print("=" * 70)
        print(
            summary_df.to_string(
                index=False
            )
        )
        print("\nSummary:", summary_path)


if __name__ == "__main__":
    main()
