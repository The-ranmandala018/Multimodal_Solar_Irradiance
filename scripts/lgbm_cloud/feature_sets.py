"""Feature-set definitions for the LightGBM cloud-feature experiment.

A0        past k, temperature, pressure, sun geometry and time at t (the LSTM's 40 x 7 window as columns)
A         A0 + SZA(t+10) and GHI_cs(t+10) (fixed geometry, no measured data after t)
B         A + cloud features (v1) from the image at or before t + has_image
C         cloud features + sun geometry + time only (no k, temperature or pressure) + has_image
B_parity  B, but cloud features from the image the CNN used (file-name time, +-120 s)

Each set reads one table in features/ (TABLE_OF).
"""

from cloud_features import FEATURE_GROUPS

K_COLS = [f"k_lag{l}" for l in (0, 1, 2, 5, 10, 20, 39)] + [
    f"k_{s}{w}" for w in (10, 20, 40) for s in ("mean", "std")] + ["k_diff10"]
MET_COLS = ["temp_t", "pres_t", "temp_d30", "pres_d30"]
GEOM_T_COLS = ["sza_t", "az_sin", "az_cos", "sin_hour", "cos_hour"]
GEOM_T10_COLS = ["sza_t10", "ghi_cs_t10"]

CLOUD_GROUPS = {g: [c for c in cols if c != "mo_dt_s"] for g, cols in FEATURE_GROUPS.items()}
CLOUD_COLS = [c for cols in CLOUD_GROUPS.values() for c in cols] + ["has_image"]

SET_A0 = K_COLS + MET_COLS + GEOM_T_COLS
SET_A = SET_A0 + GEOM_T10_COLS
SET_B = SET_A + CLOUD_COLS
SET_C = GEOM_T_COLS + GEOM_T10_COLS + CLOUD_COLS

FEATURE_SETS = {"A0": SET_A0, "A": SET_A, "B": SET_B, "C": SET_C, "B_parity": SET_B}
TABLE_OF = {"A0": "samples_setA.parquet", "A": "samples_setA.parquet", "B": "samples_B.parquet",
            "C": "samples_B.parquet", "B_parity": "samples_B_parity.parquet"}

# Groups for grouped SHAP / importance (Step 7)
GROUP_OF = {**{c: "k_history" for c in K_COLS}, **{c: "meteo" for c in MET_COLS},
            **{c: "geometry_time" for c in GEOM_T_COLS + GEOM_T10_COLS}, "has_image": "image_flag",
            **{c: f"cloud_{g}" for g, cols in CLOUD_GROUPS.items() for c in cols}}


def get_feature_set(name: str) -> list[str]:
    if name not in FEATURE_SETS:
        raise KeyError(f"feature set {name!r} unknown; have {sorted(FEATURE_SETS)}")
    return list(FEATURE_SETS[name])
