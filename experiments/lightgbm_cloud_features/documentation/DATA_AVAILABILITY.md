# Data availability, licence and regeneration of derived data

## 1. Source dataset and licence (verified 2026-10-17)
- **Dataset:** "A comprehensive dataset for the accelerated development and benchmarking of solar forecasting methods". Carreira Pedro, H., Larson, D., Coimbra, C. (UC San Diego), version V1, published 2019-06-24. Zenodo, **doi:10.5281/zenodo.2826939**.
- **Licence:** **Creative Commons Attribution 4.0 International (CC BY 4.0)**, per the licence field of the Zenodo record (https://zenodo.org/records/2826939). It permits redistribution and reuse provided the creators are credited.
- **The record also states** that use is "intended for research and development purposes only and implies explicit reference to the paper":
  - Pedro, H.T.C., Larson, D.P., Coimbra, C.F.M. (2019), *A comprehensive dataset for the accelerated development and benchmarking of solar forecasting methods*, Journal of Renewable and Sustainable Energy 11, 036102, **doi:10.1063/1.5094494**.
- **Attribution required in any publication or redistributed figure:** cite both DOIs above. Images or figures derived from the sky images must carry a credit line, for example "Sky images: Pedro, Larson & Coimbra (2019), CC BY 4.0, doi:10.5281/zenodo.2826939", and must state that they were modified (downsampled, annotated).
- **Status of the sky-image figures** (`checks/gate2/*.png`, `checks/sky_mask_preview.png`): **held out of the repository** until the authors approve. Including them would need the credit line above, for example in a `checks/gate2/ATTRIBUTION.md`, plus confirmation that the "research and development purposes" statement is compatible with the intended publication.

## 2. What this repository contains, and what it does not
| Item | In repository | How to obtain otherwise |
|---|---|---|
| Raw irradiance CSV and sky images | No | Zenodo doi:10.5281/zenodo.2826939 |
| Code to rebuild everything | Yes (`scripts/lgbm_cloud/`, plus existing `datasets/`, `preprocessing/`, `source/`) | – |
| Fitted parameters (geometry, masks, clear-sky library, thresholds) | Yes (`features/*.json`, `*.npy`, `*.npz`, about 0.5 MB) | rebuilt by the `fit_*.py` scripts |
| Derived feature tables (`image_index`, `samples_setA`, `windows_40x7`, `cloud_features`, `samples_B`, `samples_B_parity`; 1.6 GB) | **No** (deferred) | regenerate (§3); candidate for an external archive (§4) |
| Trained models (25 LightGBM `model.txt`, 2 + 1 Ridge `model.pkl` with pickle-free `model_params.json`) | Yes, pending approval | – |
| Predictions (validation and test, per seed) | Yes (`publication/`, verbatim) | – |
| All reported tables | Yes (`summary/`) | – |

## 3. Regenerating the derived tables from the raw dataset
**Prerequisites:**
- **Image modification times must be preserved.** The image clock correction uses each file's `mtime`; the file names are on a drifting camera clock.
  - Extract the Zenodo image archives with GNU tar's default behaviour, which restores the stored mtimes. **Do not** use `-m` / `--touch`.
  - Copy with `cp -p`, `rsync -t` or an equivalent.
  - Verify by comparing the first rows of `features/image_index_summary.json` (offsets by month) with the published file.
- **Folder layout:** `<root>/YYYY/MM/DD/YYYYMMDD_HHMMSS.jpg`, plus `<root>/csv_files/Folsom_irradiance_weather.csv` (merged irradiance and weather CSV used by the project), plus `<root>/corrupted_images.txt` (optional list of unreadable images).
- **Paths** are currently hard-coded to the original server; see §5.

**Commands** (from `scripts/lgbm_cloud/`, with `PY = env PYTHONNOUSERSITE=1 PYTHONPATH=<repo> python`). Runtimes are as measured on the original server (`COMPUTATIONAL_RESOURCES.md`):

| Step | Command | Produces | Time measured |
|---|---|---|---|
| 1 | `$PY build_image_index.py` | `features/image_index.parquet` | 30 s |
| 2 | `$PY build_tabular_features.py --workers 40` | `features/samples_setA.parquet`, `windows_40x7.npy` | 29 min |
| 3 | (optional; the published fitted parameters can be reused) `$PY fit_sky_geometry.py`, `fit_sky_mask.py`, `fit_clear_sky_library.py` | geometry, masks, library, thresholds | not timed |
| 4 | `env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 $PY extract_cloud_features.py --workers 48` | `features/cloud_features.parquet` (checkpointed and resumable) | 25 min (48 workers) |
| 5 | `$PY verify_cloud_features.py`; `$PY join_features.py` | `samples_B.parquet`, `samples_B_parity.parquet` | about 1 min |

**Checking the regenerated tables:**
- The extraction refuses to run unless the feature-defining files match `FROZEN_V1_manifest.json`.
- The published sha256 of every derived table is in `documentation/ARTIFACT_INVENTORY.csv`. A bit-identical rebuild is expected with the same library versions and the same image mtimes. With different versions or mtimes, compare `verify_cloud_features.py` output and the metrics instead.

## 4. External archive (deferred)
For a journal submission, deposit the derived tables (about 1.6 GB, compressible) together with the code release in a DOI-issuing archive, such as Zenodo with a GitHub release integration. The derived data inherit the source licence (CC BY 4.0) and attribution requirements. Record the archive DOI and the git commit or tag in the data availability statement. Not done yet.

## 5. Hard-coded paths that prevent reproduction elsewhere
| File | Hard-coded value | Effect |
|---|---|---|
| `scripts/lgbm_cloud/common.py` | `DATA_ROOT = /storage2/CV_Irradiance/datasets/1_Folsom` (and `CSV_PATH` from it) | every stage reads data here |
| `scripts/lgbm_cloud/run_step6.sh` | Python interpreter `/storage2/CV_Irradiance/.envs/lgbm_cloud/bin/python` | Step 6 driver |
| `scripts/lgbm_cloud/phase1/p1_*.py` | repository path and dataset paths | Phase 1 checks |
| `scripts/lgbm_cloud/benchmarks/bench_gpu_hot_path.py` | `sys.path` entry to the repository | benchmark only |
| Derived tables (`path`, `img_le_t`, `img_cnn` columns) | absolute image paths | only matters when reusing the tables on another machine; regenerating them writes local paths |
| Existing project code (`preprocessing/folsom_adapter.py` `DEFAULT_IMAGE_ROOT`, `source/...` `IMAGE_DIR`) | server paths | not used by this experiment's code paths (`image_root` is passed explicitly) |

**Proposed minimal fix (not applied):** `documentation/proposed_configurable_paths.patch`.
- It adds the environment-variable overrides `FOLSOM_DATA_ROOT` and `LGBM_PYTHON`. The defaults are identical to the current values, so behaviour without the variables is unchanged (checked: patched files compile and resolve to the same paths).
- **Consequence:** `common.py` is one of the feature-defining files hashed in `FROZEN_V1_manifest.json`. Applying the patch changes its sha256, so checkpointed extraction would refuse to run until the manifest records the change as an execution-only edit. That should happen only after re-running the 6-day checkpoint test and confirming bit-identical features.

## 6. Draft data availability statement (for adaptation)
> The Folsom irradiance, meteorological and sky-image data are publicly available from Zenodo (Pedro, Larson & Coimbra, 2019; doi:10.5281/zenodo.2826939; CC BY 4.0) and described in Pedro et al. (2019), J. Renewable Sustainable Energy 11, 036102 (doi:10.1063/1.5094494). The code, fitted feature-extraction parameters, trained models, per-sample predictions for the validation and 2016 test sets, and all result tables are available at https://github.com/The-ranmandala018/Multimodal_Solar_Irradiance (branch/tag: *to be filled*, commit *to be filled*). Derived feature tables can be regenerated from the public data with the provided scripts [and are archived at *DOI to be filled*].
