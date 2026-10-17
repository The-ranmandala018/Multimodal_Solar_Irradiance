# LightGBM with sky-image cloud features: Folsom 10-min GHI forecasting

Tabular models (LightGBM, Ridge) for forecasting k(t+10) and GHI(t+10) at Folsom, CA. They combine the LSTM's irradiance and weather history with hand-crafted cloud features from the most recent sky image at or before t, on a corrected image clock.

**Final 2016 test results** (224,022 samples, scored once; seed-averaged; W/m²):

| Model | RMSE | MAE | Skill vs clear-sky persistence |
|---|---|---|---|
| LGBM-B (history + cloud features) | **64.36** | **28.38** | **20.9%** |
| CNN+LSTM (project baseline, 3 runs) | 65.51 | 32.32 | 19.5% |
| LGBM-B-parity (CNN's file-name-matched images) | 65.76 | 29.28 | 19.2% |
| LGBM-A (no images) | 67.98 | 31.11 | 16.4% |
| Weather-only LSTM (3 runs) | 68.23 | 31.16 | 16.1% |

Data source, licence (CC BY 4.0, attribution required), regeneration of the derived tables and the hard-coded paths are covered in [`documentation/DATA_AVAILABILITY.md`](documentation/DATA_AVAILABILITY.md). Full methods, validation and test results, statistics and caveats are in [`documentation/METHODS_AND_RESULTS.md`](documentation/METHODS_AND_RESULTS.md). Hardware and runtime are in [`documentation/COMPUTATIONAL_RESOURCES.md`](documentation/COMPUTATIONAL_RESOURCES.md). Every file, with size and sha256, is listed in [`documentation/ARTIFACT_INVENTORY.md`](documentation/ARTIFACT_INVENTORY.md) and `.csv`. The frozen cloud-feature configuration is in [`FROZEN_V1.md`](FROZEN_V1.md) and `FROZEN_V1_manifest.json`.

## Directory layout

| Path | Contents |
|---|---|
| `documentation/` | methods and results, computational resources, artifact inventory, publication check |
| `summary/` | all reported validation (`val_*`) and test (`test_*`) tables, bootstraps and SHAP groups |
| `publication/` | compact prediction files: `targets_<split>.parquet` + `predictions_<split>_<model>.parquet` (k̂ per seed, verbatim copies; sources and sha256 in `predictions_sources.json`) |
| `runs/search_<set>/` | hyperparameter search trials, chosen setting, stability folds |
| `runs/final_<model>_<set>[_seed<k>]/` | `config.json` (features, parameters, seed, fit time, versions), `metrics_val.json`, `feature_importance.csv`, the saved model (`model.txt` for LightGBM; `model.pkl` and pickle-free `model_params.json` for Ridge; all verified in `documentation/model_verification.json`). Original per-run prediction files stay local; their content is in `publication/` |
| `runs/check_*` | Step 3 early checks (diagnostic) |
| `checks/` | Phase 1 time-alignment evidence, geometry, mask and threshold fits, Gate 2 sample list, trial timing, GPU benchmark and checkpoint test. The Gate 2 overlay images and mask preview (derived from the sky photographs) are held out pending attribution approval |
| `features/` | small fitted parameters (geometry, masks, clear-sky library, thresholds, summaries). The large feature tables are not published (see the inventory) |
| `logs/` | run logs |

Code is in `scripts/lgbm_cloud/`. Existing project code used unchanged: `datasets/folsom_dataset.py`, `datasets/chrono.py`, `preprocessing/folsom_adapter.py` (calibration loading). The experiments ran on a working tree at commit `c635fd9`. That commit differs from the public `3f8b700` only by CNN+LSTM seed 1–2 and image-only result files and `scripts/wait_and_evaluate.sh`; no code used here differs.

## Quick check without the raw data
From the repository root (numpy, pandas and pyarrow are needed):
```bash
python scripts/lgbm_cloud/evaluate_from_publication.py
```
This recomputes every overall validation and test metric and the main bootstrap intervals from the published files, and compares them with `summary/`. The LSTM, CNN+LSTM, image-only and persistence rows use the committed `experiments/*/evaluation/test_predictions.npz`. Comparisons whose inputs are not in the checked-out repository version, such as the CNN+LSTM seed 1–2 and image-only predictions while they are unpublished, are reported as **SKIP**, never as passed. With all inputs present: 21/21 reproduced, maximum difference 7e-15.

## Full reproduction
**Requirements:**
- The Folsom dataset (Pedro et al. 2019, doi:10.5281/zenodo.2826939): images under `<root>/YYYY/MM/DD/YYYYMMDD_HHMMSS.jpg` **with their original file modification times**, which the time correction needs, and `csv_files/Folsom_irradiance_weather.csv`.
- Set `DATA_ROOT` in `scripts/lgbm_cloud/common.py`, the Python path in `scripts/lgbm_cloud/run_step6.sh`, and the dataset paths in `datasets`/`scripts` configs, which currently point to the original server.

**Environment:** Python 3.10, numpy 2.2.6, pandas 2.3.3, pvlib 0.15.2, scikit-learn 1.7.2, lightgbm 4.6.0, shap 0.48.0, pyarrow 21.0.0, opencv 5.0.0, scipy 1.15.3, plus torch (only imported by the project's dataset module). The original env was a conda clone of the project env plus `pip install lightgbm==4.6.0 shap==0.48.0 pyarrow==21.0.0`.

**Commands**, as run, from `scripts/lgbm_cloud/`. Let `PY = env PYTHONNOUSERSITE=1 PYTHONPATH=<repo> python`:

| # | Command | Output |
|---|---|---|
| 1 | `$PY phase1/p1_samples.py out`, `p1_images.py out`, `p1_lags.py out`, `p1_true_offsets.py out` (run from a temp folder) | `checks/p1_*`, `image_offsets_*`, `csv_clock_clear_days.csv` |
| 2 | `$PY build_image_index.py` | `features/image_index.parquet` (corrected `t_img`) |
| 3 | `$PY build_tabular_features.py --workers 40` | `features/samples_setA.parquet`, `windows_40x7.npy` |
| 4 | `$PY train_tabular.py --model ridge --features A --run-name check_ridge_A` (and `--model lgbm` for A and A0) | `runs/check_*` |
| 5 | `$PY check_sun_geometry.py`; `$PY fit_sky_geometry.py`; `$PY fit_sky_mask.py`; `$PY fit_clear_sky_library.py` | geometry, masks, library, thresholds |
| 6 | `$PY extract_cloud_features.py --trial --workers 10`; `$PY gate2_overlays.py` | trial timing; Gate 2 review |
| 7 | `env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 $PY extract_cloud_features.py --workers 48` | `features/cloud_features.parquet` (checkpointed: rerun the same command to resume) |
| 8 | `$PY verify_cloud_features.py`; `$PY join_features.py` | verification; `samples_B*.parquet` |
| 9 | `./run_step6.sh` (search A/B/C → stability folds → Ridge-A/B → LGBM A, B, C, A0, B-parity × seeds 0–4; validation only; resumable) | `runs/` |
| 10 | `$PY analyze_val.py`; `$PY val_b_vs_a_by_sza.py` | `summary/val_*` |
| 11 | `$PY predict_test.py`; `$PY evaluate_test_final.py` (one-time test evaluation from saved models) | `runs/*/predictions_test.parquet`, `summary/test_*` |
| 12 | `$PY publication_check.py`; `$PY build_inventory.py`; `$PY verify_models.py` | `publication/`, `documentation/` (inventory, model verification) |

Benchmarks (not used for any result) are in `scripts/lgbm_cloud/benchmarks/`.

**Determinism:** LightGBM runs with `deterministic=True` and `force_row_wise=True` and fixed seeds. A rerun reproduces predictions bit for bit on the same machine and library versions. Cloud features are bit-reproducible on the CPU; the GPU path was evaluated and rejected because it changes about 2% of images by one pixel. See `COMPUTATIONAL_RESOURCES.md` §4.
