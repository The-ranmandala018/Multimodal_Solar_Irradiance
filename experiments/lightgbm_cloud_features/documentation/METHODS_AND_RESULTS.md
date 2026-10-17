# Methods and results: LightGBM forecasting with sky-image cloud features (Folsom, 10-min GHI)

This document is written so it can be adapted into a journal methods and results section. Every number below is taken from saved outputs. The source file is named for each table, and `publication_check.json` confirms that the reported metrics match the prediction files (62/62 checks). **Validation (§6) and test (§7) results are reported separately. No test result was used for any modelling decision.**

## 1. Objective and hypotheses
Objective: measure whether hand-crafted cloud features from ground-based sky images, aligned to the irradiance clock and using only information available at forecast time t, improve 10-minute-ahead GHI forecasts over a model that uses only past irradiance, meteorology and sun geometry. Compare this with the project's deep models (weather-only LSTM, CNN+LSTM).

Hypotheses, fixed before the test evaluation:
- **H1:** cloud features improve the tabular model (LGBM-B better than LGBM-A). Decision rule, fixed in advance: an MAE gain of ≥ 1 W/m² or ≥ 1 skill point, a 95% day-block bootstrap interval excluding 0, and the gain concentrated in mixed sky.
- **H2:** a non-linear learner is needed to use the features (LGBM-B better than Ridge-B).
- **H3:** image features alone are not sufficient (LGBM-C against LGBM-A).
- **H4:** correct image-to-irradiance time alignment matters (LGBM-B better than LGBM-B-parity, which uses the CNN pipeline's file-name-matched images).

## 2. Data
- **Source:** Folsom, CA dataset (Pedro, Larson & Coimbra 2019, doi:10.5281/zenodo.2826939), 2014–2016. One-minute GHI/DNI/DHI with temperature and pressure (`Folsom_irradiance_weather.csv`, 1,552,320 rows, timestamps in UTC), and fisheye sky images at about 1-minute intervals (775,916 images).
- **Site:** 38.642° N, −121.148° E, 60 m.
- **Derived quantities** (unchanged project code, `datasets/folsom_dataset.py`):
  - solar zenith and azimuth (pvlib)
  - Ineichen clear-sky GHI
  - clear-sky index k = GHI / GHI_cs, clipped to [0, 1.2]
  - sin and cos of the hour
  - rows with SZA > 85° removed, leaving 720,660 rows
- **Samples:** a sample at time t exists when the rows at t−39 min and t+10 min exist and an image lies within ±120 s of t by file name; this is the original project rule. **656,735 samples.**
- **Target:** k(t+10). GHI forecast = k̂ · GHI_cs(t+10). Predictions are not clipped, matching the project's evaluation script.
- **Split:** chronological, by calendar year (`datasets/chrono.py`):

  | Split | Period | Samples |
  |---|---|---|
  | Train | 2014-01-02 to 2015-10-04 | 389,442 |
  | Validation | 2015-10-04 to 2015-12-31 | 43,271 |
  | Test | 2016 (365 days) | 224,022 |

  For tuning and early stopping, the first day of validation is dropped as a purge gap, leaving 42,682 samples.
- **Test subgroups:** Jan–Aug 159,369 / Sep–Dec 64,653 samples. k-based sky class (trailing 30 min of k: clear if std < 0.05 and mean > 0.9; overcast if std < 0.10 and mean < 0.5; else mixed): clear 130,866 / mixed 66,892 / overcast 26,264. The rule is the same as `scripts/analysis_bias_and_sky_conditions.py`.

## 3. Image time alignment (Phase 1, `checks/`)
- **The CSV clock is UTC:** on 187 clear days, the best fit of GHI to pvlib clear-sky GHI needs a shift of +1 min (IQR −0.5 to +1.5) (`checks/p1_lags.json`).
- **Image file names come from a drifting camera clock.** File name minus file modified time (mtime, UTC epoch):
  - −23 s to +79 s in 2014
  - about 247 → 150 s through 2015
  - about 143 → 75 s in Jan–Aug 2016
  - **about 676–690 s (11.5 min) in Sep–Dec 2016**
  
  Source: `checks/image_offsets_by_month.csv`. This agrees with Varaschin & Silva (Solar Energy; arXiv:2503.21966), who report the file-name/date-modified discrepancy for Folsom.
- **The modified time is the correct clock.** Image-brightness changes correlated with k changes peak at lag +1 min with mtime labels in every month from Aug 2014. With file-name labels the peak moves with the offset: −3 min in 2015, −11 min in Sep–Dec 2016 (`checks/image_vs_k_lag_peaks.csv`).
- **Correction:** `t_img` = file-name time − the median of (file name − mtime) within the image's **clock segment** (the day folder, split at mid-day camera-clock resets; 3 such days). Per-image |t_img − mtime|: median 1 s, p99 22 s (`features/image_index_summary.json`).
- **New-model image rule:** the latest non-corrupted image with t − 120 s ≤ t_img ≤ t. If none exists, all cloud features are NaN and `has_image` = 0. That happened for 196 test samples, 79 validation and 197 training samples (has_image fraction 0.9991 / 0.9982 / 0.9995).
- **Parity rule (B-parity):** the image the CNN pipeline used (nearest by file name, ±120 s), whatever its true capture time. On the corrected clock those images are 1.2–2.4 min old in Jan–Aug 2016 and about 11.5 min old in Sep–Dec 2016 (`checks/p1_true_offsets.json`).

## 4. Features (`scripts/lgbm_cloud/feature_sets.py`, `cloud_features.py`; frozen v1, `FROZEN_V1.md`)
Set A is built from the same 40 × 7 history window the LSTM receives, taken from `FolsomDataset.__getitem__` before normalisation:
- **k history (14):** k at lags 0, 1, 2, 5, 10, 20, 39 min; mean and std over 10, 20 and 40 min; k(t) − k(t−10)
- **meteorology (4):** temperature and pressure at t, and their change over 30 min
- **geometry and time at t (5):** SZA, sin and cos of azimuth, sin and cos of the hour
- **geometry at t+10 (2):** SZA(t+10) and GHI_cs(t+10). These are deterministic, not measured.

Cloud features (48 plus `has_image`), computed per image at 384 × 384 within the usable sky (zenith ≤ 75°, outside a per-year obstruction mask):

| Group | n | Features |
|---|---|---|
| cover | 13 | cloud fraction at 3 thresholds × (whole sky, zenith 0–30, 30–60, 60–75°); red-blue-ratio-residual cover |
| sun | 6 | saturated fraction overall and within 5° and 15° of the sun; largest saturated blob within 25° (area, offset); `sun_detected` (area ≥ 200 px) |
| position | 6 | cloud fraction in rings 0–15, 15–30, 30–45, 45–70° from the sun (threshold 2); threshold-1 fraction within 0–15°; angle from the sun to the nearest cloud |
| texture | 15 | saturation, brightness and red-blue statistics and their clear-sky residuals; entropy; Sobel edges (whole sky and within 30° of the sun) |
| motion | 8 | phase correlation of successive frame differences on a sky plane (`mo_valid`, vx, vy, speed, response, residual ratio, upwind cloud fraction, time to sun). NaN unless valid: frame gaps 40–150 s, cloud fraction 0.03–0.97, response ≥ 0.2 |

**Cloud detection:**
- A pixel is cloud if its colour saturation is below the clear-sky value by more than ts, **or** its brightness differs from the clear-sky value by more than tg.
- The clear-sky reference is a median "library" binned by angle to the sun, pixel zenith and solar zenith, built from 3,000 strict-clear **training** images.
- (ts, tg) are set so the false-cloud rate on 600 held-out clear training images is 20 / 10 / 5%.
- Fixed red-blue-ratio thresholds were rejected: in these JPEGs clear sky has RBR ≈ 0.95 and overcast ≈ 0.99.

**Sky geometry:**
- Single centre; cubic radial curve fitted on 760 sun detections in training images (median error 1.5°, p90 3.7°).
- Monthly rotation fitted from sun blobs at the corrected time, using image content only.

**Feature sets:**

| Set | Features | Count |
|---|---|---|
| A0 | k history, meteorology, geometry at t | 23 |
| A | A0 + geometry at t+10 | 25 |
| B | A + cloud features + `has_image` | 74 |
| C | geometry at t and t+10 + cloud features + `has_image` (no k, temperature or pressure) | 56 |
| B-parity | B, with cloud features from the CNN's image | 74 |

## 5. Models, tuning and protocol
- **Ridge** (scikit-learn): median imputation with missing-value indicators, standardisation, α chosen on purged validation from {1e-3 … 1e3}. Selected α: 10 for Ridge-A, 1000 for Ridge-B.
- **LightGBM 4.6.0:** L2 objective, early stopping (200 rounds) on purged validation, at most 5,000 rounds, `deterministic=True`, `force_row_wise=True`, `max_bin` = 255, `bagging_freq` = 1.
- **Search:** 30 random settings, identical for A, B and C (space seed 2026; trial 0 = defaults):
  - num_leaves 15–127
  - min_data_in_leaf 200–5,000 (log scale)
  - learning_rate 0.02–0.1 (log scale)
  - feature_fraction and bagging_fraction 0.6–0.9
  - lambda_l2 0–10
  
  The selection criterion is purged validation k-MSE. Trial 16 was best for all three sets: num_leaves 122, min_data_in_leaf 270, learning_rate 0.0204, feature_fraction 0.781, bagging_fraction 0.747, lambda_l2 6.01. A0 reuses A's settings and B-parity reuses B's (`runs/search_*/best.json`).
- **Stability check** (not used for selection): 4 time blocks within train with 1-day gaps, number of rounds fixed from validation. The chosen setting ranked first on every fold for B and C. For A it ranked last of 4 candidates on every fold, but the spread was about 0.3% of k-MSE (`runs/search_*/stability_summary.json`).
- **Seeds:** 5 per LightGBM set (0–4). Reported predictions are averaged over seeds, and per-seed spread is also reported.
- **Test protocol:** one-time scoring of the saved models (no refitting) after all modelling decisions were frozen. The saved models reproduce their validation predictions exactly (maximum difference 0.0).
- **Comparison models** are the project's saved test predictions. Each family is averaged over its runs, and per-run metrics are reported:
  - weather-only LSTM: 3 runs
  - CNN+LSTM: 3 runs (`baseline_lstm_cnn`, `cnn_lstm_seed1`, `cnn_lstm_seed2`)
  - image-only CNN: 1 run
  
  Their targets match ours sample by sample (maximum difference 6e-5 W/m²).

### Metrics and statistics
- **Accuracy:** MAE, RMSE, MBE, R², nMAE = MAE / mean(GHI), nRMSE = RMSE / mean(GHI), all on GHI in W/m² (`common.regression_metrics`).
- **Forecast skill:** FS = 1 − RMSE / RMSE_baseline (as in `scripts/compute_baselines_and_skill.py`). The baselines are clear-sky persistence, k(t)·GHI_cs(t+10), and naive persistence, GHI(t).
- **Paired differences:** day-block bootstrap. Whole days are resampled with replacement, 2,000 resamples (seed 0). The statistic is MAE(a) − MAE(b) and RMSE(a) − RMSE(b) on the same samples, with a 95% percentile interval (`analyze_val.boot_diff`). There is no correction for multiple comparisons.

## 6. Validation results (purged validation set, n = 42,682; used for selection)
Source: `summary/val_metrics_overall.csv`, `summary/val_paired_bootstrap.csv`, `summary/val_shap_groups.csv`, `summary/val_B_vs_A_by_sky_and_sza.csv`.

| Model | RMSE | MAE | MBE | R² | k-MSE | FS (clear-sky persistence) | Per-seed RMSE mean ± sd |
|---|---|---|---|---|---|---|---|
| LGBM-B | **56.16** | **28.73** | 2.01 | 0.911 | 0.0204 | 22.3% | 56.25 ± 0.04 |
| LGBM-B-parity | 57.33 | 29.37 | 1.58 | 0.907 | 0.0212 | 20.7% | 57.40 ± 0.03 |
| Ridge-B | 59.93 | 32.20 | 1.51 | 0.898 | 0.0234 | 17.1% | – |
| LGBM-A | 60.61 | 31.34 | 1.29 | 0.896 | 0.0234 | 16.2% | 60.65 ± 0.04 |
| LGBM-A0 | 60.62 | 31.48 | 1.37 | 0.896 | 0.0234 | 16.2% | 60.65 ± 0.03 |
| LGBM-C | 60.69 | 35.79 | 1.83 | 0.896 | 0.0246 | 16.1% | 60.76 ± 0.11 |
| Ridge-A | 63.07 | 33.63 | −1.29 | 0.887 | 0.0254 | 12.8% | – |
| Clear-sky persistence | 72.30 | 32.52 | −0.17 | 0.852 | – | 0% | – |

Reference: the existing models' best-checkpoint validation k-MSE on the full (unpurged) validation set is 0.0238–0.0240 for the weather-only LSTM and 0.0210–0.0215 for the CNN+LSTM.

Paired, validation (95% CI):
- LGBM-B − LGBM-A: MAE −2.60 [−3.23, −2.01], RMSE −4.46 [−5.57, −3.41]
  - by sky class, MAE: clear −0.60, mixed −5.49, overcast −2.13
- LGBM-B-parity − LGBM-B: RMSE +1.17 [+0.88, +1.50]
- LGBM-B − Ridge-B: MAE −3.46 [−4.06, −2.87]

Grouped SHAP for LGBM-B, seed 0 (30,000 validation rows, share of mean |SHAP|):
- k history 0.53
- texture 0.17
- position 0.16
- geometry/time 0.05
- motion 0.04
- cover 0.03
- sun 0.02
- meteorology 0.01

In mixed sky the cloud groups together account for 0.49. LGBM-B was better than LGBM-A in every solar-zenith band within every sky class, including clear sky at SZA > 60° (MAE −0.5 to −0.9).

## 7. Final test results (2016, n = 224,022; scored once)
Source: `summary/test_metrics_overall.csv`, `test_metrics_by_period.csv`, `test_metrics_by_sky.csv`, `test_metrics_by_sza_sky.csv`, `test_metrics_by_month.csv`, `test_paired_bootstrap.csv`, `test_haze_check.csv`, `test_metrics_per_run.csv`.

| Model | RMSE | MAE | MBE | R² | FS (clear-sky persistence) | FS (naive) | Per-run RMSE mean ± sd (n) |
|---|---|---|---|---|---|---|---|
| **LGBM-B** | **64.36** | **28.38** | 1.17 | 0.947 | **20.9%** | 26.0% | 64.47 ± 0.11 (5) |
| CNN+LSTM | 65.51 | 32.32 | −7.00 | 0.945 | 19.5% | 24.6% | 66.41 ± 0.63 (3) |
| LGBM-B-parity | 65.76 | 29.28 | 1.21 | 0.945 | 19.2% | 24.3% | 65.85 ± 0.03 (5) |
| LGBM-A0 | 67.94 | 31.16 | 1.07 | 0.941 | 16.5% | 21.8% | 67.98 ± 0.04 (5) |
| LGBM-A | 67.98 | 31.11 | 1.07 | 0.941 | 16.4% | 21.8% | 68.02 ± 0.04 (5) |
| Ridge-B | 68.09 | 33.06 | 2.49 | 0.941 | 16.3% | 21.7% | – (1) |
| Weather-only LSTM | 68.23 | 31.16 | 3.23 | 0.941 | 16.1% | 21.5% | 68.56 ± 0.18 (3) |
| LGBM-C | 68.36 | 35.54 | 0.39 | 0.941 | 16.0% | 21.4% | 68.44 ± 0.08 (5) |
| Ridge-A | 70.47 | 32.25 | 0.97 | 0.937 | 13.4% | 18.9% | – (1) |
| Image-only CNN | 70.92 | 38.21 | 4.98 | 0.936 | 12.8% | 18.4% | – (1) |
| Clear-sky persistence | 81.37 | 31.71 | 0.19 | 0.916 | 0% | 6.4% | – |
| Naive persistence | 86.93 | 45.18 | 3.82 | 0.904 | −6.8% | 0% | – |

The first three columns are the run-averaged predictions.

By period and sky class (RMSE / MAE, W/m²):

| Model | Jan–Aug | Sep–Dec | Clear | Mixed | Overcast |
|---|---|---|---|---|---|
| LGBM-B | 69.30 / 30.71 | 50.14 / 22.63 | 22.98 / 6.74 | 108.52 / 70.07 | 52.05 / 30.03 |
| LGBM-B-parity | 70.16 / 31.24 | 53.42 / 24.43 | 23.31 / 6.84 | 110.69 / 72.34 | 54.55 / 31.36 |
| LGBM-A | 72.86 / 33.43 | 54.12 / 25.38 | 23.68 / 7.34 | 114.53 / 76.94 | 56.75 / 32.81 |
| CNN+LSTM | 70.00 / 34.93 | 52.82 / 25.89 | 25.17 / 12.28 | 109.51 / 72.01 | 53.86 / 31.10 |
| Weather-only LSTM | 73.17 / 33.49 | 54.18 / 25.39 | 24.00 / 7.52 | 114.94 / 76.84 | 56.53 / 32.56 |

Paired differences on test (95% day-block CI, 365 days):

| Comparison | Subset | ΔMAE | ΔRMSE |
|---|---|---|---|
| LGBM-B − LGBM-A | all | −2.73 [−3.14, −2.33] | −3.62 [−4.36, −2.94] |
| | Jan–Aug / Sep–Dec | −2.72 / −2.75 | −3.56 / −3.98 |
| | clear / mixed / overcast | −0.59 / **−6.87** / −2.78 | −0.70 / −6.01 / −4.70 |
| LGBM-B − Ridge-B | all | −4.68 [−5.18, −4.17] | −3.73 [−4.53, −2.90] |
| LGBM-B-parity − LGBM-B | all | +0.90 [+0.72, +1.08] | +1.40 [+1.05, +1.83] |
| | Sep–Dec | +1.80 [+1.37, +2.30] | +3.28 [+2.26, +4.45] |
| LGBM-C − LGBM-A | all | +4.43 [+3.53, +5.32] | +0.38 [−0.73, +1.52] |
| | overcast | +17.82 | +15.74 |
| LGBM-A0 − LGBM-A | all | +0.06 [+0.04, +0.08] | −0.04 [−0.08, −0.00] |
| LGBM-A − Weather-only LSTM | all | −0.05 [−0.19, +0.10] | −0.25 [−0.47, −0.01] |
| LGBM-B − CNN+LSTM | all | −3.94 [−4.41, −3.50] | −1.15 [−1.77, −0.60] |
| | Jan–Aug / Sep–Dec | −4.22 / −3.26 | −0.70 [−1.41, −0.00] / −2.68 [−3.44, −1.89] |
| LGBM-B-parity − CNN+LSTM | all | −3.05 [−3.52, −2.59] | +0.26 [−0.37, +0.87] |

**Summer low-sun haze check** (`summary/test_haze_check.csv`):
- In Jun–Aug clear sky, the median cloud fraction (threshold 2) rises with solar zenith: 0.04 (≤ 50°), 0.11 (50–60°), 0.15 (60–70°), 0.22 (70–85°). The share of samples with cloud fraction > 0.15 is 0.07 / 0.31 / 0.49 / 0.76. So the known false-cloud artefact is present in the test features.
- LGBM-B is nevertheless better than LGBM-A there: MAE −0.40 [−0.70, −0.06] at 60–70° and −0.86 [−1.05, −0.68] at 70–85°. On summer clear samples with SZA > 60° and cloud fraction > 0.15 (n = 9,939): MAE −0.67 [−0.94, −0.39].
- Across all clear-sky samples with cloud fraction > 0.15 (n = 29,553): MAE −0.20 [−0.84, +0.70], so no gain and no harm.

## 8. Interpretation
- **H1 supported.** LGBM-B is better than LGBM-A in both test periods and every sky class. The gain is largest in mixed sky and meets the decision rule fixed in advance.
- **H2 supported:** LGBM-B is better than Ridge-B.
- **H3 supported:** images alone (C) are worse than A overall, and much worse in overcast, though better in mixed sky.
- **H4 supported.** File-name-matched images cost 1.4 W/m² RMSE overall and 3.3 W/m² in Sep–Dec 2016, when they are about 11.5 min old. With the same old images, LightGBM ties the CNN+LSTM on RMSE (+0.26 [−0.37, +0.87]). The RMSE advantage of LGBM-B over the CNN+LSTM (−1.15) is therefore mostly due to the time alignment, not the learner. The CNN+LSTM's larger MAE reflects its bias (MBE −7.0 W/m², clear-sky MAE 12.3 against 6.7).
- LGBM-A matches the weather-only LSTM, and the t+10 geometry features (A against A0) add nothing.

## 9. Leakage controls, limitations and caveats
1. **No future information:** features use rows ≤ t and images with t_img ≤ t (checked: image age 0–120 s). The t+10 inputs are deterministic geometry only.
2. **Parameters fitted on training data only:** cloud thresholds and the clear-sky library on training images; the radial lens curve on training sun detections; LightGBM, Ridge, imputation and scaling on train, with selection on validation.
3. **Image-only calibrations use test-period images.** The monthly rotation and the yearly obstruction mask were fitted per month or year, including 2016, from image content only (sun blobs; median brightness). No irradiance or targets were used, which mirrors the project's original calibration procedure. It is still an unsupervised use of test-period images.
4. **The test set was scored once**, after freezing; it was not used for tuning, selection, mask refinement or calibration. Any later work must treat these test numbers as already seen.
5. **Validation covers only Oct–Dec 2015.** Summer behaviour (for example haze) could not be checked before test. A single hyperparameter setting was chosen, and for set A its choice is not stable across folds, though the differences are negligible.
6. **Comparison models:**
   - The CNN+LSTM and LSTM numbers come from earlier saved predictions (3 runs, against 5 seeds here), and they used file-name-matched images (see H4).
   - The CNN's saved calibration used an unfitted rotation (96.7°) in every January (`FROZEN_V1.md` §2).
   - Family averages are ensembles, so per-run metrics are reported alongside them.
7. **The sky class is derived from past k**, which is target-related but available at t, and is used only for reporting.
8. **Cloud-mask limitations** (documented at Gate 2, `checks/gate2/`):
   - false cloud at low sun and in haze (confirmed on test in summer, without harming forecasts)
   - incomplete overcast coverage (cloud fraction ≈ 0.85 instead of ≈ 1)
   - thin cirrus partly missed
   
   Texture and cover features are image measures, not cloud optical depth.
9. **Time alignment:** relies on file mtimes, cross-checked by image–irradiance lag analysis. An unexplained +3 min lag remains in Jan–Jul 2014 (training only).
10. **Bootstrap intervals** are per comparison, with no multiplicity correction. Day blocks do not capture correlation between consecutive days.
11. **Provenance:** the extraction driver used for the reported full extraction was later extended with checkpointing. The feature code is hash-identical, and the extended driver reproduced 3,763 rows bit-identically. The driver revision used for the run is not stored as a file; its sha256 is recorded in the manifest (`FROZEN_V1.md`, `FROZEN_V1_manifest.json`).
