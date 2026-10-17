# Cloud features, version 1: frozen algorithm and parameters

Frozen on 2026-10-17 after the Gate 2 review was accepted. The full extraction
(`features/cloud_features.parquet`) uses exactly the files hashed in
`FROZEN_V1_manifest.json`. `scripts/lgbm_cloud/join_features.py` is the only listed
file written after the extraction started; it is Step 5 and is not used by extraction.

**Rule:** the masks, thresholds and geometry below are **not** tuned against
validation or test performance. A version 2 needs a written reason from the Step 7
analysis, done on validation only, and gets its own folder and manifest.

All fits use image content only, or training-period (2014-01-02 to 2015-10-04)
samples only. No irradiance from validation or test is used anywhere.

## 1. Image timestamps (`build_image_index.py`)
- File names come from a drifting camera clock. The file modified time (`st_mtime`, UTC epoch) agrees with the irradiance clock (Phase 1; Varaschin & Silva, arXiv:2503.21966).
- `t_img = file-name time − median(file-name − mtime)` over the image's **clock segment**. A segment is one day folder, split where a 15-image running median of the offset jumps by more than 60 s, i.e. where the camera clock was reset mid-day.
- Mid-day resets were found on 2014-01-26, 2015-02-03 and 2016-09-13; 1,086 segments in total.
- Per-image |t_img − mtime|: median 1 s, p99 22 s, max 408 s. The tail is isolated files with a late mtime.
- New-model image rule: the latest non-corrupted image with `t_img ≤ t` and `t − t_img ≤ 120 s`.
- Parity rule: the image `FolsomDataset` matched for the CNN (file-name time, ±120 s).

## 2. Sky geometry (`fit_sky_geometry.py` → `features/sky_geometry.json`)
- Images are read at 384 × 384 (`cv2.IMREAD_REDUCED_COLOR_4`).
- Centre: the median of the saved monthly calibration centres, (cx, cy) = (195.74, 188.47) px.
- Radial curve: `zenith = c1·r + c2·r² + c3·r³` with c = (0.62915, −5.0835e−4, −1.8089e−6).
  - Fitted on 760 visible-sun detections in **train** clear-sky samples (SZA < 75).
  - Pinned to zenith 90° at R = 195.17 px.
  - Error: median 1.52°, p90 3.75°.
  - The saved monthly power-law calibrations were off by up to about 15° in winter and are not used.
- Rotation: one `alpha` per (year, month), from sun blobs at the corrected `t_img`, using up to 300 images per month selected from image content only.
  - Range 99.7°–108.2°, median absolute spread 0.7–4.5° per month.
  - Side finding: every January in `calibration_results/` uses the source's starting value of 96.7°, not a fitted one.

## 3. Usable sky (`fit_sky_mask.py` → `features/sky_mask_<year>.npy`)
- Zenith ≤ **75°**. At 80° a shed and a pole at the bottom are still inside.
- Per-year obstruction mask: a pixel is an obstruction if its median brightness over 1,500 daytime images (SZA < 70) is below 80, beyond 40° zenith, dilated by 3 px.
- About 5% of the 80° disc is masked; the mask is identical across years, so the camera did not move.
- Saturated pixels (min RGB ≥ 250) are excluded from cloud classification.

## 4. Cloud detection (`fit_clear_sky_library.py` → `clear_sky_library.npz`, `cloud_thresholds.json`)
- Fixed red-blue-ratio thresholds do **not** work on these JPEGs: clear-sky pixel RBR ≈ 0.95, overcast ≈ 0.99.
- Clear-sky library:
  - Median of colour saturation, brightness and RBR per bin of angle to sun (10°), pixel zenith (7.5°) and solar zenith (10°).
  - Built from 3,000 strict-clear **train** images (k-based `clear` and std(k, 30 min) < 0.02).
  - Lookups interpolate linearly between bin centres.
- Rule: a pixel is cloud if `sat − lib_sat < ts` **or** `|gray − lib_gray| > tg`.
- Thresholds, set so the false-cloud rate on 600 held-out strict-clear train images, taken from different days, is 20 / 10 / 5%:

  | k | ts | tg | achieved false-cloud rate |
  |---|---|---|---|
  | 1 | −0.0653 | 17.33 | 19.9% |
  | 2 | −0.0984 | 21.65 | 10.0% |
  | 3 | −0.1292 | 26.17 | 5.0% |

  An extra cover feature uses RBR residual > 0.0935 (90% quantile on held-out clear).
- Separation at k = 2 (per-image median cloud fraction):
  - held-out clear: 0.069
  - train overcast: 0.853
  - overcast pixels within 35° of the sun detected: 77%

## 5. Features (`cloud_features.py`)
Groups:
- **cover** (`cf*`): 3 thresholds × whole sky and zenith rings 0–30 / 30–60 / 60–75°, plus `cf_rbr_all`
- **sun** (`sun_*`): saturation fractions overall and within 5° and 15° of the sun; largest saturated blob within 25° of the sun (area, offset); `sun_detected` = area ≥ 200 px. "Not detected" does not mean "occluded".
- **position** (`pos_*`): k = 2 cloud fraction in rings 0–15 / 15–30 / 30–45 / 45–70° from the sun; k = 1 fraction in 0–15°; angle from the sun to the nearest cloud pixel
- **texture** (`tx_*`): saturation, brightness and RBR statistics and their residuals, entropy, Sobel edge strength, overall and within 30° of the sun. These are image measures, not optical depth.
- **motion** (`mo_*`): see below

Motion:
- Measured on a sky plane (tan-zenith coordinates, 128 × 128, zenith ≤ 70°) by phase correlation of **successive frame differences**: frames t−2 → t−1 against t−1 → t, past frames only.
- `mo_valid = 1` only if:
  - both frame gaps are 40–150 s
  - the k = 2 cloud fraction is between 0.03 and 0.97
  - the correlation response is ≥ 0.2
- Otherwise speed, direction, residual ratio, upwind cloud fraction and time-to-sun are **NaN**.
- In the trial, valid motion directions within one day agreed to within 3–10°.

## 6. Known limitations (from the Gate 2 review; overlays kept in `checks/gate2/page_1..5.png`)
1. **False cloud on clear skies at low sun or in haze.** 5 of about 17 clear-labelled tiles score cf2 0.15–0.35. Examples: 2015-08-11 SZA 73 (0.29), 2015-06-24 SZA 60 (0.35), 2014-09-06 SZA 68 (0.15), 2015-09-10 SZA 63 (0.19), 2014-08-18 (0.17). The false cloud is usually near the horizon on the side away from the sun. Dirt and droplet spots on the dome add speckle (2015-09-21, 0.10).
2. **Overcast coverage is incomplete.** A crescent about 30–50° from the sun is often unmarked, because cloud brightness crosses the clear-sky brightness there and saturation is low in both. Uniform overcast reads cf2 ≈ 0.79–0.91, not about 1.
3. Thin cirrus is only partly detected. Raindrops on the dome are counted as cloud.
4. The 2014 Jan–Jul image-vs-k lag of about +3 min (Phase 1) is unexplained, and the geometry fit cannot correct it.

## 7. Next decision
After Steps 5–6, compare LGBM-A with LGBM-B on **validation**, with Step 7-style sky-condition, month and feature-group breakdowns on validation only. Use that to decide whether a mask-refinement round (v2) is justified. The 2016 test set stays untouched until final evaluation.

## Execution note
The first full-extraction launch (11:20, 64 workers) oversubscribed the CPUs, with load average about 4,000, because each worker also ran OpenCV/BLAS thread pools. It was stopped after about 2 minutes with no output. The relaunch sets `cv2.setNumThreads(1)` per worker and `OMP/OPENBLAS/MKL/NUMBA` threads to 1, with 48 workers. Feature computation is unchanged; see `execution_changes` in the manifest.

## Checkpointing and resume (added before any future full run)
- `extract_cloud_features.py` full runs are chunked by day folder (`checkpoint.py`).
- Each chunk is written to a temp file, fsynced, read back and validated: row count and paths equal the chunk's input, every feature column is present, and the range and motion-NaN checks of `verify_cloud_features.py` pass. Only then is it atomically renamed into `features/cloud_parts_v1/`. A ledger record in `_done/<day>.json`, holding the row count, sha256 and frozen-config id, marks it complete.
- Rerunning the same command resumes. Chunks with a matching ledger record, config id, sha256 and row count are skipped; missing, partial, corrupted or stale chunks are deleted and redone. The final file is assembled only when every chunk is complete.
- The frozen-config id is a hash of the feature-defining files (`cloud_features.py`, `sky_geometry.py`, `common.py`, image index, geometry, masks, library, thresholds). A run aborts if any of them differs from `FROZEN_V1_manifest.json`.
- Tested with an interruption, a corrupted part and a stray temp file: resumed chunks were bit-identical to the verified full extraction.

## GPU evaluation (not adopted)
See `checks/gpu_benchmark_and_checkpoint_test.json`.
- On GPU 0, the extraction hot path (colour maps, clear-sky lookups, cloud masks, cover fractions) ran 5.3× faster per image. It was **not bit-identical**: in about 2% of images one boundary pixel flips, from last-digit differences in arccos and interpolation. The end-to-end gain is at most about 1.8×, roughly 25 down to 14 min.
- LightGBM's OpenCL GPU was 1.55× faster but non-deterministic, with different predictions; double precision was slower than the CPU.
- The CPU is kept for both, so v1 stays bit-reproducible.
