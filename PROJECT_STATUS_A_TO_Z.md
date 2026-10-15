# Multimodal Solar Irradiance Forecasting: Project Status, A to Z

**Audit date:** 2026-10-15 (about 10:00 local server time, +0530)
**Repository:** `/storage2/CV_Irradiance/Multimodal_Solar_Irradiance` (branch `main`, HEAD `8159b59`)
**Remote:** `https://github.com/The-ranmandala018/Multimodal_Solar_Irradiance` (`origin/main` = `8159b59`, checked with `git ls-remote` during the audit)
**Scope:** Read-only audit. No training was started, no code or results were changed, and nothing was committed or pushed. This file is the only file the audit created.

Legend: **VERIFIED** means checked against files, logs, checkpoints or git. **UNKNOWN / NEEDS VERIFICATION** means the evidence was not found. **AUDIT CALC** means a statistic computed during this audit, in memory only, from saved prediction files (nothing was written to disk).

---

## 1. Executive Summary

- **Task.** Forecast GHI 10 minutes ahead at Folsom, CA. The model predicts the clear-sky index `k(t+10)` (clipped to [0, 1.2]), and GHI is rebuilt as `k × GHI_cs(Ineichen)`. Train/val = 2014–2015 (last 10 % of those samples is validation), test = 2016. Only samples with SZA ≤ 85° and a sky image are used. **VERIFIED** (`datasets/folsom_dataset.py`, `datasets/chrono.py`).
- **Finished and evaluated on the full 2016 test set (224,022 samples):**
  - CNN+LSTM baseline, 1 run, unseeded (`experiments/baseline_lstm_cnn`): RMSE **65.936**, MAE **32.997**, R² **0.94465**, MBE **−8.730** W/m², best epoch 7, best val loss 0.021046.
  - Weather-only LSTM, 3 seeds: RMSE **68.432 / 68.476 / 68.771** (mean 68.560 ± 0.184), MAE **31.334 / 31.330 / 32.570**.
  - Persistence baselines: naive RMSE **86.926**, clear-sky persistence RMSE **81.367**.
  - Every preliminary number in the earlier project notes matches the files exactly.
- **Running right now (do not interrupt):**
  - `cnn_lstm_seed1` (GPU 0): epoch 1 done, epoch 2 at about 36k of 48,681 batches.
  - `cnn_lstm_seed2` (GPU 1): epoch 1 done, epoch 2 at about 35k of 48,681 batches.
  - `image_only_seed0` (GPU 0): started 09:53 today, epoch 1 at about 2.3k batches.
  - All three run under `scripts/train_with_watchdog.sh` in tmux, with patience **10**.
- **Scientific status.** The images help RMSE and hurt MAE and bias:
  - RMSE: CNN+LSTM is 2.5 W/m² better than weather-only seed 0. AUDIT CALC: block-bootstrap 95 % CI for the RMSE difference is [−3.38, −1.59] W/m².
  - MAE: CNN+LSTM is worse (+1.7 W/m²; CI [+1.03, +2.31]).
  - Bias: CNN+LSTM has a clear negative bias (−8.7 W/m²), mostly in clear skies.
  - The gain is concentrated in **mixed-sky** conditions (RMSE 110.1 vs 115.3). In **clear** skies the weather-only model is better.
  - The image ablation shows the trained CNN+LSTM *depends* on the images. That dependence alone does not prove the images add value.
- **The main comparability problem.** The single CNN+LSTM run differs from the weather-only runs in more than the input modality:
  - It was **unseeded**.
  - It was trained mostly with **2-GPU DDP** (effective batch 16). Its best checkpoint has `module.`-prefixed keys.
  - It was evaluated on 2 GPUs.
  - The weather-only runs were seeded, single-device, batch 8, and seed 2 was trained on CPU.
  - Firm claims therefore have to wait for `cnn_lstm_seed1/2`. Those runs also use patience 10 instead of 5, which is a further confound.
- **Git.** Nothing is unpushed. Two tracked files have uncommitted changes (they add the image-only model). Several result folders exist only locally: `weather_only_seed2` (complete with evaluation), `cnn_lstm_seed1/2`, `image_only_seed0`, `analysis/`, root `checkpoints/`.

---

## 2. Research Problem and Objectives

| Item | Finding | Evidence |
|---|---|---|
| Research question | Does adding sky images to weather/irradiance history improve 10-min-ahead GHI forecasting? | `models/multimodal_model.py`, the `uses_images` and `MODEL_TYPES` model families, `scripts/plot_model_comparison.py` docstring |
| Horizon | 10 minutes (`horizons: [10]`) | `scripts/train_gpu.py` config, `scripts/evaluate_test.py` `make_config` |
| Site | Folsom, CA (lat 38.642, lon −121.148, alt 60 m) | `train_gpu.py` config |
| Target | `k_index(t+10)` = GHI / GHI_cs (Ineichen, pvlib), clipped to [0, 1.2] | `FolsomDataset._add_solar_features`, `__getitem__` |
| Reported metric space | "GHI_clipped_target" = k_clipped × GHI_cs at t+10, in W/m². This is **not** raw measured GHI: wherever raw k > 1.2 the target is capped. | `evaluate_test.py` lines that build `y_ghi_true` |
| Planned comparisons | Naive and clear-sky persistence, weather-only, image-only (in progress), multimodal, with multiple seeds | `scripts/compute_baselines_and_skill.py`, `experiments/*` |
| Formal written objectives / README | **Not present.** The root `README.md` is empty (0 bytes). | `README.md` |

---

## 3. Project Architecture

```
Folsom fisheye JPG (UTC filename)              Folsom CSV (1-min, UTC): GHI, DNI, DHI, temperature, pressure
   │  monthly geometric camera calibration          │  pvlib: SZA, azimuth, GHI_cs (Ineichen) → k_index, sin/cos hour
   │  (cx, cy, R, p, az_sign, az_alpha)              │  keep SZA ≤ 85°
   ▼                                                 ▼
remap to azimuth–zenith grid 180×720            40-step × 7-feature history (t−39 … t), train-only z-score
   → 10 channels (x, y, z, θ, φ, R, G, B, intensity, solid-angle weight)
   ▼                                                 ▼
FolsomCNNEncoder (4 conv blocks) → 256-D       BaselineTemporalEncoder (1-layer LSTM) → 128-D
                         └──── ConcatenationFusion → 384-D ────┘
                                         ▼
                 BaselineRegressionHead (LayerNorm → Linear 256 → GELU → Dropout 0.3 → Linear 1)
                                         ▼
                                   k̂(t+10) → ĜHI = k̂ × GHI_cs(t+10)
```

- Image preprocessing runs **on the fly** on the CPU in DataLoader workers. No `.npy` cache is used (`preprocessing/folsom_adapter.py` docstring).
- The original source file `source/Folsom_2014_DAYLIGHT_INDEPENDENT_P_RESUME(1).py` is loaded through `importlib` and is not modified.
- The CPU-side preprocessing is the throughput bottleneck. Epochs take about 3,450 s for 2-GPU DDP and about 8,100–8,300 s for single GPU with 2 jobs running at once (`training_history.csv` files).

---

## 4. Repository Structure and Important Files

| Path | Purpose | Git state |
|---|---|---|
| `datasets/folsom_dataset.py` | CSV loading, solar features, image matching (±120 s), sample construction, train-only normalisation, DataLoaders (DDP-aware) | tracked |
| `datasets/chrono.py` | `chronological_year_split`: 2014–15 train/val (last 10 % = val), 2016 test | tracked |
| `preprocessing/folsom_adapter.py` | `FolsomPreprocessor`: reuses saved monthly calibrations, falls back to in-memory calibration | tracked |
| `preprocessing/projection.py` | (129600, 10) → (10, 180, 720) tensor | tracked |
| `source/Folsom_2014_DAYLIGHT_INDEPENDENT_P_RESUME(1).py` | Original calibration / projection pipeline (2,649 lines) | tracked |
| `models/visual_encoder.py`, `temporal.py`, `fusion.py`, `heads.py` | CNN, LSTM, concatenation fusion, regression head | tracked |
| `models/multimodal_model.py` | `cnn_lstm`, `weather_only`, and (uncommitted) `image_only` models | tracked, **modified** |
| `scripts/train_gpu.py` | Training entry point (resume, DDP, AMP, early stopping, `--seed`, `--model`) | tracked |
| `scripts/evaluate_test.py` | Interruption-safe test evaluation (1 or 2 GPUs), writes `test_metrics.csv` and `test_predictions.npz` | tracked |
| `scripts/compute_baselines_and_skill.py` | Persistence baselines and forecast skill → `full_metrics_summary.csv` | tracked |
| `scripts/image_ablation_test.py`, `run_image_ablation_gpu1.sh`, `plot_image_ablation.py` | Image ablation (7 modes) | tracked |
| `scripts/analysis_bias_and_sky_conditions.py` | Validation-estimated bias correction and sky-class breakdown | tracked |
| `scripts/plot_model_comparison.py` | Weather-only vs CNN+LSTM comparison figure and CSV | tracked |
| `scripts/auto_train_watchdog.sh`, `auto_evaluate_watchdog.sh` | **Legacy** watchdogs hard-wired to `baseline_lstm_cnn`. Do not reuse. | tracked |
| `scripts/train_with_watchdog.sh` | Per-experiment, seeded training watchdog that writes `DONE` | tracked, **modified** (comment only) |
| `scripts/wait_and_run_weather_only.sh`, `scripts/wait_for_gpu.sh` | Wait-for-free-GPU helpers | **untracked** |
| `scripts/calibrate_*.py`, `plot_azimuth_calibration.py`, `plot_calibration_summary.py` | Camera calibration and its plots | tracked |
| `scripts/*smoke_test.py` | Model, real-data and GPU smoke tests | tracked |
| `calibration_results/Folsom_{2014,2015,2016}_monthly_calibrations.json` | 36 monthly camera calibrations | tracked |
| `calibration_results/Folsom_2014_01_monthly_calibration.json` | Early single-month (Jan 2014) calibration file | **untracked** |
| `analysis/calibration_summary/` | Calibration summary CSVs and PNGs (2014–2016). `analysis/azimuth_calibration/` is empty. | **untracked** |
| `checkpoints/multimodal_gpu_best.pt` | Early CNN+LSTM checkpoint (epoch 1, val 0.02533, dated 2026-09-30). Superseded. | **untracked, NOT ignored** |
| `.agents/`, `.codex/` | Empty folders | not tracked |
| `report/` | Ignored by `.gitignore`. Not present locally. | ignored |

---

## 5. Dataset and Preprocessing Pipeline

All rows below are **VERIFIED** from `datasets/folsom_dataset.py` unless stated otherwise.

| Step | Detail |
|---|---|
| Raw data | `/storage2/CV_Irradiance/datasets/1_Folsom/csv_files/Folsom_irradiance_weather.csv` (columns `Datetime, GHI, DNI, DHI, temperature, pressure`, 1-min). Images in `/storage2/CV_Irradiance/datasets/1_Folsom/YYYY/MM/DD/YYYYMMDD_HHMMSS.jpg`. |
| Time zone | Naive timestamps are treated as UTC by pvlib. Consistent evidence: the CSV starts at `2014-01-02 08:00` with GHI = 0 (local midnight), image names such as `20160615_000000.jpg` are daylight in PDT, and the source has `TIMESTAMP_TIMEZONE = "UTC"`. |
| Cleaning | Drop rows missing GHI, temperature or pressure. Keep SZA ≤ 85°. |
| Features (7) | `k_index, temperature, pressure, SZA, Azimuth, sin_hour, cos_hour`. DNI and DHI are not used. |
| History | 40 steps at 60 s (t−39 … t). Built by reindexing with `method="nearest", tolerance=10min`, then ffill/bfill, then 0. A sample is kept only if `t−39 min` exists. Gaps inside the window are filled, not rejected. No future leakage: t itself is in the index, so "nearest" never selects a row after t. |
| Image match | Nearest image within ±120 s of t. Samples without an image are dropped **for all models**, so weather-only uses the same sample set (`load_images=False` only skips loading). |
| Target | `k_index(t+10)` from the un-normalised `target_df`, clipped to [0, 1.2]. t+10 must exist. |
| Normalisation | z-score fitted on training-sample rows only (rows at sample times t, not the full history windows). It is applied to the feature dataframe used for all splits. No test leakage. |
| Split | Chronological. 2014+2015 → train (first 90 %) / val (last 10 %, so the validation period is late 2015: autumn/winter. *Exact date range: NEEDS VERIFICATION*). 2016 → test. |
| Sample counts | Train ≈ 389,448 (48,681 batches × 8; the last batch may be partial, so the exact count is *NEEDS VERIFICATION*). Val = **43,271** (`experiments/analysis/bias_correction.csv`, `val_n`). Test = **224,022** (all `test_metrics.csv`). |
| Image representation | Monthly camera calibration → azimuth–zenith remap (180 × 720) → 10 channels. 6 of the 10 channels (x, y, z, θ, φ, solid-angle weight) are geometric constants identical for every image. Only R, G, B and intensity vary. |

---

## 6. Model Architectures and Inputs/Outputs

| Model key | Class | Inputs used | Encoders | Head input | Params (state-dict keys) | Evidence |
|---|---|---|---|---|---|---|
| `cnn_lstm` | `MultimodalSolarForecastModel` | image (10×180×720) and weather (40×7) | CNN → 256, LSTM → 128 | 384 | 36 keys | `models/multimodal_model.py` |
| `weather_only` | `WeatherOnlyForecastModel` | weather only (the image is a placeholder `zeros(1)`) | LSTM → 128 | 128 | 10 keys | same, plus checkpoints |
| `image_only` | `ImageOnlyForecastModel` (**uncommitted**) | image only | CNN → 256 | 256 | 32 keys | `git diff models/multimodal_model.py`, `image_only_seed0` mid checkpoint |

- Output: one value, k̂(t+10). Predictions are not clipped. AUDIT CALC: 0.03 % of CNN+LSTM test predictions are negative; clipping them changes RMSE by less than 0.001 W/m².
- Loss: MSE on k. Optimiser: AdamW, lr 1e-4, no LR scheduler. AMP: enabled only on CUDA.

---

## 7. Training Configuration and Reproducibility

Source: `scripts/train_gpu.py` (VERIFIED).

| Aspect | Implementation |
|---|---|
| Max epochs | `--epochs` (default 30). All runs used 30. |
| Early stopping | `--patience` (default 5), `--min-delta 1e-4`. Improvement means `val < best − 1e-4`. |
| Seeds | `--seed` seeds Python, NumPy, torch and CUDA before the model is built (added in commit `db511ff`). Each epoch's shuffle seed is drawn from torch's RNG. Not bit-exact: cuDNN and AMP are not deterministic, and DataLoader worker seeds are not explicitly controlled. |
| GPU selection | Single process uses `cuda` (= first visible device). The watchdog sets `CUDA_VISIBLE_DEVICES=$GPU_ID`. DDP runs via `torchrun --ddp`. |
| Checkpoints | `best_model.pt` (on improvement), `last_checkpoint.pt` (every epoch), `mid_epoch_checkpoint.pt` (every 100 batches, deleted at epoch end). |
| What is saved | model, optimizer, AMP scaler (last/mid only; not saved in best), RNG states (torch, CUDA, Python, NumPy), epoch, batch index, running loss, epoch shuffle seed, best val loss, best epoch, patience counter. |
| Mid-epoch resume | Reseeds with the saved `epoch_seed` to rebuild the same shuffle order, then skips the batches already done. At most 99 batches are recomputed. Not supported under DDP while a mid checkpoint exists (explicit `RuntimeError`). |
| Atomic saves | **No.** `torch.save` writes directly to the target path. A crash during a save could corrupt `last_checkpoint.pt` or `best_model.pt`. (Evaluation shards *are* atomic: `.tmp.npz` then `os.replace`.) |
| Resume reproducibility | Restores model, optimizer, scaler and RNG state, but DataLoader worker RNG and dropout streams after a restart do not exactly match an uninterrupted run. Results are statistically equivalent, not identical. |
| History / log overwrite | A fresh start (no `--resume`) truncates `training_history.csv`. `train_with_watchdog.sh` always passes `--resume` when a checkpoint exists, refuses `baseline_lstm_cnn`, refuses to start if a `DONE` exists, and refuses to start when a `train_gpu.py` for the same `--experiment-dir` is already running. Separate seeds use separate folders. |
| Concurrent jobs | Separate experiment folders, so there is no file collision. `cnn_lstm_seed1` and `image_only_seed0` **share GPU 0** and all three jobs share the CPUs used for image preprocessing. This slows them down but does not corrupt them. |
| Legacy watchdog hazard | `scripts/auto_train_watchdog.sh` is hard-wired to `baseline_lstm_cnn` and to `pgrep -f "python scripts/train_gpu.py"`. While any other training runs it just sleeps. If launched when no training is running, it would resume the original baseline and train an extra epoch (as noted in the `db511ff` message). **Do not run it.** |

### 7.1 Per-run settings actually used (VERIFIED from logs and checkpoints)

| Run | Seed | Device | Effective batch | Patience | AMP | Evaluated on |
|---|---|---|---|---|---|---|
| `baseline_lstm_cnn` | **none** (predates `--seed`) | Epoch 1 single GPU, then **2-GPU DDP** (`best_model.pt` keys start with `module.`; commit `23fe0a1`) | 8 in epoch 1, then **16** (8 per GPU) | 5 | yes | 2 GPUs (rank_0 and rank_1 shards) |
| `weather_only_seed0` | 0 | single GPU (Quadro GV100) | 8 | 5 | yes | 1 rank (device not printed) |
| `weather_only_seed1` | 1 | single GPU | 8 | 5 | yes | **CPU** (log warns "no accelerator") |
| `weather_only_seed2` | 2 | **CPU** ("Device: cpu" in `auto_training.log`) | 8 | 5 | **no** | 1 rank (device not printed) |
| `cnn_lstm_seed1` / `seed2` | 1 / 2 | single GPU (0 / 1) | 8 | **10** | yes | not yet |
| `image_only_seed0` | 0 | single GPU 0 | 8 | **10** | yes | not yet |

> **Note.** The header of `train_with_watchdog.sh` says its recipe is "identical to the original baseline run (single GPU, batch 8…)". That is **not accurate**: the original baseline was mostly trained with 2-GPU DDP and was unseeded.

---

## 8. Complete Experiment Inventory

| Experiment | Model / inputs | Seed | Last epoch | Best epoch | Best val loss (MSE on k) | Status | Valid ckpt | Test eval | In GitHub? |
|---|---|---|---|---|---|---|---|---|---|
| `checkpoints/multimodal_gpu_best.pt` | cnn_lstm | unknown | 1 | 1 | 0.025329 | Early trial, superseded. Origin and log: NEEDS VERIFICATION. | yes (best only) | no | **no** (untracked, not ignored) |
| `experiments/baseline_lstm_cnn` | cnn_lstm (image + weather) | none | 12 | 7 | 0.021046 | **Completed**: early stop at epoch 12 (`last_checkpoint.pt`: `epochs_without_improvement=5`). No training log survives (`*.log` is ignored and no `auto_training.log` is present locally). | yes | **yes**, full 224,022 | Results yes; checkpoints no (ignored) |
| `experiments/weather_only_seed0` | weather_only | 0 | 9 | 4 | 0.023885 | **Completed**: early stop at 9 ("Training finished normally" 2026-10-14 00:11) | yes | **yes** | Results yes; `DONE` untracked |
| `experiments/weather_only_seed1` | weather_only | 1 | 11 | 6 | 0.023777 | **Completed**: early stop at 11 (2026-10-14 11:38) | yes | **yes** | Results yes (commit `8159b59`); `DONE` untracked |
| `experiments/weather_only_seed2` | weather_only (trained on CPU) | 2 | 10 | 5 | 0.024029 | **Completed**: early stop at 10 (2026-10-14 19:25). Evaluation finished 2026-10-15 05:45. | yes | **yes** | **no** (whole folder untracked) |
| `experiments/cnn_lstm_seed1` | cnn_lstm | 1 | 1 complete; epoch 2 at batch 36,400 (mid ckpt) | 1 | 0.022100 | **Running** (PID 27349, GPU 0, tmux `cnn_seed1`) | yes (epoch 1) | no | no |
| `experiments/cnn_lstm_seed2` | cnn_lstm | 2 | 1 complete; epoch 2 at batch 35,000 (mid ckpt) | 1 | 0.022592 | **Running** (PID 27350, GPU 1) | yes (epoch 1) | no | no |
| `experiments/image_only_seed0` | image_only | 0 | epoch 1 at batch 2,300 (mid ckpt) | none yet | none | **Running** (PID 60707, GPU 0, started 2026-10-15 09:53). Depends on **uncommitted** model code. | mid only | no | no |
| `experiments/baseline_lstm_cnn/evaluation/image_ablation_subset20000/` | ablation of the baseline checkpoint | subset seed 0 | n/a | n/a | n/a | **Completed**: 7 modes. An empty `.lock` file is left over (harmless flock target). | n/a | 20k subset | CSVs yes; `.lock` untracked |
| `experiments/comparison` | weather_only_seed0 vs baseline | n/a | n/a | n/a | n/a | **Completed** (uses seed 0 only) | n/a | n/a | yes |
| `experiments/analysis` | bias correction and sky-class breakdown (baseline vs weather seed 0) | n/a | n/a | n/a | n/a | **Completed** | n/a | n/a | yes |

The `cnn_lstm_seed1/2` watchdogs show three restarts of the watchdog itself (05:49, 06:02, 06:14). No "exited with code" lines were logged, which suggests the restarts were manual. The runs then resumed from mid-epoch checkpoints at batches 1,300, about 4,200–4,300, and 5,000. This is the expected resume behaviour.

---

## 9. Results Table with Verified Metrics

Full 2016 test set, n = 224,022. GHI units are W/m² on the clipped target. Values come from `*/evaluation/test_metrics.csv`, `full_metrics_summary.csv` and `comparison/model_comparison.csv`. MBE values come from the CSVs or are AUDIT CALC where marked.

| Model | Best ep. | Val loss | GHI MAE | GHI RMSE | GHI R² | MBE | nRMSE | k MAE | k RMSE | k R² |
|---|---|---|---|---|---|---|---|---|---|---|
| Naive persistence | n/a | n/a | 45.182 | 86.926 | 0.90381 | +3.822 | n/a | n/a | n/a | n/a |
| Clear-sky persistence | n/a | n/a | **31.711** | 81.367 | 0.91572 | +0.192 | n/a | n/a | n/a | n/a |
| Weather-only seed 0 | 4 | 0.023885 | 31.334 | 68.432 | 0.94038 | +0.892 | 0.1462 | 0.06720 | 0.13166 | 0.81082 |
| Weather-only seed 1 | 6 | 0.023777 | 31.330 | 68.476 | 0.94031 | +2.222 (AUDIT CALC) | 0.1463 | 0.06817 | 0.13128 | 0.81190 |
| Weather-only seed 2 | 5 | 0.024029 | 32.570 | 68.771 | 0.93979 | +6.560 (AUDIT CALC) | 0.1469 | 0.06913 | 0.13211 | 0.80950 |
| **Weather-only, mean ± sd (3 seeds)** | n/a | n/a | 31.745 ± 0.715 | 68.560 ± 0.184 | 0.94016 | n/a | n/a | n/a | n/a | n/a |
| Weather-only 3-seed ensemble (AUDIT CALC) | n/a | n/a | 31.155 | 68.234 | n/a | n/a | n/a | n/a | n/a | n/a |
| **CNN+LSTM baseline (1 run)** | 7 | **0.021046** | 32.997 | **65.936** | **0.94465** | **−8.730** | 0.1409 | 0.07096 | 0.12729 | 0.82317 |

Target mean = 468.045 W/m² (identical in all files, so all runs share the same samples).

Forecast skill FS = 1 − RMSE / RMSE_ref (`model_comparison.csv`):

| | vs clear-sky persistence | vs naive persistence |
|---|---|---|
| CNN+LSTM | 0.1896 | 0.2415 |
| Weather-only seed 0 | 0.1590 | 0.2128 |

### 9.1 Verification of the preliminary notes

| Note value | File value | Status |
|---|---|---|
| WO s0: 224,022 / MAE 31.334 / RMSE 68.432 / R² 0.94038 / ep 4 / val 0.023885 | identical | VERIFIED |
| CNN+LSTM: 224,022 / MAE 32.997 / RMSE 65.936 / R² 0.94465 / MBE −8.730 / ep 7 / val 0.021046 | identical | VERIFIED |
| Naive RMSE 86.926 / clear-sky RMSE 81.367 | identical | VERIFIED |

### 9.2 Are the weather-only and CNN+LSTM results directly comparable?

**Comparable in these respects (VERIFIED):**
- Same test samples (224,022, same target mean). The saved targets match sample by sample after undoing the 2-rank ordering of the baseline. AUDIT CALC: targets match within 0.05 W/m², and they do *not* match without reordering.
- Same target, same clipping, same normalisation, same split, same loss, lr, max epochs and patience (5).

**Not comparable in these respects:**
- Baseline unseeded vs seeded.
- Baseline DDP (effective batch 16, DistributedSampler shuffling) vs single device (batch 8).
- `weather_only_seed2` trained on CPU without AMP.
- Only **one** CNN+LSTM run, so seed variance for the multimodal model is unknown.
- Validation loss under DDP is averaged over padded shards. The difference is small, but best-epoch selection is not strictly identical.

**Cautions for downstream analysis:**
- `test_predictions.npz` stores **no timestamps**. Any per-sample analysis must undo the rank interleaving. `analysis_bias_and_sky_conditions.py` does this correctly (`load_saved_predictions`). Any new script must do the same.
- Naive persistence uses **raw** GHI(t) as the prediction, while the target is clipped. This is a small inconsistency in favour of the models.

---

## 10. Baseline Comparison

| Metric | Clear-sky pers. | Weather-only (3-seed mean) | CNN+LSTM | Reading |
|---|---|---|---|---|
| RMSE | 81.37 | 68.56 | **65.94** | CNN+LSTM is best: −2.6 W/m² vs the weather-only mean, which is about 14 times the weather-only seed SD (0.18). |
| MAE | **31.71** | 31.74 | 33.00 | CNN+LSTM is **worst**: clear-sky persistence and weather-only beat it. |
| R² | 0.916 | 0.940 | **0.945** | Follows RMSE. |
| MBE | +0.19 | +0.9 to +6.6 | **−8.73** | CNN+LSTM under-predicts systematically. |

AUDIT CALC, paired comparison of CNN+LSTM vs weather-only seed 0. This is a moving-block bootstrap with 600-sample contiguous blocks (about 1 day; the day boundaries are approximate because the npz has no timestamps) and 2,000 resamples.
- ΔRMSE 95 % CI: **[−3.38, −1.59]** W/m².
- ΔMAE 95 % CI: **[+1.03, +2.31]** W/m².

Both differences are robust within this single test year, but they point in **opposite directions**:
- CNN+LSTM reduces large errors (RMSE is dominated by cloudy and ramp events).
- CNN+LSTM increases typical small errors, largely through its negative bias in clear skies.

---

## 11. Calibration and Bias Analysis

### 11.1 What "calibration" means in this project (VERIFIED)

It is **geometric sky-camera calibration**, not irradiance or forecast calibration. Per month and per year, the code fits:
- the sky-circle centre and radius (cx, cy, R),
- the radial projection `θ = 90·r_norm^p`,
- the azimuth mapping `az_sign·(φ_img − α)`,

by detecting the sun in the images and matching it to the pvlib solar position. Source functions: `optimize_p_independently`, `optimize_azimuth_from_previous` and `select_calibration_points` in `source/Folsom_2014_…py`. The fitted parameters build the azimuth–zenith remap.

| File | Content |
|---|---|
| `calibration_results/Folsom_2014_monthly_calibrations.json`, `_2015_`, `_2016_` | 12 months each, with quality metrics. These are what the adapter actually uses. |
| `calibration_results/Folsom_2014_01_monthly_calibration.json` (**untracked**) | Single January 2014 result. `p = 0.36`, `α = 96.7`, `cx = 802.65`, `cy = 780.92`, R = 801.26. No quality metrics. The values match January in the 2014 master file except for rounding of p (0.36000000000000004). **It is not used by the adapter**, which loads only `Folsom_{year}_monthly_calibrations.json`. It covers one month only. |
| `analysis/calibration_summary/folsom_yearly_calibration_summary.csv` (**untracked**) | Mean azimuth RMSE 9.98° / 8.60° / 8.61°, mean radial RMSE 13.6° / 14.1° / 14.0°, mean p 0.578 / 0.641 / 0.605 for 2014 / 2015 / 2016 |

**Calibration observations:**
- **Leakage:** the calibration uses only images and astronomical sun positions, never irradiance targets, so there is no *target* leakage. However, the 2016 calibrations are fitted on 2016 (test-year) images. That is transductive use of unlabeled test inputs. It is acceptable if stated in the paper, but a strict protocol would freeze a calibration fitted on 2014–2015.
- **Physical plausibility:** the projection exponent p swings seasonally from 0.21 (Dec 2014) to 1.07 (Jun 2015). Radial RMSE is 11–18°. A fixed fisheye lens should not change its projection with the season, so the radial fit is probably confounded with seasonal sun-elevation coverage. The image features therefore differ month to month in a systematic way. Whether this harms the forecasts is **UNKNOWN / NEEDS VERIFICATION**.
- January α is fixed to the constant `JANUARY_AZ_ALPHA = 96.7` every year rather than being optimised (`folsom_adapter.calibrate_month`, and identical values in all three JSONs).

### 11.2 Forecast-bias analysis (VERIFIED, `experiments/analysis/bias_correction.csv`)

The bias is estimated on **validation** data only (n = 43,271) and then subtracted from the test predictions. There is no test leakage (`analysis_bias_and_sky_conditions.py`, function `predict_validation`).

| Model | Val bias | Test MAE before → after | RMSE before → after | MBE before → after |
|---|---|---|---|---|
| CNN+LSTM | −4.91 | 32.997 → **31.008** | 65.936 → **65.467** | −8.73 → −3.82 |
| Weather-only s0 | +0.06 | 31.334 → 31.339 | 68.432 → 68.431 | +0.89 → +0.84 |

**Bias findings:**
- CNN+LSTM under-predicts on both validation and test. The test bias is larger (seasonal shift: the validation period is late 2015).
- After a simple constant correction, CNN+LSTM beats weather-only s0 on **both** MAE (31.01 vs 31.33) and RMSE.
- This correction is a post-hoc fix chosen after looking at the test results. It should be reported as such, or built into the protocol up front (for example as a k-space or SZA-dependent correction fitted on validation).

---

## 12. Sky-Condition and Image-Ablation Analysis

### 12.1 Sky-condition breakdown

Source: `experiments/analysis/sky_condition_breakdown.csv` and `sky_condition_rmse.png`. Sky classes use the trailing 30-min k: clear = std < 0.05 and mean > 0.9; overcast = std < 0.10 and mean < 0.5; mixed = everything else. These are ad-hoc thresholds, not the Stein Variability Index.

RMSE in W/m², with MBE in brackets.

| Class (n, share) | Naive | Clear-sky pers. | Weather-only s0 | CNN+LSTM |
|---|---|---|---|---|
| Clear (130,866; 58 %) | 37.26 | 25.82 | **23.95** (+0.37) | 25.50 (**−8.08**) |
| Mixed (66,892; 30 %) | 145.85 | 139.86 | 115.31 (+0.15) | **110.08** (−12.43) |
| Overcast (26,264; 12 %) | 57.96 | 57.68 | 56.74 (+5.39) | **54.59** (−2.55) |

**Sky-condition findings:**
- The images help in mixed skies (−5.2 RMSE) and overcast skies (−2.2).
- In clear skies the images hurt: weather-only and clear-sky persistence are better there (clear-sky persistence MAE 7.23 and weather-only MAE 7.49, vs CNN+LSTM MAE 12.98). This is the source of the worse overall MAE and the negative bias.
- The class thresholds are not sensitivity-tested. Only weather seed 0 is in this table.

### 12.2 Image ablation

Source: baseline checkpoint, 20,000-sample random subset (seed 0), `image_ablation_metrics_subset20000.csv`.

| Mode | RMSE | Δ vs real | MAE | MBE |
|---|---|---|---|---|
| real | 67.63 | n/a | 33.22 | −8.88 |
| stale10 (image from 10 min earlier) | 71.06 | +3.42 (+5.1 %) | 34.90 | −8.13 |
| stale30 | 74.24 | +6.60 | 36.51 | −8.09 |
| stale60 | 77.08 | +9.44 | 38.13 | −8.63 |
| sametime (same clock time, other day ±7 d) | 116.01 | +48.37 | 68.72 | −12.78 |
| shuffled (random other test image) | 124.45 | +56.82 | 78.91 | −15.36 |
| zero image | 162.19 | +94.56 | 134.49 | −76.60 |

**What the ablation demonstrates:**
- The trained CNN+LSTM relies heavily on the image branch: wrong-sky images make it much worse than even naive persistence.
- The model is sensitive to image freshness, in a monotonic way (stale10 < stale30 < stale60).

**What the ablation does not demonstrate:**
- That images add value over weather. Removing an input from a model trained with it is out-of-distribution. The fair test is the separately trained weather-only model (Section 10) and the pending image-only model.
- The ablation is on a subset, has no confidence intervals, and covers only the baseline checkpoint.

Possibly useful observation: stale10 costs only 3.4 W/m². This suggests that most of the image value is "current cloud state" rather than fine-grained motion. This is a hypothesis, not a finding.

---

## 13. Training Logs, Checkpoints, and Resume Status

| Experiment | Logs present | Checkpoints present | Resume state |
|---|---|---|---|
| baseline_lstm_cnn | `training_history.csv` only. **No training log** survives locally. The evaluation and ablation logs exist. | best (ep 7), last (ep 12) | Finished. Do **not** resume. |
| weather_only_seed0/1/2 | `auto_training.log`, `test_evaluation.log`, history, plot; `wait_and_run.log` (s1) | best, last | Finished (DONE files dated 2026-10-14 00:11 / 11:38 / 19:25, consistent with "Training finished normally" in the logs) |
| cnn_lstm_seed1 | `auto_training.log`, history (1 epoch) | best (ep 1), last (ep 1), mid (ep 2, batch 36,400) | Running, resumable |
| cnn_lstm_seed2 | same | best (ep 1), last (ep 1), mid (ep 2, batch 35,000) | Running, resumable |
| image_only_seed0 | `auto_training.log`, empty history | mid (ep 1, batch 2,300) | Running, resumable |

**Live processes at audit time (VERIFIED with `ps` and `nvidia-smi`; not touched):**

| PID | Job | GPU | GPU memory |
|---|---|---|---|
| 27330 / 27349 | `train_with_watchdog.sh cnn_lstm_seed1 1 30 10` | 0 | about 1.27 GB |
| 27333 / 27350 | `train_with_watchdog.sh cnn_lstm_seed2 2 30 10` | 1 | about 1.27 GB |
| 60697 / 60707 | `train_with_watchdog.sh image_only_seed0 0 30 10` | 0 | about 1.26 GB |

**Expected duration:** one CNN+LSTM epoch currently takes about 8,100–8,300 s (about 2.3 h). With patience 10, each run needs at least 11 epochs and at most 30, so roughly 1–3 more days per run. image_only_seed0 will be similar or slower. These durations are an estimate.

---

## 14. GitHub Versus Local State

**Remote:** `origin/main` = local `HEAD` = `8159b59` ("Add weather-only seed 1 results"). **There are no unpushed commits.** The earlier observation of "ahead by 2" is now outdated. Those two commits were most likely `dc9ce5b` and `8159b59` (the latest two); which two they were exactly is UNKNOWN.

| Category | Items |
|---|---|
| Committed and pushed | All source code, 3 calibration JSONs, baseline results (metrics, predictions, history, ablation CSVs, log and plots), weather_only s0/s1 results, comparison, analysis CSV/PNG |
| Uncommitted changes to tracked files | `models/multimodal_model.py` (+`ImageOnlyForecastModel`, the `image_only` key; **the running `image_only_seed0` depends on it**), `scripts/train_with_watchdog.sh` (comment only) |
| Untracked, important results | `experiments/weather_only_seed2/` (complete: metrics, predictions, history, plot), `experiments/cnn_lstm_seed1/`, `experiments/cnn_lstm_seed2/`, `experiments/image_only_seed0/` (in progress), `experiments/weather_only_seed{0,1}/DONE`, `experiments/baseline_lstm_cnn/plots/training_validation_loss.png`, `analysis/calibration_summary/*` |
| Untracked, scripts | `scripts/wait_and_run_weather_only.sh`, `scripts/wait_for_gpu.sh` |
| Untracked, probably not needed | `calibration_results/Folsom_2014_01_monthly_calibration.json` (duplicate of Jan 2014; keep locally), `.../image_ablation_subset20000/.lock` (empty), empty `.agents/` and `.codex/` |
| Ignored (correctly) | `experiments/**/checkpoints/`, `rank_*_progress.npz`, `*.tmp.npz`, `*.log` (except force-added files such as `image_ablation_subset20000.log`) |
| **Not ignored but should not be committed** | `checkpoints/multimodal_gpu_best.pt` (7.3 MB): a blind `git add .` would commit it |
| Sizes | Repo objects about 19.6 MB loose. Each `test_predictions.npz` is about 2.3–2.8 MB, which is acceptable for git. Experiment checkpoints total about 73 MB and are ignored. |

---

## 15. Completed Work (VERIFIED)

1. Camera calibration for all 36 months of 2014–2016, with quality metrics and summary plots (commits `aec1187` … `29ff6d0`, `f9300ec`).
2. In-memory preprocessing adapter and multimodal dataset with chronological split and train-only normalisation (`7be4d40`, `c2e50c5`, `73c77cb`).
3. CNN+LSTM model, an early abandoned Mamba temporal encoder (`d6a190c` replaced by `64b4d73`), smoke tests.
4. Resumable training (mid-epoch), watchdogs, DDP (`7e2f9c4` … `23fe0a1`).
5. Baseline CNN+LSTM training (12 epochs, best 7) and full test evaluation (`c5f0bb5` … `4ff30a8`).
6. Persistence baselines and forecast skill (`74ad3e2`, `92db959`).
7. Image ablation: 7 modes on a 20k subset (`c49c15e`, `6f6a7c9`, `d9fa8be`, `05eacb4`, `e1a9065`).
8. Seeded per-experiment training (`db511ff`). Weather-only model (`bfc21ab`).
9. Weather-only seeds 0, 1 and 2: trained and fully evaluated (seed 2 is local only).
10. Comparison plot (`ebff54c`). Validation-based bias correction and sky-class breakdown (`dc9ce5b`).

## 16. Partially Completed Work

| Item | State |
|---|---|
| CNN+LSTM seeds 1, 2 | Training, epoch 2 of up to 30 (patience 10) |
| Image-only seed 0 | Training, epoch 1. Model code uncommitted. |
| Comparison and analysis outputs | Use only weather seed 0 and the single baseline run. They need regenerating with all seeds (to new filenames). |
| Image ablation | 20k subset only, baseline checkpoint only |
| Documentation | Root `README.md` is empty. There is no methods write-up in the repo. |

## 17. Known Problems and Risks

| # | Problem / risk | Severity | Evidence |
|---|---|---|---|
| 1 | Baseline CNN+LSTM is unseeded and DDP-trained (effective batch 16), unlike every other run. This confounds the modality comparison. | High | `best_model.pt` `module.` keys, `23fe0a1`, `db511ff` |
| 2 | The new CNN+LSTM and image-only seeds use **patience 10**, while the baseline and weather-only seeds used 5 | High (for fair comparison) | `ps` args, `auto_training.log` |
| 3 | `weather_only_seed2` trained on CPU without AMP | Low–Medium | `auto_training.log` "Device: cpu" |
| 4 | Running `image_only_seed0` depends on uncommitted code. If that code is reverted, the run cannot be evaluated or reproduced. | High | `git diff` |
| 5 | Checkpoint saves are non-atomic, so a crash during a save could corrupt a best or last checkpoint | Medium | `train_gpu.py` `torch.save(..., path)` |
| 6 | `test_predictions.npz` has no timestamps, and the rank order differs between 1- and 2-GPU evaluation | Medium | `evaluate_test.py`; AUDIT CALC |
| 7 | The legacy `auto_train_watchdog.sh` / `auto_evaluate_watchdog.sh` would modify `baseline_lstm_cnn` if started | Medium | script contents |
| 8 | `checkpoints/` (root) is not ignored | Low | `git check-ignore` |
| 9 | Two jobs share GPU 0 and all share the CPUs, which makes epochs slow (about 2.3 h) | Low (performance) | `nvidia-smi`, history |
| 10 | No original training log for the baseline | Low | `baseline_lstm_cnn/logs/` |
| 11 | The calibration exponent p varies by season, a sign of possible geometric miscalibration | Medium (scientific) | calibration JSONs |
| 12 | The "GHI" metrics are on the clipped target, not raw measured GHI | Medium (reporting) | `evaluate_test.py` |

## 18. Research Limitations

- One site (Folsom), one test year (2016), one horizon (10 min).
- A single multimodal run so far, so multimodal seed variance is unknown.
- The validation period is late 2015 only. Model and epoch selection is biased toward winter conditions.
- Weather features are mostly the irradiance history (k) plus T, P and geometry. There is no humidity, wind, cloud or NWP data.
- Simple architecture (small CNN, concatenation fusion). The CNN sees 6 constant geometry channels.
- The sky-class thresholds are ad hoc. There are no significance tests in the repo (only the bootstrap computed during this audit).
- Calibration is fitted on test-year images (transductive).
- Metrics are on the clipped target. Night and SZA > 85° are excluded, as is typical but worth stating.

## 19. Exact Remaining Tasks

1. **Wait** for `cnn_lstm_seed1`, `cnn_lstm_seed2` and `image_only_seed0` to write `DONE`. Verify `Training finished normally` in each `auto_training.log` and check `training_history.csv`.
2. **Commit the image-only model code now** (it does not interrupt the running jobs), so the running experiment is reproducible.
3. Evaluate each finished run with `scripts/evaluate_test.py` into its own `evaluation/` folder (1 GPU, so the output stays in time order).
4. Decide the patience problem: either (a) also run weather-only with patience 10 (seeds 0–2, about 6 h each on GPU), or (b) report best-epoch results and note that the best epochs found so far (4–7) are well below 5 + best. Recommended: (a), or confirm via the histories that no CNN+LSTM best epoch occurs later than best + 5.
5. Regenerate the comparison, bias and sky-condition analyses using **all seeds** of every model (mean ± sd, paired bootstrap per day), writing to **new** output names so existing results are not overwritten. This needs a small script change (the model lists are hard-coded to seed 0 and the baseline).
6. Run the image ablation on the new seeded CNN+LSTM checkpoints (at least real, shuffled, stale10).
7. Add timestamps to saved predictions in future evaluations, or save a separate index file.
8. Write `README.md` and a methods section, including the protocol caveats from Sections 9.2 and 11.
9. Optional or scientific: freeze the calibration from 2014–15 and test its sensitivity; drop the constant geometry channels; try SZA-dependent bias correction fitted on validation; add a second test period or a cross-year split.

## 20. Prioritised Next-Step Roadmap

| Priority | Task | Needs GPU? | Blocks |
|---|---|---|---|
| P0 | Do not touch the running jobs. Commit `models/multimodal_model.py` and `scripts/train_with_watchdog.sh`, plus `weather_only_seed2` results and the two wait scripts (see Section 21). | no | reproducibility |
| P0 | Add `checkpoints/` to `.gitignore` | no | safe commits |
| P1 | When the seeds finish: evaluate CNN+LSTM s1/s2 and image-only s0 | yes (about 1–2 h each) | all claims |
| P1 | Multi-seed comparison table and paired day-block bootstrap (CPU) | no | claims |
| P2 | Resolve the patience mismatch (weather-only patience-10 reruns, or a justified argument) | yes / cheap | fairness |
| P2 | Sky-condition and bias analysis across seeds. Validation-fitted bias correction as a pre-registered step. | CPU | paper |
| P3 | Ablation on the new checkpoints. Calibration sensitivity. Image-only interpretation. | yes | depth |
| P3 | README, methods and limitations | no | sharing |

## 21. Recommended Commands for Future Work

Run these from the repo root. None of these were executed during the audit.

```bash
# Monitor (read-only)
tail -f experiments/cnn_lstm_seed1/logs/auto_training.log
cat experiments/cnn_lstm_seed{1,2}/logs/training_history.csv experiments/image_only_seed0/logs/training_history.csv
nvidia-smi;  tmux ls
ls experiments/*/DONE

# Git: preserve code and finished results, without touching running outputs
echo "checkpoints/" >> .gitignore
git add .gitignore models/multimodal_model.py scripts/train_with_watchdog.sh \
        scripts/wait_and_run_weather_only.sh scripts/wait_for_gpu.sh
git commit -m "Add image-only baseline model; ignore root checkpoints/"
git add experiments/weather_only_seed2/evaluation/test_metrics.csv \
        experiments/weather_only_seed2/evaluation/test_predictions.npz \
        experiments/weather_only_seed2/logs/training_history.csv \
        experiments/weather_only_seed2/plots/training_validation_loss.png \
        experiments/baseline_lstm_cnn/plots/training_validation_loss.png
git commit -m "Add weather-only seed 2 results"
git status    # review before pushing
git push origin main
# Do NOT add experiments/cnn_lstm_seed*/ or image_only_seed0/ until they finish.

# After a run has DONE: single-GPU evaluation (time-ordered output)
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 PYTHONPATH=$PWD \
  ~/miniconda3/envs/solar_gpu/bin/python -u scripts/evaluate_test.py \
  --checkpoint experiments/cnn_lstm_seed1/checkpoints/best_model.pt \
  --output-dir experiments/cnn_lstm_seed1/evaluation \
  > experiments/cnn_lstm_seed1/logs/test_evaluation.log 2>&1

# Optional fairness rerun (weather-only, patience 10), in a new folder
MODEL=weather_only GPU_ID=1 bash scripts/train_with_watchdog.sh weather_only_p10_seed0 0 30 10

# Never run (they target the original baseline folder):
#   scripts/auto_train_watchdog.sh   scripts/auto_evaluate_watchdog.sh
```

## 22. File Paths and Evidence Supporting Each Conclusion

| Conclusion | Evidence |
|---|---|
| Target is clipped k(t+10). GHI is rebuilt with Ineichen GHI_cs. | `datasets/folsom_dataset.py` `_add_solar_features`, `__getitem__`; `scripts/evaluate_test.py` |
| Split is 2014–15 / 2016, last 10 % = val | `datasets/chrono.py`, `get_data_loaders` |
| Train-only normalisation | `get_data_loaders` (`train_mask`) |
| Test n = 224,022 and val n = 43,271 | `*/evaluation/test_metrics.csv`, `experiments/analysis/bias_correction.csv` |
| Baseline metrics, ep 7, val 0.021046 | `experiments/baseline_lstm_cnn/evaluation/test_metrics.csv`, `best_model.pt` metadata |
| Baseline trained under DDP and unseeded | `best_model.pt` (`module.` keys), commits `23fe0a1`, `f8b0d9e`, `db511ff` (seed added later) |
| Baseline early-stopped at epoch 12 | `logs/training_history.csv`, `last_checkpoint.pt` (`epoch=12`, `epochs_without_improvement=5`) |
| Weather-only seed metrics and epochs | `experiments/weather_only_seed{0,1,2}/evaluation/test_metrics.csv`, `auto_training.log`, checkpoint metadata |
| WO s2 trained on CPU; WO s1 evaluated on CPU | `weather_only_seed2/logs/auto_training.log`, `weather_only_seed1/logs/test_evaluation.log` |
| Persistence baselines | `experiments/baseline_lstm_cnn/evaluation/full_metrics_summary.csv`, `scripts/compute_baselines_and_skill.py` |
| Forecast skill | `experiments/comparison/model_comparison.csv` |
| Bias correction with no test leakage | `experiments/analysis/bias_correction.csv`, `scripts/analysis_bias_and_sky_conditions.py` |
| Sky-condition results | `experiments/analysis/sky_condition_breakdown.csv` |
| Ablation results | `experiments/baseline_lstm_cnn/evaluation/image_ablation_metrics_subset20000.csv`, `image_ablation_subset20000.log` |
| Running jobs, patience 10, GPU sharing | `ps`/`nvidia-smi` at audit time; `experiments/{cnn_lstm_seed1,cnn_lstm_seed2,image_only_seed0}/logs/auto_training.log` |
| Mid-epoch progress | `mid_epoch_checkpoint.pt` metadata (`batch_idx` 36,400 / 35,000 / 2,300) |
| Camera calibration method and values | `source/Folsom_2014_…py`, `preprocessing/folsom_adapter.py`, `calibration_results/*.json`, `analysis/calibration_summary/*.csv` |
| Paired bootstrap, seed mean ± sd, ensemble, prediction reordering | AUDIT CALC (in memory, from the four `test_predictions.npz` files) |
| Git state | `git status`, `git log`, `git diff`, `git ls-remote origin`, `git check-ignore`, `.gitignore` |
