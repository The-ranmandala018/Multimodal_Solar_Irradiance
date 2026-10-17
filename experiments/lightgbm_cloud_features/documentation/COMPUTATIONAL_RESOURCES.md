# Computational resources and runtime: LightGBM cloud-feature experiment

Every value is labelled **Measured** (recorded by the code or the filesystem at run time), **Derived** (computed from measured values), **Observed** (seen interactively during the session, not logged systematically) or **Unavailable** (not recorded and not recoverable). Nothing here is estimated unless it says so.

All runs took place on 2026-10-17, local time UTC+05:30. Timestamps come from file birth times (`stat %w`, ext4) and modification times (`stat %y`) of the logs and outputs, unless stated otherwise.

## 1. Hardware and software

| Item | Value | Status / source |
|---|---|---|
| CPU | 2 × Intel Xeon Gold 5218R @ 2.10 GHz; 20 cores per socket, 2 threads per core, **80 logical CPUs**; 2 NUMA nodes | Measured (`lscpu`) |
| RAM | 251 GiB | Measured (`free -g`) |
| GPUs | 2 × NVIDIA Quadro GV100, 32 GiB each, driver 535.309.01 | Measured (`nvidia-smi`) |
| GPU use in the reported experiments | **None.** All feature extraction, training and evaluation ran on the CPU. GPU 0 was used only for a separate benchmark (§4) | Measured |
| OS | Ubuntu 22.04.5 LTS, kernel 6.8.0-138-generic | Measured (`/etc/os-release`, `uname -r`) |
| Storage | Data and outputs on `/storage2` (3.8 TB, local disk) | Measured (`df -h`) |
| Shared server | Other users' CPU and GPU jobs ran at the same time (for example, another user's training job on GPU 1) | Observed |
| Python | 3.10.21 | Measured (`versions` field in every summary and `config.json`) |
| Packages | numpy 2.2.6, pandas 2.3.3, pvlib 0.15.2, scikit-learn 1.7.2, lightgbm 4.6.0, shap 0.48.0, pyarrow 21.0.0, opencv (cv2) 5.0.0, torch 2.14.0+cu126 (not used for results), scipy 1.15.3 | Measured (`versions` in configs; scipy from `pip list`) |
| Environment | Conda env cloned from the project's `solar_gpu` env, plus lightgbm, shap and pyarrow; run with `PYTHONNOUSERSITE=1` | Recorded in `FROZEN_V1_manifest.json` |
| Source code | Working tree at commit `c635fd9` for existing code (`datasets/`, `preprocessing/`, `source/`), identical in those folders to the public `3f8b700`. New code in `scripts/lgbm_cloud/` was uncommitted at run time and is identified by sha256 in `ARTIFACT_INVENTORY.csv` and `FROZEN_V1_manifest.json` | Measured (`git_head` field; `git diff 3f8b700 c635fd9`) |

## 2. Runtime by stage

| Stage | Start → end | Wall time | Parallelism | Status / source |
|---|---|---|---|---|
| Phase 1 alignment checks (4 scripts) | not recorded | Unavailable | 1 process | Ran from a temporary directory; only the outputs (`checks/p1_*.json`) were kept |
| Corrected image index (775,916 images) | output 10:10:31 | **29.6 s** | 1 process | Measured (`features/image_index_summary.json`) |
| Sample table and Set A (656,735 samples) | 10:10:41 → 10:39:17 | **1,715.9 s** (window building 1,708 s) | 40 forked workers | Measured (`samples_setA_summary.json`, `logs/build_tabular_features.log`) |
| Step 3 early check: Ridge-A / LGBM-A / LGBM-A0 fits | log ends 10:39:44 / 10:39:52 / 10:40:00 | **10.5 s / 4.9 s / 5.0 s** (fit only) | LightGBM 32 threads | Measured (`runs/check_*/config.json` `fit_seconds`) |
| Sun-geometry check, geometry fit, obstruction masks, clear-sky library and thresholds, Gate 2 overlays | outputs 10:41:57, 10:46:58, 10:47:57–10:48:54, 11:00:07–11:00:27, 11:00:55–11:01:06 | Unavailable (only output times) | 32–48 forked workers | Output times measured; durations not recorded |
| Extraction trial (1,000 images) | – | **7.3 s** wall; **67.5 ms**/image median (p95 92.5) | 10 workers | Measured (`checks/cloud_trial_timing.json`) |
| Full extraction, launch 1 (stopped) | 11:20:34 (log birth) → about 11:22:50 | about 2 min, no output | 64 workers | Log birth measured. Observed: server load average about 4,000 from OpenCV/BLAS thread oversubscription; stopped by hand |
| Full extraction, launch 2 (killed) | about 11:23 → 11:43:25 (last log write) | reached 700/1,083 days at 1,152 s, no output | 48 workers × 1 thread | Measured (`logs/extract_cloud_features_killed_1123.log`, which reused launch 1's log file); killed by a server reboot (uptime 31 min at 18:35) |
| **Full extraction, launch 3 (reported)** | 18:35:10 → 19:00:59 | **1,510.2 s** (25.2 min) of processing; 740,265 images; **87.3 ms**/image/worker median (p95 137.1) | 48 workers × 1 thread (`cv2.setNumThreads(1)`, OMP/BLAS/MKL/NUMBA threads = 1) | Measured (`features/cloud_features_summary.json`, log birth and mtime) |
| Extraction verification | output 19:01:53 | Unavailable (under 1 min between outputs) | 1 process | Output time measured |
| Join (Step 5) | 19:02:07 → 19:02:18 | about 11 s | 1 process | Derived from output birth and mtime |
| **Step 6 pipeline, total** | **19:10:11 → 19:26:31** | **980 s (16 min 20 s)** | see below | Measured (`logs/run_step6.log` birth and mtime) |
| Search A (30 trials) | 19:10:11 → 19:13:53 | 222 s (sum of trial fits 218.4 s) | 3 searches in parallel, LightGBM 24 threads each | Measured (log times; `runs/search_A/trials.csv` `seconds`) |
| Search B (30 trials) | 19:10:11 → 19:19:16 | 545 s (sum of fits 542.4 s; trials 7.0–56.5 s) | as above | Measured |
| Search C (30 trials) | 19:10:11 → 19:14:39 | 268 s (sum of fits 264.8 s) | as above | Measured |
| Stability folds A / B / C (4 candidates × 4 folds each) | 19:19:17 → 19:20:20 / 19:21:38 / 19:20:42 | 63 s / 141 s / 85 s | 3 in parallel, 24 threads each | Measured (log birth and mtime) |
| Final runs (2 Ridge + 25 LightGBM) | 19:21:38 → 19:26:31 | 293 s; sum of fit times 412.9 s | up to 3 LightGBM runs in parallel, 24 threads each | Measured (`runs/final_*/config.json` `fit_seconds`; log times) |
| – Ridge-A / Ridge-B | – | 9.4 s / 31.2 s (fit, including α search on val) | 1 thread | Measured |
| – LightGBM A / A0 / B / C / B-parity, per seed | – | 9.4–10.4 / 7.5–8.5 / 20.6–24.5 / 14.4–17.6 / 16.5–18.9 s | 24 threads | Measured |
| Validation analysis (incl. SHAP on 30,000 rows) | outputs 20:56:00 → 20:56:17 | about 17 s after the first output; start not recorded | 1 process | Partly measured |
| Test prediction from saved models (27 runs) | 20:58:57 → 20:59:12 | **15 s** | 1 process | Measured (`logs/predict_test.log`) |
| Final test evaluation (incl. bootstraps) | 20:59:12 → 21:00:26 (log 21:00:15 → 21:00:26) | **11 s** | 1 process | Measured (`logs/evaluate_test_final.log`) |

**Derived totals (CPU wall time of the successful path):** feature preparation about 30 + 1,716 s; extraction 1,510 s; Step 6 980 s; test prediction and evaluation 26 s. That is about 70 minutes, excluding the geometry, mask and library fits, which were not timed. The two failed extraction launches consumed about 25 minutes more.

## 3. Resource utilisation

| Measurement | Status |
|---|---|
| Peak RAM per process / total | **Unavailable**: not recorded (no `/usr/bin/time -v` or memory profiler) |
| CPU utilisation over time | **Unavailable** as a time series. **Observed** snapshots during the full extraction: 50 processes using about 4,700% CPU (`ps`), load average about 53; during the stopped first launch the load average was about 4,000 |
| Per-image compute profile (extraction, 1 thread, 300 images) | **Measured** (`cProfile`): 77 ms/image; numpy reductions about 43%, `scipy.ndimage.map_coordinates` (clear-sky lookups) about 27%, other array arithmetic about 11%, JPEG reduced decode about 5%, OpenCV small ops (connected components, phase correlation, Sobel) under 3%. Recorded in `checks/gpu_benchmark_and_checkpoint_test.json` |
| Data-loading overhead | Partly measured: reduced JPEG decode about 3.5 ms/image (profile). Loading feature tables for training is not timed separately |
| Disk usage of outputs | Measured: 2.0 GB in total, of which feature tables are 1.60 GB (`ARTIFACT_INVENTORY.csv`) |
| GPU utilisation during experiments | Not applicable (no GPU use) |

## 4. GPU evaluation (benchmark only; not used for any reported result)
Source: `checks/gpu_benchmark_and_checkpoint_test.json`.

| Benchmark | CPU | GPU (Quadro GV100, GPU 0) | Outcome |
|---|---|---|---|
| Extraction hot path (colour maps, clear-sky lookups, cloud masks, cover fractions), 1,000 images | 34.1 ms/image, 1 thread | 6.5 ms/image (torch, including host-to-device transfer) | 5.3× faster, but **not bit-identical**: 25/1,000 images differ by about one boundary pixel (16/1,000 with sun angles on CPU), max \|Δ\| 1.3e-4 in cover fractions. Not adopted |
| LightGBM, 300 rounds on Set B | 7.6 s (32 threads, deterministic, a rerun is bit-identical) | OpenCL single precision 4.9 s; double precision 11.6 s | Single precision is non-deterministic (prediction Δ up to 0.0035 in k). Not adopted. The CUDA learner is not in this LightGBM build |

## 5. Computational settings
- Extraction: per-day work units, 48 forked workers, one thread each. Images read at 384 × 384 (`cv2.IMREAD_REDUCED_COLOR_4`).
- LightGBM: `deterministic=True`, `force_row_wise=True`, `num_threads` = 24 in Step 6 (32 in the Step 3 checks), `max_bin` = 255, seeds 0–4 for `seed`, `bagging_seed`, `feature_fraction_seed` and `data_random_seed`.
- Determinism evidence: trial 0 of search A reproduced the Step 3 check exactly (val k-MSE 0.023453). The saved models reproduce their validation predictions with a maximum difference of 0.0 (`summary/test_report.json`, checks).
